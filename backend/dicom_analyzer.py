import os
import re
import json

from groq import Groq

MODEL = "llama-3.3-70b-versatile"

# Internal/binary fields that never belong in the clinical context handed to
# the AI — the parser's job was to extract everything; this is where we let
# the model decide what's clinically meaningful, not us.
SKIP_KEYS = {"image_jpeg_bytes", "has_image", "all_text_values", "machine_type"}

# Purely identifying fields — if this is all that's present alongside
# SKIP_KEYS, there's no clinical data to analyze.
PATIENT_META_KEYS = {"patient_name", "patient_id", "patient_dob"}

MAX_VALUE_LEN = 200  # skip long binary-ish strings when building context

NO_DATA_RESPONSE = {
    "value_analysis": [],
    "abnormal_count": 0,
    "clinical_pattern": None,
    "urgency": "ROUTINE",
    "urgency_reason": "No clinical measurements found",
    "action": "Ensure DICOM file contains clinical data",
    "summary": "No clinical measurements were extracted.",
}


def _load_env_fallback():
    """Best-effort load of backend/.env if GROQ_API_KEY isn't already set."""
    if os.environ.get("GROQ_API_KEY"):
        return
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


def _build_values_text(extracted):
    lines = []
    for k, v in extracted.items():
        if k in SKIP_KEYS:
            continue
        if v is None:
            continue
        s = str(v).strip()
        if s == "":
            continue
        if len(s) >= MAX_VALUE_LEN:
            continue
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _has_only_patient_metadata(extracted):
    for k, v in extracted.items():
        if k in SKIP_KEYS or k in PATIENT_META_KEYS:
            continue
        if v is None:
            continue
        s = str(v).strip()
        if s == "" or len(s) >= MAX_VALUE_LEN:
            continue
        return False
    return True


def _build_prompt(machine_type, patient_age, patient_sex, values_text):
    return f"""You are a senior consultant ophthalmologist reviewing a DICOM file from an ophthalmic imaging device.

Device: {machine_type}
Patient: {patient_age}y {patient_sex}

ALL DATA EXTRACTED FROM DICOM FILE:
{values_text}

Your task:
1. Identify which values are clinically meaningful measurements (ignore technical DICOM metadata like UIDs, timestamps, pixel spacing, image dimensions etc)

2. For each clinically meaningful measurement, use your medical knowledge to determine if it is NORMAL, BORDERLINE, or ABNORMAL
   Do not ask for reference ranges — use your ophthalmology training

3. Identify any clinical patterns from the combination of findings

4. Assign urgency based on findings:
   ROUTINE — all normal
   MONITOR — borderline findings
   REFER   — abnormal findings
   URGENT  — critical findings

5. Give one specific clinical action

Return ONLY valid JSON, no other text:
{{
  "value_analysis": [
    {{
      "measurement": "human readable name",
      "value": "value with unit",
      "status": "NORMAL|BORDERLINE|ABNORMAL|INFO",
      "clinical_meaning": "one sentence max"
    }}
  ],
  "abnormal_count": 0,
  "clinical_pattern": "pattern name or null",
  "urgency": "ROUTINE|MONITOR|REFER|URGENT",
  "urgency_reason": "one sentence",
  "action": "one specific clinical sentence",
  "summary": "2-3 sentence clinical summary"
}}

Important rules:
- Use INFO status for non-clinical metadata you include for completeness
- Only include measurements the doctor would care about in value_analysis
- Skip DICOM UIDs, pixel dimensions, transfer syntax, and other technical fields
- If this is an AFIO ScanAI export, the dr_grade, glaucoma_cdr, hr_probability fields are AI screening results — interpret them clinically
- Temperature should be 0.1 for consistency"""


def _extract_json(text):
    """Parse a JSON object out of a model response, tolerating any stray
    text (markdown fences, commentary) around the actual JSON."""
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None


def analyze_dicom(extracted_dict):
    machine_type = extracted_dict.get("machine_type", "Unknown")
    patient_age = extracted_dict.get("patient_age", "Unknown")
    patient_sex = extracted_dict.get("patient_sex", "Unknown")

    values_text = _build_values_text(extracted_dict)

    if not values_text.strip() or _has_only_patient_metadata(extracted_dict):
        return dict(NO_DATA_RESPONSE)

    _load_env_fallback()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"error": "GROQ_API_KEY not set in environment"}

    try:
        prompt = _build_prompt(machine_type, patient_age, patient_sex, values_text)
    except Exception as e:
        return {"error": f"Failed to build prompt: {e}"}

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL,
            max_tokens=1200,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
    except Exception as e:
        return {"error": f"Groq API call failed: {e}"}

    result = _extract_json(content)
    if result is None:
        return {"error": "Failed to parse Groq response as JSON", "raw": content}

    return result


if __name__ == '__main__':
    import sys
    extracted = json.loads(sys.stdin.read())
    result = analyze_dicom(extracted)
    print(json.dumps(result, indent=2))

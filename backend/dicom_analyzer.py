import os
import json

from groq import Groq

MODEL = "llama-3.3-70b-versatile"

SKIP_KEYS = {"image_jpeg_bytes", "has_image", "patient_name", "patient_id", "dob"}

URGENCY_COLORS = {
    "ROUTINE": "#10B981",
    "MONITOR": "#F59E0B",
    "REFER": "#EF4444",
    "URGENT": "#7F1D1D",
}


def get_urgency_color(urgency):
    return URGENCY_COLORS.get(str(urgency or "").upper(), "#F59E0B")


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


def _flatten_values(extracted_dict):
    """Build a flat 'label: value' list from the extracted DICOM dict, skipping
    identifying/non-clinical fields."""
    lines = []

    def walk(prefix, value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in SKIP_KEYS:
                    continue
                walk(k, v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(f"{prefix}[{i}]", v)
        else:
            if value is None or value == "":
                return
            lines.append(f"{prefix}: {value}")

    for key, val in extracted_dict.items():
        if key in SKIP_KEYS:
            continue
        walk(key, val)

    return lines


def _build_prompt(extracted_dict):
    machine_type = extracted_dict.get("machine_type", "Unknown")
    sex = extracted_dict.get("sex") or "unknown sex"
    age = extracted_dict.get("age") or extracted_dict.get("patient_age")
    age_str = f"{age}y " if age else ""

    value_lines = _flatten_values(extracted_dict)
    values_text = "\n".join(value_lines) if value_lines else "(no measurements extracted)"

    prompt = f"""You are a senior consultant ophthalmologist.
You received an ophthalmic machine report.

Machine: {machine_type}
Patient: {age_str}{sex}

ALL MEASUREMENTS:
{values_text}

Using your clinical knowledge determine for each measurement if it is NORMAL, BORDERLINE
or ABNORMAL. Identify clinical patterns.
Do not ask for reference ranges — use your medical training.

Return JSON with:
{{
  "value_analysis": [
    {{
      "measurement": str,
      "value": str,
      "status": "NORMAL|BORDERLINE|ABNORMAL",
      "clinical_meaning": str (one sentence)
    }}
  ],
  "abnormal_count": int,
  "clinical_pattern": str or null,
  "urgency": "ROUTINE|MONITOR|REFER|URGENT",
  "urgency_reason": str,
  "action": str (one specific sentence),
  "summary": str (2-3 sentences for doctor)
}}"""
    return prompt


def analyze_dicom(extracted_dict):
    _load_env_fallback()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"error": "GROQ_API_KEY not set in environment"}

    try:
        prompt = _build_prompt(extracted_dict)
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

    try:
        result = json.loads(content)
    except Exception as e:
        return {"error": f"Failed to parse Groq response as JSON: {e}", "raw": content}

    return result


if __name__ == '__main__':
    import sys, json
    extracted = json.loads(sys.stdin.read())
    result = analyze_dicom(extracted)
    print(json.dumps(result, indent=2))

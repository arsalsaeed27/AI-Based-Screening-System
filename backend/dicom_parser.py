import io
import sys
import json

import pydicom
import pydicom.multival
import pydicom.valuerep
import pydicom.uid
from PIL import Image

# VRs the generic tag loop pulls in automatically. Numeric-ish VRs plus the
# text VRs that commonly carry clinical labels/measurements on non-AFIO
# machines. SQ is handled separately (recursed), pixel data is always
# skipped, and everything else (OW, UN, AT, dates/times/person-names that
# are already captured by the fixed standard-tag block below) is ignored.
NUMERIC_VRS = {"DS", "IS", "FL", "FD", "SL", "SS", "UL", "US", "OB", "OF"}
TEXT_VRS = {"LO", "LT", "ST", "SH", "CS", "UT", "UI"}
INCLUDE_VRS = NUMERIC_VRS | TEXT_VRS

PIXEL_DATA_TAG = (0x7FE0, 0x0010)

# Fixed, always-extracted patient/study tags. Keys use snake_case regardless
# of what pydicom calls the tag, so downstream code has a stable contract.
STANDARD_TAGS = {
    "patient_name": (0x0010, 0x0010),
    "patient_id": (0x0010, 0x0020),
    "patient_dob": (0x0010, 0x0030),
    "patient_sex": (0x0010, 0x0040),
    "patient_age": (0x0010, 0x1010),
    "study_date": (0x0008, 0x0020),
    "study_time": (0x0008, 0x0030),
    "modality": (0x0008, 0x0060),
    "manufacturer": (0x0008, 0x0070),
    "manufacturer_model": (0x0008, 0x1090),
    "institution": (0x0008, 0x0080),
    "study_description": (0x0008, 0x1030),
    "series_description": (0x0008, 0x103E),
    "laterality": (0x0020, 0x0060),
}

MAX_BINARY_LEN = 256  # bytes; longer OB/OF blobs become a placeholder, not raw text
MAX_MULTIVALUE_LEN = 64  # items; longer multi-valued elements get truncated


def _coerce_value(value):
    """Convert a pydicom element value into something JSON-serializable
    (str, int, float, or a list of those). Returns None if there's nothing
    usable (empty string, empty sequence, undecodable binary, etc.)."""
    if value is None:
        return None

    if isinstance(value, (pydicom.multival.MultiValue, list, tuple)):
        items = list(value)
        truncated = len(items) > MAX_MULTIVALUE_LEN
        if truncated:
            items = items[:MAX_MULTIVALUE_LEN]
        coerced = [_coerce_value(v) for v in items]
        coerced = [c for c in coerced if c is not None]
        if not coerced:
            return None
        if truncated:
            coerced.append(f"...(+{len(value) - MAX_MULTIVALUE_LEN} more)")
        return coerced[0] if len(coerced) == 1 else coerced

    if isinstance(value, bytes):
        if len(value) == 0:
            return None
        if len(value) > MAX_BINARY_LEN:
            return f"<binary {len(value)} bytes>"
        for enc in ("utf-8", "latin-1"):
            try:
                text = value.decode(enc).strip("\x00").strip()
                if text:
                    return text
                break
            except Exception:
                continue
        return value.hex()

    if isinstance(value, pydicom.valuerep.PersonName):
        try:
            return str(value)
        except Exception:
            return None

    if isinstance(value, pydicom.valuerep.IS):
        try:
            return int(value)
        except Exception:
            return str(value)

    if isinstance(value, pydicom.valuerep.DSfloat):
        try:
            return float(value)
        except Exception:
            return str(value)

    if isinstance(value, pydicom.uid.UID):
        return str(value)

    if isinstance(value, (int, float, str)):
        return value

    try:
        text = str(value).strip()
        return text if text else None
    except Exception:
        return None


def _safe_str(ds, tag, default=None):
    """Fetch a tag by (group, element), coerced to a plain string. Never raises."""
    try:
        if tag not in ds:
            return default
        value = ds[tag].value
        if value is None:
            return default
        coerced = _coerce_value(value)
        if coerced is None:
            return default
        if isinstance(coerced, list):
            return ", ".join(str(c) for c in coerced)
        return str(coerced)
    except Exception:
        return default


def _private_tag(ds, group, element):
    try:
        tag = (group, element)
        if tag not in ds:
            return None
        return _coerce_value(ds[tag].value)
    except Exception:
        return None


def _detect_machine_type(manufacturer, model):
    text = f"{manufacturer or ''} {model or ''}".lower()

    if "topcon" in text and "triton" in text:
        return "Topcon Triton OCT"
    if "topcon" in text and "oct" in text:
        return "Topcon 3D OCT-2000"
    if ("carl zeiss" in text or "zeiss" in text) and "humphrey" in text:
        return "Humphrey HFA"
    if "heidelberg" in text:
        return "Heidelberg Spectralis"
    if "optovue" in text:
        return "Optovue AVANTI"
    if "optos" in text:
        return "Optos Ultra-Widefield"
    if "ziemer" in text or "galilei" in text:
        return "Galilei Corneal Topography"
    if "afio scanai" in text:
        return "AFIO ScanAI Export"

    combined = " ".join(p for p in ((manufacturer or "").strip(), (model or "").strip()) if p)
    return combined or "Unknown"


def _extract_afio_scanai(ds):
    """Friendly field names for AFIO ScanAI's own private block (0099,10xx),
    kept for backward compatibility with scans this system itself exported.
    The generic tag loop below also picks these up under tag_0099_10xx keys;
    this just gives the AI/UI nicer labels for its own data."""
    data = {}
    field_map = {
        "dr_grade": 0x1001,
        "dr_severity_label": 0x1002,
        "dr_confidence": 0x1003,
        "cdr": 0x1010,
        "glaucoma_risk_level": 0x1011,
        "hr_probability": 0x1020,
        "hr_detected": 0x1021,
        "triage_level": 0x1030,
        "scan_id": 0x1031,
    }
    for key, elem in field_map.items():
        value = _private_tag(ds, 0x0099, elem)
        if value is not None:
            data[key] = value
    return data


def _extract_image_jpeg_bytes(ds):
    try:
        if "PixelData" not in ds:
            return None
        arr = ds.pixel_array
        img = Image.fromarray(arr)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception:
        return None


def _parse_sequence(seq_elem, prefix, out, all_text_values, depth=0, max_depth=4):
    """Recurse into a Sequence (SQ) element, flattening each item's tags into
    out with keys prefixed by the sequence name and item index."""
    if depth > max_depth:
        return
    seq_name = seq_elem.keyword or f"tag_{seq_elem.tag.group:04x}_{seq_elem.tag.element:04x}"

    try:
        items = seq_elem.value
    except Exception:
        return

    for i, item in enumerate(items):
        item_prefix = f"{prefix}{seq_name}[{i}]"
        try:
            iterator = iter(item)
        except Exception:
            continue
        for elem in iterator:
            try:
                if (elem.tag.group, elem.tag.element) == PIXEL_DATA_TAG:
                    continue
                if elem.VR == "SQ":
                    _parse_sequence(elem, f"{item_prefix}.", out, all_text_values, depth + 1, max_depth)
                    continue
                if elem.VR not in INCLUDE_VRS:
                    continue
                key = elem.keyword or f"tag_{elem.tag.group:04x}_{elem.tag.element:04x}"
                value = _coerce_value(elem.value)
                if value is None:
                    continue
                out[f"{item_prefix}.{key}"] = value
                if isinstance(value, str) and value.strip():
                    all_text_values.append(value.strip())
            except Exception:
                continue


def parse_dicom_file(filepath):
    """Extract everything usable from any DICOM file, from any machine.
    Returns a flat, JSON-serializable dict (image bytes excepted — see the
    CLI entry point below)."""
    try:
        ds = pydicom.dcmread(filepath, force=True, stop_before_pixels=False)
    except Exception as e:
        return {"error": f"Failed to read DICOM file: {e}"}

    result = {}

    # 2. Fixed standard patient/study tags
    for key, tag in STANDARD_TAGS.items():
        value = _safe_str(ds, tag, None)
        if value:
            result[key] = value

    if not result.get("laterality"):
        # Some machines use Image Laterality (0020,0062) instead of Laterality (0020,0060)
        alt = _safe_str(ds, (0x0020, 0x0062), None)
        if alt:
            result["laterality"] = alt

    machine_type = _detect_machine_type(result.get("manufacturer"), result.get("manufacturer_model"))
    result["machine_type"] = machine_type

    # 3 + 4. Every other tag, generically, plus recursion into sequences
    all_text_values = []
    try:
        for elem in ds:
            try:
                if (elem.tag.group, elem.tag.element) == PIXEL_DATA_TAG:
                    continue
                if elem.VR == "SQ":
                    _parse_sequence(elem, "", result, all_text_values)
                    continue
                if elem.VR not in INCLUDE_VRS:
                    continue

                key = elem.keyword or f"tag_{elem.tag.group:04x}_{elem.tag.element:04x}"
                value = _coerce_value(elem.value)
                if value is None:
                    continue

                # Don't clobber the fixed standard-tag keys above.
                if key not in result:
                    result[key] = value
                if isinstance(value, str) and value.strip():
                    all_text_values.append(value.strip())
            except Exception:
                continue
    except Exception:
        pass

    # 6. Image, if present
    result["has_image"] = False
    image_bytes = _extract_image_jpeg_bytes(ds)
    if image_bytes:
        result["image_jpeg_bytes"] = image_bytes
        result["has_image"] = True

    # 7. AFIO ScanAI friendly field names (no-op for non-AFIO files — the
    # private tags just won't be present)
    try:
        for key, value in _extract_afio_scanai(ds).items():
            if key not in result:
                result[key] = value
    except Exception:
        pass

    # 8. Drop empties/None (image_jpeg_bytes stays for now — the CLI entry
    # point below strips it before JSON output; see note there)
    result = {k: v for k, v in result.items() if v is not None and v != ""}

    # 9. Flat list of every non-empty string value, for AI analysis
    seen = set()
    deduped = []
    for v in all_text_values:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    result["all_text_values"] = deduped

    return result


if __name__ == '__main__':
    result = parse_dicom_file(sys.argv[1])
    # image_jpeg_bytes is raw bytes, not JSON-serializable, and is only ever
    # consumed via dicom_image_extractor.py (which re-reads the .dcm file
    # directly) — strip it here rather than in parse_dicom_file() so the
    # function itself still returns the full in-memory result to Python
    # callers.
    result.pop('image_jpeg_bytes', None)
    print(json.dumps(result, indent=2, default=str))

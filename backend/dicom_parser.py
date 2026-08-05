import io
import sys
import json

import pydicom
from PIL import Image


def _safe_get(ds, tag, default=None):
    """Fetch a tag's value by (group, element) tuple or keyword, never raising."""
    try:
        if tag not in ds:
            return default
        value = ds[tag].value
        if value is None:
            return default
        return value
    except Exception:
        return default


def _safe_str(ds, tag, default=""):
    value = _safe_get(ds, tag, None)
    if value is None:
        return default
    try:
        return str(value)
    except Exception:
        return default


def _private_tag(ds, group, element):
    try:
        tag = (group, element)
        if tag not in ds:
            return None
        return ds[tag].value
    except Exception:
        return None


def _detect_machine_type(manufacturer, model):
    text = f"{manufacturer or ''} {model or ''}".lower()

    if "afio" in text:
        return "AFIO ScanAI"
    if "topcon" in text:
        return "Topcon Triton / 3D OCT"
    if "ziemer" in text or "galilei" in text:
        return "Galilei Corneal Topography"
    if "zeiss" in text or "humphrey" in text:
        return "Humphrey Visual Field"
    if "heidelberg" in text or "spectralis" in text:
        return "Heidelberg Spectralis"
    if "optovue" in text or "avanti" in text:
        return "Optovue Avanti"
    if "optos" in text:
        return "Optos Ultra-Widefield"
    return "Unknown"


def _extract_topcon(ds):
    data = {}
    try:
        block = ds.private_block(0x0099, "AFIO ScanAI")
        cdr = block[0x10].value if 0x10 in block else None
        if cdr is not None:
            data["cdr"] = str(cdr)
    except Exception:
        pass

    tag_map = {
        "cdr": (0x0099, 0x1010),
        "rnfl_total": (0x0099, 0x1040),
        "rnfl_symmetry": (0x0099, 0x1041),
        "disc_area": (0x0099, 0x1042),
        "rim_area": (0x0099, 0x1043),
    }
    for key, (group, elem) in tag_map.items():
        try:
            value = _private_tag(ds, group, elem)
            if value is not None and key not in data:
                data[key] = str(value)
        except Exception:
            pass

    # Standard ophthalmic tags group 0x0022
    ophthalmic_tag_map = {
        "rnfl_total_std": (0x0022, 0x1930),
        "disc_area_std": (0x0022, 0x1932),
    }
    for key, (group, elem) in ophthalmic_tag_map.items():
        try:
            value = _safe_get(ds, (group, elem))
            if value is not None:
                data[key] = str(value)
        except Exception:
            pass

    return data


def _extract_galilei(ds):
    data = {}
    field_map = {
        "sim_k": (0x0046, 0x1010),
        "flat_sim_k": (0x0046, 0x1011),
        "steep_sim_k": (0x0046, 0x1012),
        "kmax": (0x0046, 0x1013),
        "cct": (0x0046, 0x1020),
        "thinnest_cornea": (0x0046, 0x1021),
        "wtw": (0x0046, 0x1030),
        "aqd": (0x0046, 0x1031),
    }
    for key, (group, elem) in field_map.items():
        try:
            value = _private_tag(ds, group, elem)
            if value is not None:
                data[key] = str(value)
        except Exception:
            pass
    return data


def _extract_humphrey(ds):
    data = {}
    field_map = {
        "md": (0x0024, 0x0083),
        "psd": (0x0024, 0x0085),
        "vfi": (0x0024, 0x0100),
        "ght": (0x0024, 0x0087),
    }
    for key, (group, elem) in field_map.items():
        try:
            value = _safe_get(ds, (group, elem))
            if value is not None:
                data[key] = str(value)
        except Exception:
            pass
    return data


def _extract_heidelberg(ds):
    data = {}
    try:
        value = _safe_get(ds, (0x0022, 0x1418))
        if value is not None:
            data["rnfl_average"] = str(value)
    except Exception:
        pass
    return data


def _extract_optovue(ds):
    data = {}
    field_map = {
        "rnfl": (0x0099, 0x1050),
        "gcc": (0x0099, 0x1051),
    }
    for key, (group, elem) in field_map.items():
        try:
            value = _private_tag(ds, group, elem)
            if value is not None:
                data[key] = str(value)
        except Exception:
            pass
    return data


def _extract_afio_scanai(ds):
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
        try:
            value = _private_tag(ds, 0x0099, elem)
            if value is not None:
                data[key] = str(value)
        except Exception:
            pass
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


def _is_numeric(value):
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except Exception:
            return False
    return False


KNOWN_TAGS = {
    (0x0010, 0x0010), (0x0010, 0x0020), (0x0010, 0x0030), (0x0010, 0x0040),
    (0x0008, 0x0020), (0x0008, 0x0060), (0x0008, 0x0070), (0x0008, 0x1090),
    (0x0020, 0x0060),
}


def _extract_unknown_numeric_tags(ds):
    others = {}
    try:
        for elem in ds:
            try:
                tag = (elem.tag.group, elem.tag.element)
                if tag in KNOWN_TAGS:
                    continue
                if elem.VR in ("OB", "OW", "OF", "UN", "SQ"):
                    continue
                value = elem.value
                if _is_numeric(value):
                    key = f"{tag[0]:04X},{tag[1]:04X}"
                    others[key] = value
            except Exception:
                continue
    except Exception:
        pass
    return others


def parse_dicom_file(filepath):
    result = {
        "patient_name": None,
        "patient_id": None,
        "dob": None,
        "sex": None,
        "study_date": None,
        "modality": None,
        "manufacturer": None,
        "device_model": None,
        "laterality": None,
        "machine_type": "Unknown",
        "has_image": False,
        "image_jpeg_bytes": None,
    }

    try:
        ds = pydicom.dcmread(filepath)
    except Exception as e:
        result["error"] = f"Failed to read DICOM file: {e}"
        return result

    try:
        result["patient_name"] = _safe_str(ds, (0x0010, 0x0010), None)
    except Exception:
        pass
    try:
        result["patient_id"] = _safe_str(ds, (0x0010, 0x0020), None)
    except Exception:
        pass
    try:
        result["dob"] = _safe_str(ds, (0x0010, 0x0030), None)
    except Exception:
        pass
    try:
        result["sex"] = _safe_str(ds, (0x0010, 0x0040), None)
    except Exception:
        pass
    try:
        result["study_date"] = _safe_str(ds, (0x0008, 0x0020), None)
    except Exception:
        pass
    try:
        result["modality"] = _safe_str(ds, (0x0008, 0x0060), None)
    except Exception:
        pass
    try:
        result["manufacturer"] = _safe_str(ds, (0x0008, 0x0070), None)
    except Exception:
        pass
    try:
        result["device_model"] = _safe_str(ds, (0x0008, 0x1090), None)
    except Exception:
        pass
    try:
        result["laterality"] = _safe_str(ds, (0x0020, 0x0060), None)
    except Exception:
        pass

    machine_type = _detect_machine_type(result["manufacturer"], result["device_model"])
    result["machine_type"] = machine_type

    try:
        result["has_image"] = "PixelData" in ds
    except Exception:
        result["has_image"] = False

    try:
        result["image_jpeg_bytes"] = _extract_image_jpeg_bytes(ds)
    except Exception:
        result["image_jpeg_bytes"] = None

    clinical_data = {}
    try:
        if machine_type == "Topcon Triton / 3D OCT":
            clinical_data.update(_extract_topcon(ds))
        elif machine_type == "Galilei Corneal Topography":
            clinical_data.update(_extract_galilei(ds))
        elif machine_type == "Humphrey Visual Field":
            clinical_data.update(_extract_humphrey(ds))
        elif machine_type == "Heidelberg Spectralis":
            clinical_data.update(_extract_heidelberg(ds))
        elif machine_type == "Optovue Avanti":
            clinical_data.update(_extract_optovue(ds))
        elif machine_type == "Optos Ultra-Widefield":
            pass  # image only, nothing else to extract
        elif machine_type == "AFIO ScanAI":
            clinical_data.update(_extract_afio_scanai(ds))
    except Exception:
        pass

    result["clinical_data"] = clinical_data

    try:
        result["other_numeric_tags"] = _extract_unknown_numeric_tags(ds)
    except Exception:
        result["other_numeric_tags"] = {}

    return result


if __name__ == '__main__':
    import sys, json
    result = parse_dicom_file(sys.argv[1])
    # remove non-serializable bytes
    result.pop('image_jpeg_bytes', None)
    print(json.dumps(result, indent=2, default=str))

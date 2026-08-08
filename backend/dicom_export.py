import io
import os
import sys
import json
import base64
import datetime

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid
from pydicom.encaps import encapsulate
from PIL import Image

SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.77.1.5.1"  # Ophthalmic Photography 8 Bit Image Storage
TRANSFER_SYNTAX_UID = "1.2.840.10008.1.2.4.50"  # JPEG Baseline (Process 1)

PLACEHOLDER_SIZE = 512

LATERALITY_MAP = {"OD": "R", "OS": "L", "OU": "B"}


def _laterality(eye):
    return LATERALITY_MAP.get(str(eye or "").upper(), "")


def _format_dicom_date(value):
    """Best-effort conversion of a date-ish value to DICOM DA format (YYYYMMDD)."""
    if not value:
        return ""
    s = str(value)

    known_formats = ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                      "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y")
    for fmt in known_formats:
        try:
            return datetime.datetime.strptime(s, fmt).strftime("%Y%m%d")
        except ValueError:
            continue

    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _load_fundus_image(scan_data):
    """Locate and decode the scan's fundus image from whichever field holds it.

    Accepts a base64 data URL, raw base64, or a filesystem path. Falls back
    to a black placeholder if nothing usable is found.
    """
    candidates = [
        scan_data.get("fundusImage"),
        scan_data.get("fundus_image_path"),
        scan_data.get("imagePath"),
    ]

    od_results = scan_data.get("odResults") or {}
    os_results = scan_data.get("osResults") or {}
    candidates.append(od_results.get("fundusImage"))
    candidates.append(os_results.get("fundusImage"))

    img = None
    for candidate in candidates:
        if not candidate or not isinstance(candidate, str):
            continue
        try:
            if candidate.startswith("data:"):
                _, b64data = candidate.split(",", 1)
                raw = base64.b64decode(b64data)
                img = Image.open(io.BytesIO(raw))
            elif os.path.exists(candidate):
                img = Image.open(candidate)
            else:
                raw = base64.b64decode(candidate)
                img = Image.open(io.BytesIO(raw))
            img.load()
            break
        except Exception:
            img = None
            continue

    if img is None:
        img = Image.new("RGB", (PLACEHOLDER_SIZE, PLACEHOLDER_SIZE), color=(0, 0, 0))
    else:
        img = img.convert("RGB")

    return img


def _image_to_jpeg_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def create_dicom_from_scan(scan_data):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SOP_CLASS_UID
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = TRANSFER_SYNTAX_UID
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = SOP_CLASS_UID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    # ── Patient tags ──
    ds.PatientName = scan_data.get("patientName") or "Unknown"
    ds.PatientID = str(scan_data.get("patientId") or "")
    sex = str(scan_data.get("patientSex") or "")[:1].upper()
    ds.PatientSex = sex if sex in ("M", "F", "O") else ""
    ds.PatientBirthDate = _format_dicom_date(scan_data.get("patientDob"))

    # ── Study tags ──
    now = datetime.datetime.now()
    ds.StudyDate = _format_dicom_date(scan_data.get("timestamp")) or now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDescription = "AI-Assisted Retinal Screening"
    ds.InstitutionName = "Armed Forces Institute of Ophthalmology"
    ds.ReferringPhysicianName = scan_data.get("referringClinician") or ""

    # ── Series tags ──
    ds.Modality = "OP"
    ds.Manufacturer = "AFIO ScanAI"
    ds.ManufacturerModelName = "RetinalScan AI v2.0"
    ds.SeriesDescription = "Fundus Photography + AI Analysis"
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1

    # ── AI results as private tags (group 0x0099) ──
    dr = scan_data.get("drResult") or {}
    gl = scan_data.get("glaucomaResult") or {}
    hr = scan_data.get("hrResult") or {}
    triage = scan_data.get("triage") or {}

    block = ds.private_block(0x0099, "AFIO ScanAI", create=True)
    block.add_new(0x01, "LO", str(dr.get("grade", "")))
    block.add_new(0x02, "LO", str(dr.get("severityLabel", "")))
    block.add_new(0x03, "LO", str(dr.get("confidence", "")))
    block.add_new(0x10, "LO", str(gl.get("cdr", "")))
    block.add_new(0x11, "LO", str(gl.get("riskLevel", "")))
    block.add_new(0x20, "LO", str(hr.get("probability", "")))
    block.add_new(0x21, "LO", str(hr.get("detected", "")))
    block.add_new(0x30, "LO", str(triage.get("level", "")))
    block.add_new(0x31, "LO", str(scan_data.get("scanId", "")))

    # ── Eye laterality ──
    ds.Laterality = _laterality(scan_data.get("patientEye"))

    # ── Image ──
    img = _load_fundus_image(scan_data)
    jpeg_bytes = _image_to_jpeg_bytes(img)

    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "YBR_FULL_422"
    ds.PlanarConfiguration = 0
    ds.Rows = img.height
    ds.Columns = img.width
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.LossyImageCompression = "01"
    ds.LossyImageCompressionMethod = "ISO_10918_1"
    ds.PixelData = encapsulate([jpeg_bytes])

    return ds


if __name__ == '__main__':
    import sys, json
    scan_data = json.loads(sys.stdin.read())
    ds = create_dicom_from_scan(scan_data)
    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds)
    sys.stdout.buffer.write(buf.getvalue())

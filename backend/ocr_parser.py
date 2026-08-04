import re
import json
import sys

def parse_report(lines):
    text = ' '.join(lines)
    data = {}

    # ── GALILEI CORNEAL TOPOGRAPHY ──
    m = re.search(r'Sim[Kk]\s+([\d.]+)D', text)
    if m: data['simk'] = m.group(1) + 'D'

    m = re.search(r'[Ff]lat\s+Sim[Kk]\s+([\d.]+)D', text)
    if m: data['flat_simk'] = m.group(1) + 'D'

    m = re.search(r'Steep Sim[Kk]\s+([\d.]+)D', text)
    if m: data['steep_simk'] = m.group(1) + 'D'

    m = re.search(r'[Kk]max.aa\s+([\d.]+)D', text)
    if not m: m = re.search(r'[Kk]max\s*[\-\.]\s*aa\s*([\d.]+)', text)
    if m: data['kmax'] = m.group(1) + 'D'

    # Add Astig
    m = re.search(r'^Astig\s+([\d.]+)D', text, re.MULTILINE)
    if m: data['astig_simk'] = m.group(1) + 'D'

    m = re.search(r'Central\s+(\d{3,4})\s*[Uu][Mm]', text)
    if m: data['cct'] = m.group(1) + ' µm'

    # Fix thinnest — look for "Thinnest" specifically
    # not just any 3-digit number before µm
    for i, line in enumerate(lines):
        if 'Thinnest' in line or 'thinnest' in line:
            for j in range(i+1, min(i+3, len(lines))):
                digit_match = re.search(r'(\d{3,4})', lines[j])
                if digit_match:
                    data['thinnest_cornea'] = digit_match.group(1) + ' µm'
                    break

    m = re.search(r'W[IT]W.*?([\d.]+)mm', text)
    if m: data['wtw'] = m.group(1) + ' mm'

    m = re.search(r'AQD\s+([\d.]+)mm', text)
    if m: data['aqd'] = m.group(1) + ' mm'

    m = re.search(r'Mean K\s+([\d.]+)D', text)
    if m: data['mean_k'] = m.group(1) + 'D'

    # ── TOPCON TRITON DISC REPORT ──
    m = re.search(r'Linear CDR\s+([\d.]+)', text)
    if m: data['linear_cdr_od'] = m.group(1)

    m = re.search(r'Vertical CDR\s+([\d.]+)', text)
    if m: data['vertical_cdr_od'] = m.group(1)

    m = re.search(r'Disc Area.*?([\d.]+)\s*mm', text)
    if m: data['disc_area'] = m.group(1) + ' mm²'

    m = re.search(r'Rim Area.*?([\d.]+)\s*mm', text)
    if m: data['rim_area'] = m.group(1) + ' mm²'

    m = re.search(r'RNFL Symmetry\s+(\d+)%', text)
    if m: data['rnfl_symmetry'] = m.group(1) + '%'

    m = re.search(r'Total Thickness\s+(\d+)', text)
    if m: data['rnfl_total'] = m.group(1) + ' µm'

    # ── PATIENT INFO (any machine) ──
    m = re.search(r'Name\s*[:\-]?\s*([A-Za-z ]{3,40})', text)
    if m: data['patient_name'] = m.group(1).strip()

    m = re.search(r'(?:Age|DOB)\s*[:\-]?\s*(\d{1,3})', text)
    if m: data['patient_age'] = m.group(1)

    m = re.search(r'ID\s*[:\-]?\s*([A-Z0-9\-]{4,20})', text)
    if m: data['patient_id'] = m.group(1).strip()

    m = re.search(r'(OD|OS|OU|Right|Left)\s', text)
    if m: data['eye'] = m.group(1)

    m = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{2,4})', text)
    if m: data['date'] = m.group(1)

    # ── DETECT MACHINE TYPE ──
    if 'SimK' in text or 'Galilei' in text or 'Topography' in text:
        data['machine_type'] = 'Galilei Corneal Topography'
    elif 'RNFL' in text or 'CDR' in text or 'Disc' in text:
        data['machine_type'] = 'Topcon Triton OCT'
    elif 'Optomap' in text or 'OPTOS' in text:
        data['machine_type'] = 'Optos Ultra-Widefield'
    elif 'Humphrey' in text or 'MD ' in text or 'PSD' in text:
        data['machine_type'] = 'Humphrey Visual Field'
    elif 'RNFL' in text and 'Spectralis' in text:
        data['machine_type'] = 'Heidelberg Spectralis'
    else:
        data['machine_type'] = 'Unknown Machine'

    return data

if __name__ == '__main__':
    lines = json.loads(sys.stdin.read())
    result = parse_report(lines)
    print(json.dumps(result, indent=2))

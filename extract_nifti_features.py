"""
PROTEAS-MRI NIfTI Feature Extractor
=====================================
Parcourt l'arborescence des patients (P01, P02, ...) et extrait les features
complémentaires issues des fichiers NIfTI (.nii.gz) :
  - Volumes et ratios (tumeur, oedème, cerveau, nécrose, enhancing)
  - Intensité ADC dans la tumeur
  - Distribution de dose RT (dose moyenne, max, V95)
  - Features dynamiques inter-timepoints (delta volume, vélocité)

Format de sortie : même structure que PROTEAS-MRI_radiomics_data.xlsx
  Colonne 1 : NiftiFeature  (ex: mask_tumor__t1c__baseline__original_shape_BrainNormalizedVolume)
  Colonne 2 : NiftiValue

Usage:
    pip install nibabel numpy pandas openpyxl SimpleITK
    python extract_nifti_features.py

Structure attendue du dossier racine :
    ROOT_PATH/
      P01/
        P01_brain_mask.nii.gz
        P01_CT.nii.gz
        P01_RTP.nii.gz          (optionnel)
        BraTS/
          P01_seg.nii.gz        (labels: 1=nécrose, 2=oedème, 4=enhancing)
        DICOM/
          P01_ADC.nii.gz        (optionnel, séquence de diffusion)
        tumor_segmentation/
          baseline.nii.gz
          follow_up_1.nii.gz
          follow_up_2.nii.gz    (optionnel)
          follow_up_3.nii.gz    (optionnel)
          follow_up_4.nii.gz    (optionnel)
"""

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT_PATH   = "17253793/"          # ← adapte ce chemin
OUTPUT_PATH = "nifti_features.csv"

# Mapping timepoint label -> patterns de fichiers possibles dans tumor_segmentation/
TIMEPOINTS = ["baseline", "follow_up_1", "follow_up_2", "follow_up_3", "follow_up_4"]

TP_FILE_PATTERNS = {
    "baseline":    ["baseline.nii.gz", "*baseline*", "*BL*", "*bl*"],
    "follow_up_1": ["follow_up_1.nii.gz", "*follow_up_1*", "fu1.nii.gz", "*fu1*", "*FU1*"],
    "follow_up_2": ["follow_up_2.nii.gz", "*follow_up_2*", "fu2.nii.gz", "*fu2*", "*FU2*"],
    "follow_up_3": ["follow_up_3.nii.gz", "*follow_up_3*", "fu3.nii.gz", "*fu3*", "*FU3*"],
    "follow_up_4": ["follow_up_4.nii.gz", "*follow_up_4*", "fu4.nii.gz", "*fu4*", "*FU4*"],
}

# Labels BraTS standard
BRATS_NECROSIS  = 1
BRATS_EDEMA     = 2
BRATS_ENHANCING = 4

# ── Utilitaires NIfTI ──────────────────────────────────────────────────────────

def load_nii(path):
    """Charge un NIfTI, retourne (data_array, voxel_volume_mm3) ou (None, None)."""
    if path and os.path.exists(path):
        img = nib.load(path)
        data = img.get_fdata()
        zooms = img.header.get_zooms()[:3]
        voxel_vol = float(np.prod(zooms))
        return data, voxel_vol
    return None, None

def find_file(folder, patterns):
    """Retourne le premier fichier correspondant à l'un des patterns glob."""
    for pat in patterns:
        matches = glob.glob(os.path.join(folder, pat))
        if matches:
            return matches[0]
    return None

def volume_mm3(mask_data, voxel_vol):
    """Volume en mm³ d'un masque binaire."""
    return float(np.sum(mask_data > 0)) * voxel_vol

def safe_mean(arr, mask):
    """Moyenne d'arr dans le masque, None si masque vide."""
    vals = arr[mask > 0]
    return float(np.mean(vals)) if len(vals) > 0 else None

def safe_max(arr, mask):
    vals = arr[mask > 0]
    return float(np.max(vals)) if len(vals) > 0 else None

def safe_std(arr, mask):
    vals = arr[mask > 0]
    return float(np.std(vals)) if len(vals) > 0 else None

def dose_vx(dose_data, mask, threshold_pct, dose_max):
    """Volume (%) du masque recevant >= threshold_pct % de la dose max."""
    if dose_max == 0:
        return None
    voxels_above = np.sum((dose_data >= threshold_pct / 100.0 * dose_max) & (mask > 0))
    total = np.sum(mask > 0)
    return float(voxels_above / total * 100) if total > 0 else None

# ── Extraction par patient ─────────────────────────────────────────────────────

def extract_patient(patient_id, patient_dir):
    """
    Retourne une liste de dicts :
      { 'NiftiFeature': str, 'NiftiValue': float }
    """
    records = []

    def add(mask, seq, timepoint, feature, value):
        if value is not None:
            key = f"{mask}__{seq}__{timepoint}__{feature}"
            records.append({"NiftiFeature": key, "NiftiValue": round(value, 6)})

    pid = patient_id  # ex: P04a

    # ── Fichiers fixes (indépendants du timepoint) ──
    brain_path = find_file(patient_dir, [f"{pid}_brain_mask.nii.gz", "*brain_mask*"])
    ct_path    = find_file(patient_dir, [f"{pid}_CT.nii.gz", "*CT*"])
    rtp_path   = find_file(patient_dir, [f"{pid}_RTP.nii.gz", "*RTP*", "*dose*"])
    brats_path = find_file(os.path.join(patient_dir, "BraTS"), ["*seg*", "*.nii.gz"])
    adc_path   = find_file(os.path.join(patient_dir, "DICOM"), ["*ADC*", "*adc*"])

    brain_data, brain_vox = load_nii(brain_path)
    ct_data,    ct_vox    = load_nii(ct_path)
    rtp_data,   _         = load_nii(rtp_path)
    brats_data, brats_vox = load_nii(brats_path)
    adc_data,   _         = load_nii(adc_path)

    brain_vol = volume_mm3(brain_data, brain_vox) if brain_data is not None else None

    # ── BraTS : features globales (timepoint = baseline par convention) ──
    if brats_data is not None:
        necrosis_mask  = (brats_data == BRATS_NECROSIS).astype(np.uint8)
        edema_mask     = (brats_data == BRATS_EDEMA).astype(np.uint8)
        enhancing_mask = (brats_data == BRATS_ENHANCING).astype(np.uint8)
        whole_tumor    = ((brats_data > 0)).astype(np.uint8)

        vol_nec  = volume_mm3(necrosis_mask,  brats_vox)
        vol_ede  = volume_mm3(edema_mask,      brats_vox)
        vol_enh  = volume_mm3(enhancing_mask,  brats_vox)
        vol_whole= volume_mm3(whole_tumor,     brats_vox)

        add("mask_tumor", "brats", "baseline", "original_shape_NecrosisVolume_mm3",  vol_nec)
        add("mask_tumor", "brats", "baseline", "original_shape_EdemaVolume_mm3",     vol_ede)
        add("mask_tumor", "brats", "baseline", "original_shape_EnhancingVolume_mm3", vol_enh)
        add("mask_tumor", "brats", "baseline", "original_shape_WholeTumorVolume_mm3",vol_whole)

        if vol_whole and vol_whole > 0:
            add("mask_tumor", "brats", "baseline", "original_shape_NecrosisFraction",  vol_nec  / vol_whole)
            add("mask_tumor", "brats", "baseline", "original_shape_EnhancingFraction", vol_enh  / vol_whole)
            add("mask_tumor", "brats", "baseline", "original_shape_EdemaFraction",     vol_ede  / vol_whole)

        if brain_vol and brain_vol > 0:
            add("mask_brain", "brats", "baseline", "original_shape_TumorBrainRatio", vol_whole / brain_vol)

        # ADC dans les sous-régions BraTS
        if adc_data is not None:
            add("mask_tumor",   "adc", "baseline", "original_firstorder_ADC_Mean_Enhancing", safe_mean(adc_data, enhancing_mask))
            add("mask_tumor",   "adc", "baseline", "original_firstorder_ADC_Mean_Necrosis",  safe_mean(adc_data, necrosis_mask))
            add("mask_oedema",  "adc", "baseline", "original_firstorder_ADC_Mean_Edema",     safe_mean(adc_data, edema_mask))
            add("mask_tumor",   "adc", "baseline", "original_firstorder_ADC_Std_Enhancing",  safe_std(adc_data,  enhancing_mask))

        # Dose RT dans les sous-régions BraTS
        if rtp_data is not None:
            dose_max_brain = safe_max(rtp_data, brain_data) if brain_data is not None else None
            dose_max_val   = dose_max_brain or safe_max(rtp_data, whole_tumor)

            for mask_name, mask_arr in [
                ("mask_tumor",  enhancing_mask),
                ("mask_oedema", edema_mask),
                ("mask_brain",  brain_data if brain_data is not None else whole_tumor),
            ]:
                add(mask_name, "rtp", "baseline", "original_dose_Mean_Gy",  safe_mean(rtp_data, mask_arr))
                add(mask_name, "rtp", "baseline", "original_dose_Max_Gy",   safe_max(rtp_data,  mask_arr))
                add(mask_name, "rtp", "baseline", "original_dose_Std_Gy",   safe_std(rtp_data,  mask_arr))
                if dose_max_val:
                    add(mask_name, "rtp", "baseline", "original_dose_V95_pct",
                        dose_vx(rtp_data, mask_arr, 95, dose_max_val))

    # ── CT : densité dans le cerveau ──
    if ct_data is not None and brain_data is not None:
        add("mask_brain", "ct", "baseline", "original_firstorder_HU_Mean", safe_mean(ct_data, brain_data))
        add("mask_brain", "ct", "baseline", "original_firstorder_HU_Std",  safe_std(ct_data,  brain_data))

    # ── Features dynamiques (par timepoint) depuis tumor_segmentation/ ──
    seg_dir = os.path.join(patient_dir, "tumor_segmentation")
    prev_vol = None
    prev_tp  = None
    voxel_vol_seg = None

    for tp in TIMEPOINTS:
        seg_path = find_file(seg_dir, TP_FILE_PATTERNS[tp])
        if seg_path is None:
            continue

        seg_data, vox = load_nii(seg_path)
        if seg_data is None:
            continue
        voxel_vol_seg = vox

        vol = volume_mm3(seg_data, vox)
        add("mask_tumor", "t1c", tp, "original_shape_TumorVolume_mm3", vol)

        if brain_vol and brain_vol > 0:
            add("mask_tumor", "t1c", tp, "original_shape_TumorBrainRatio", vol / brain_vol)

        # ADC dans le masque de tumeur à ce timepoint
        if adc_data is not None:
            add("mask_tumor", "adc", tp, "original_firstorder_ADC_Mean", safe_mean(adc_data, seg_data))
            add("mask_tumor", "adc", tp, "original_firstorder_ADC_Std",  safe_std(adc_data,  seg_data))

        # Delta volume vs timepoint précédent
        if prev_vol is not None:
            delta = vol - prev_vol
            ratio = vol / prev_vol if prev_vol > 0 else None
            add("mask_tumor", "t1c", tp, "original_dynamic_DeltaVolume_mm3",    delta)
            add("mask_tumor", "t1c", tp, "original_dynamic_VolumeGrowthRatio",  ratio)

        prev_vol = vol
        prev_tp  = tp

    # Delta total baseline → dernier timepoint disponible
    tp_vols = {}
    for tp in TIMEPOINTS:
        seg_path = find_file(seg_dir, TP_FILE_PATTERNS[tp])
        if seg_path:
            seg_data, vox = load_nii(seg_path)
            if seg_data is not None:
                tp_vols[tp] = volume_mm3(seg_data, vox)

    if "baseline" in tp_vols and len(tp_vols) > 1:
        last_tp   = [t for t in TIMEPOINTS if t in tp_vols][-1]
        total_delta = tp_vols[last_tp] - tp_vols["baseline"]
        ratio_total = tp_vols[last_tp] / tp_vols["baseline"] if tp_vols["baseline"] > 0 else None
        add("mask_tumor", "t1c", last_tp, "original_dynamic_TotalDeltaVolume_mm3",      total_delta)
        add("mask_tumor", "t1c", last_tp, "original_dynamic_TotalVolumeGrowthRatio",    ratio_total)

    return records


# ── Export Excel ───────────────────────────────────────────────────────────────

HEADER_FILL  = PatternFill("solid", start_color="1F4E79")
HEADER_FONT  = Font(name="Arial", bold=True, color="FFFFFF", size=10)
SUBHDR_FILL  = PatternFill("solid", start_color="2E75B6")
SUBHDR_FONT  = Font(name="Arial", bold=True, color="FFFFFF", size=10)
CELL_FONT    = Font(name="Arial", size=10)
ALT_FILL     = PatternFill("solid", start_color="EBF3FB")
THIN_BORDER  = Border(
    left=Side(style="thin", color="BDD7EE"),
    right=Side(style="thin", color="BDD7EE"),
    top=Side(style="thin", color="BDD7EE"),
    bottom=Side(style="thin", color="BDD7EE"),
)

def style_sheet(ws, df):
    """Applique le même style visuel que le fichier radiomics source."""
    ws.freeze_panes = "A2"

    col_widths = {"NiftiFeature": 65, "NiftiValue": 22}
    for col_idx, col_name in enumerate(df.columns, 1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = col_widths.get(col_name, 18)

    # En-tête
    for col_idx, col_name in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font      = HEADER_FONT
        cell.fill      = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = THIN_BORDER
    ws.row_dimensions[1].height = 30

    # Données
    for row_idx, row in enumerate(df.itertuples(index=False), 2):
        fill = ALT_FILL if row_idx % 2 == 0 else None
        for col_idx, value in enumerate(row, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.font   = CELL_FONT
            cell.border = THIN_BORDER
            if fill:
                cell.fill = fill
            if col_idx == 2 and isinstance(value, float):
                cell.number_format = "0.000000"
                cell.alignment = Alignment(horizontal="right")
            else:
                cell.alignment = Alignment(horizontal="left")


def build_excel(all_data: dict):
    wb = Workbook()
    wb.remove(wb.active)  # retire la feuille vide par défaut

    # Feuille maître (tous patients)
    all_records = []
    for pid, records in all_data.items():
        for r in records:
            all_records.append({"Patient_ID": pid, **r})

    master_df = pd.DataFrame(all_records)[["Patient_ID", "NiftiFeature", "NiftiValue"]]
    ws_master = wb.create_sheet("ALL_PATIENTS")
    ws_master.append(list(master_df.columns))
    for row in master_df.itertuples(index=False):
        ws_master.append(list(row))
    style_sheet(ws_master, master_df)

    # Une feuille par patient (même logique que le fichier radiomics)
    for pid, records in sorted(all_data.items()):
        if not records:
            continue
        df = pd.DataFrame(records)
        ws = wb.create_sheet(pid)
        ws.append(list(df.columns))
        for row in df.itertuples(index=False):
            ws.append(list(row))
        style_sheet(ws, df)

    wb.save(OUTPUT_PATH)
    print(f"\n✓ Export : {OUTPUT_PATH}")
    print(f"  {len(all_records)} features  |  {len(all_data)} patients")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(ROOT_PATH):
        print(f"[ERREUR] Dossier racine introuvable : {ROOT_PATH}")
        return

    patient_dirs = sorted([
        d for d in os.listdir(ROOT_PATH)
        if os.path.isdir(os.path.join(ROOT_PATH, d)) and d.startswith("P")
    ])
    print(f"Patients détectés : {patient_dirs}\n")

    all_data = {}
    for pid in patient_dirs:
        pdir = os.path.join(ROOT_PATH, pid)
        print(f"  Traitement {pid} …")
        records = extract_patient(pid, pdir)
        all_data[pid] = records
        print(f"    → {len(records)} features extraites")

    build_excel(all_data)


if __name__ == "__main__":
    main()
"""
================================================================================
PIPELINE ML — PRÉDICTION DES PARAMÈTRES PDE DE CROISSANCE TUMORALE
================================================================================

Modèle physique (Fisher-KPP avec chimiothérapie) :

    ∂u/∂t = D·∇²u + r·u·(1−u) − α·c(x,t)·u

  où :
    u(x,t)  = densité normalisée de cellules tumorales  [0, 1]
    D        = coefficient de diffusion/invasion          [mm²/jour]
    r        = taux de prolifération nette                [1/jour]
    α        = efficacité du traitement                   [1/jour]
    c(x,t)  = concentration normalisée du médicament

Relations clés du front de Fisher-KPP :
    vitesse du front  : v = 2√(r·D)      [mm/jour]
    largeur du front  : λ = √(D/r)       [mm]

Structure du dataset :
  ./                                    ← dossier du script + CSV
  ├── tumor_pde_ml_pipeline.py
  ├── radiomics_features.csv  (ou .xlsx)
  ├── clinical_features.csv   (ou .xlsx)
  ├── nifti_features.csv      (ou .xlsx)
  └── 17253793/
      └── P04a/
          ├── BraTS/
          │   ├── baseline/   fla.nii.gz  t1.nii.gz  t1c.nii.gz  t2.nii.gz
          │   ├── fu1/        …
          │   ├── fu2/        …
          │   └── fu3/        …
          ├── tumor_segmentation/
          │   ├── P04a_tumor_mask_baseline.nii.gz
          │   ├── P04a_tumor_mask_fu1.nii.gz
          │   ├── P04a_tumor_mask_fu2.nii.gz
          │   └── P04a_tumor_mask_fu3.nii.gz
          ├── P04a_brain_mask.nii.gz
          ├── P04a_CT.nii.gz
          └── P04a_RTP.nii.gz

Correspondance timepoints :
  baseline / follow_up_1…4  →  baseline / fu1…fu4  →  _tumor_mask_baseline…fu4.nii.gz

Usage :
  # Entraînement (depuis le dossier contenant le script)
  python tumor_pde_ml_pipeline.py

  # Avec options
  python tumor_pde_ml_pipeline.py --data_dir 17253793 --dt_days 56 --output results/

  # Prédiction nouveau patient (sorties dans prediction/<patient_id>/)
  python tumor_pde_ml_pipeline.py --predict --patient_id P04a --simulate_days 30

  # Prédiction + masque binaire + série temporelle 4D
  python tumor_pde_ml_pipeline.py --predict --patient_id P04a --simulate_days 200 \
      --n_snapshots 20 --bin_threshold 0.5

  # Modélisation spatiale GM/WM/CSF (diffusion plus rapide en matière blanche)
  python tumor_pde_ml_pipeline.py --predict --patient_id P04a --simulate_days 200 \
      --d_wm_gm_ratio 7 --csf_factor 0.1
  # (désactiver : --no_tissue → simulation homogène classique)

Modélisation spatiale (modèle de Swanson) :
  Si un T1 baseline est disponible, le cerveau est segmenté en CSF/GM/WM (K-means)
  et la diffusion devient hétérogène : ∂u/∂t = ∇·(D(x)·∇u) + r(x)·u(1−u) − α(x)·c·u.
  D(x), r(x), α(x) dérivent du tissu (WM = ratios réglables, GM = référence, CSF réduit).

Sorties de prédiction (prediction/<patient_id>/) :
  <PID>_u_T<j>j.nii.gz      densité 3D finale (continu 0–1)
  <PID>_mask_T<j>j.nii.gz   masque binaire 3D final (u > seuil)
  <PID>_u_4D.nii.gz         série temporelle 4D de densité (X,Y,Z,T)
  <PID>_mask_4D.nii.gz      série temporelle 4D binaire (X,Y,Z,T)
  <PID>_4D_timepoints.csv   correspondance frame → jour
  <PID>_tissue_map.nii.gz   carte de tissus (1=CSF, 2=GM, 3=WM) si T1 dispo
  <PID>_volumes_sim.csv     volume tumoral jour par jour
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import argparse
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.ndimage import binary_erosion, label

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_DRUG = np.log(2) / 14.0   # demi-vie médicament ≈ 14 jours [1/j]
C0          = 0.90                 # concentration normalisée à t=0

TARGET_PARAMS = ["D", "r", "alpha"]

# Correspondance label tabulaire → dossier BraTS → suffixe masque tumeur
TIMEPOINT_TO_FOLDER = {
    "baseline":    "baseline",
    "follow_up_1": "fu1",
    "follow_up_2": "fu2",
    "follow_up_3": "fu3",
    "follow_up_4": "fu4",
}
TIMEPOINT_TO_INT = {
    "baseline": 0, "follow_up_1": 1,
    "follow_up_2": 2, "follow_up_3": 3, "follow_up_4": 4,
}

RADIOMICS_COLS = [
    "MeshVolume", "SurfaceArea", "Sphericity", "Maximum3DDiameter",
    "Elongation", "VoxelValue_Mean", "VoxelValue_Maximum", "VoxelValue_SD",
    "GLCM_Entropy", "GLCM_Contrast",
    "GLSZM_SizeZoneNonUniformity", "GLDM_DependenceNonUniformity",
]
CLINICAL_COLS = [
    "Age", "KPS", "WHO_PS", "Dose_Gy", "Fractions",
    "Prior_Chemo", "Histology_enc", "N_metastases",
]

# Matrice X réduite : 6 variables les plus prédictives (bruit radiomique filtré).
FEATURE_MATRIX_COLS = [
    "t1c_Maximum3DDiameter",  # D
    "t1c_Sphericity",         # r
    "t1c_VoxelValue_Mean",    # densité cellulaire (proxy)
    "ratio_T_O_Volume",       # infiltration (proxy)
    "Age",
    "Prior_Chemo",            # α
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — LECTURE ROBUSTE DES FICHIERS TABULAIRES
# ─────────────────────────────────────────────────────────────────────────────

def _read_tabular_file(path: Path, label: str = "") -> pd.DataFrame:
    """
    Lit un CSV ou Excel (.xlsx/.xls).
    Gère les fichiers xlsx renommés en .csv (magic bytes PK).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label or 'Fichier'} introuvable : {path.resolve()}")

    with open(path, "rb") as f:
        magic = f.read(4)
    is_excel_zip = magic.startswith(b"PK")
    suf = path.suffix.lower()

    if suf in (".xlsx", ".xls") or is_excel_zip:
        for sheet in ("ALL_PATIENTS", "Sheet1", "Feuil1", 0):
            try:
                df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
                if df is not None and len(df) > 0:
                    return df
            except Exception:
                continue
        return pd.read_excel(path, sheet_name=0, engine="openpyxl")

    # CSV : détection automatique encodage + séparateur
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        for sep in (None, ",", ";", "\t"):
            try:
                return pd.read_csv(path, encoding=enc, sep=sep, engine="python")
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    return pd.read_csv(path, encoding="cp1252", sep=None,
                       engine="python", encoding_errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CHARGEMENT DES DONNÉES TABULAIRES
# ─────────────────────────────────────────────────────────────────────────────

def load_tabular_data(script_dir: Path):
    """
    Cherche radiomics / clinical / nifti_features dans script_dir/features
    puis dans script_dir (priorité .xlsx puis .csv).
    """
    def _find(base):
        candidates = [script_dir / "features", script_dir]
        for folder in candidates:
            for ext in (".xlsx", ".csv", ".xls"):
                p = folder / (base + ext)
                if p.exists():
                    return p
        raise FileNotFoundError(
            f"Fichier {base}.xlsx / .csv introuvable dans {script_dir / 'features'} "
            f"ni dans {script_dir}"
        )

    rad_path  = _find("radiomics_features")
    clin_path = _find("clinical_features")
    nif_path  = _find("nifti_features")

    # ── Radiomics ─────────────────────────────────────────────────────────────
    rad = _read_tabular_file(rad_path, "Radiomics")
    rad.columns = rad.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    # ── Clinical ──────────────────────────────────────────────────────────────
    clin = _read_tabular_file(clin_path, "Clinical")
    # Strip AVANT le rename pour capturer les espaces trailing dans les noms de colonnes
    clin.columns = clin.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    clin = clin.rename(columns={
        "Patient ID (Zenodo)":      "Patient_ID",
        "Tumour Histology":         "Histology",
        "Prior Chemo":              "Prior_Chemo",
        "Karnofsky PS (%)":         "KPS",
        "Age (yrs)":                "Age",
        "Number of metastases":     "N_metastases",
        "Rx dose at tumor margins": "Dose_raw",
        "Fractions":                "Fractions",
        "WHO PS":                   "WHO_PS",
    })
    if "Patient_ID" not in clin.columns:
        for alt in ("Patient ID", "PatientID", "patient_id", "ID"):
            if alt in clin.columns:
                clin = clin.rename(columns={alt: "Patient_ID"})
                break
    if "Patient_ID" not in clin.columns:
        raise ValueError(f'clinical : colonne patient introuvable. Colonnes : {list(clin.columns)}')
    clin["Patient_ID"] = clin["Patient_ID"].astype(str).str.strip()

    if "Dose_raw" in clin.columns:
        clin["Dose_Gy"] = clin["Dose_raw"].astype(str).str.extract(r"(\d+)").astype(float)
    elif "Dose_Gy" not in clin.columns:
        clin["Dose_Gy"] = np.nan

    if "Prior_Chemo" in clin.columns:
        pc = clin["Prior_Chemo"]
        if pd.api.types.is_numeric_dtype(pc):
            clin["Prior_Chemo"] = (pc.fillna(0) > 0).astype(int)
        else:
            clin["Prior_Chemo"] = (
                pc.astype(str).str.strip().str.lower()
                  .isin(("yes", "y", "1", "true", "oui"))
            ).astype(int)
    else:
        clin["Prior_Chemo"] = 0

    clin["Histology"] = clin.get("Histology", pd.Series(dtype=str)).astype(str).str.strip()
    clin["Histology_enc"] = clin["Histology"].map(
        {"NSCLC": 0, "Breast Ca": 1, "SCLC": 2}
    ).fillna(3).astype(int)

    # ── NIfTI features (format long : Patient_ID / NiftiFeature / NiftiValue) ─
    nifti_df = _read_tabular_file(nif_path, "NIfTI features")
    nifti_df.columns = nifti_df.columns.astype(str).str.strip()
    # Renommages alternatifs de colonnes
    for old, new in (("Feature", "NiftiFeature"), ("Value", "NiftiValue")):
        if new not in nifti_df.columns and old in nifti_df.columns:
            nifti_df = nifti_df.rename(columns={old: new})

    log.info(f"Radiomics {rad.shape} | Clinique {clin.shape} | NIfTI feat {nifti_df.shape}")
    return rad, clin, nifti_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — RÉSOLUTION DES CHEMINS PATIENTS
# ─────────────────────────────────────────────────────────────────────────────

def resolve_paths(patient_dir: Path, patient_id: str) -> dict:
    """
    Construit le dictionnaire de tous les chemins utiles pour un patient.

        patient_dir : .../17253793/<PatientID>/
        patient_id  : ex. "P04a"

    Clés retournées :
        paths["t1c"][tp_label]  → BraTS/<folder>/t1c.nii.gz
        paths["flair"][tp_label]→ BraTS/<folder>/fla.nii.gz
        paths["seg"][tp_label]  → tumor_segmentation/<PID>_tumor_mask_<suffix>.nii.gz
        paths["brain_mask"]     → <PID>_brain_mask.nii.gz
    """
    brats_dir = patient_dir / "BraTS"
    seg_dir   = patient_dir / "tumor_segmentation"

    paths = {"t1c": {}, "flair": {}, "t1": {}, "t2": {}, "seg": {}}
    for tp_label, folder in TIMEPOINT_TO_FOLDER.items():
        tp_folder = brats_dir / folder
        paths["t1c"][tp_label]   = tp_folder / "t1c.nii.gz"
        paths["flair"][tp_label] = tp_folder / "fla.nii.gz"
        paths["t1"][tp_label]    = tp_folder / "t1.nii.gz"
        paths["t2"][tp_label]    = tp_folder / "t2.nii.gz"
        paths["seg"][tp_label]   = seg_dir / f"{patient_id}_tumor_mask_{folder}.nii.gz"

    paths["brain_mask"] = patient_dir / f"{patient_id}_brain_mask.nii.gz"
    paths["ct"]         = patient_dir / f"{patient_id}_CT.nii.gz"
    paths["rtp"]        = patient_dir / f"{patient_id}_RTP.nii.gz"
    return paths


def available_timepoints(paths: dict) -> list:
    """Timepoints pour lesquels le masque de segmentation existe."""
    order = ["baseline", "follow_up_1", "follow_up_2", "follow_up_3", "follow_up_4"]
    return [tp for tp in order if paths["seg"][tp].exists()]


def list_patients(data_dir: Path) -> list:
    if not data_dir.exists():
        return []
    return sorted(d.name for d in data_dir.iterdir()
                  if d.is_dir() and not d.name.startswith("."))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — EXTRACTION FEATURES NIfTI DEPUIS MASQUES DE SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def extract_volume_from_mask(seg_path: Path):
    """Volume tumoral [mm³] depuis un masque NIfTI. None si absent."""
    if not seg_path.exists():
        return None
    try:
        import nibabel as nib
        img  = nib.load(str(seg_path))
        data = img.get_fdata()
        zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
        return float((data > 0).sum() * np.prod(zooms))
    except Exception as e:
        log.debug(f"Masque non lisible ({seg_path.name}) : {e}")
        return None


def extract_nifti_features(seg_path: Path, t1c_path: Path = None,
                            flair_path: Path = None) -> dict:
    """
    Features géométriques + intensités depuis les NIfTI réels.

    Retourne un dict avec les clés :
      nii_Volume_mm3, nii_Sphericity, nii_MaxDiameter_mm, nii_Elongation,
      nii_T1CE_Mean, nii_T1CE_Std, nii_T1CE_Max,
      nii_FLAIR_Mean, nii_FLAIR_Std
    """
    try:
        import nibabel as nib
    except ImportError:
        log.warning("nibabel non installé (pip install nibabel). "
                    "Features NIfTI non calculées.")
        return {}

    if not seg_path.exists():
        return {}

    feats = {}
    seg_img  = nib.load(str(seg_path))
    seg_data = seg_img.get_fdata()
    zooms    = np.array(seg_img.header.get_zooms(), dtype=float)
    vox_vol  = float(np.prod(zooms))

    mask = (seg_data > 0).astype(np.uint8)
    nvox = int(mask.sum())
    if nvox == 0:
        return feats

    vol = nvox * vox_vol
    feats["nii_Volume_mm3"] = vol

    # Géométrie
    coords  = np.argwhere(mask > 0)
    bbox_mm = (coords.max(axis=0) - coords.min(axis=0)) * zooms[:3]
    bbox_mm = np.maximum(bbox_mm, 1e-3)
    feats["nii_MaxDiameter_mm"] = float(np.linalg.norm(bbox_mm))
    feats["nii_Elongation"]     = float(bbox_mm.min() / bbox_mm.max())

    eroded = binary_erosion(mask)
    border = mask.astype(float) - eroded.astype(float)
    sa     = float(border.sum()) * (vox_vol ** (2.0 / 3.0))
    feats["nii_Sphericity"] = float(
        (np.pi ** (1/3)) * ((6 * vol) ** (2/3)) / sa
    ) if sa > 0 else 0.0

    # Intensités T1CE
    if t1c_path and t1c_path.exists():
        try:
            t1c_data = nib.load(str(t1c_path)).get_fdata()
            if t1c_data.shape == seg_data.shape:
                vals = t1c_data[mask > 0]
                if len(vals) > 0:
                    feats["nii_T1CE_Mean"] = float(vals.mean())
                    feats["nii_T1CE_Std"]  = float(vals.std())
                    feats["nii_T1CE_Max"]  = float(vals.max())
        except Exception as e:
            log.debug(f"T1CE : {e}")

    # Intensités FLAIR
    if flair_path and flair_path.exists():
        try:
            fl_data = nib.load(str(flair_path)).get_fdata()
            if fl_data.shape == seg_data.shape:
                vals = fl_data[mask > 0]
                if len(vals) > 0:
                    feats["nii_FLAIR_Mean"] = float(vals.mean())
                    feats["nii_FLAIR_Std"]  = float(vals.std())
        except Exception as e:
            log.debug(f"FLAIR : {e}")

    return feats


def extract_all_nifti_features(data_dir: Path) -> pd.DataFrame:
    """Extrait les features NIfTI au baseline pour tous les patients."""
    patients = list_patients(data_dir)
    if not patients:
        return pd.DataFrame()

    log.info(f"Extraction NIfTI baseline ({len(patients)} patients)…")
    records = []
    for pid in patients:
        paths = resolve_paths(data_dir / pid, pid)
        feats = extract_nifti_features(
            paths["seg"]["baseline"],
            paths["t1c"]["baseline"],
            paths["flair"]["baseline"],
        )
        if feats:
            feats["Patient_ID"] = pid
            records.append(feats)
            log.info(f"  ✓ {pid} — Vol={feats.get('nii_Volume_mm3',0):.1f} mm³")
        else:
            log.debug(f"  ✗ {pid} — aucune feature extraite")

    return pd.DataFrame(records) if records else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — VOLUMES LONGITUDINAUX (NIfTI réels + fallback CSV)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_longitudinal_from_csv(nifti_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait (Patient_ID, timepoint_int, Volume_mm3) depuis nifti_features.
    Utilisé en fallback si les masques NIfTI sont absents.
    """
    tv = nifti_df[
        nifti_df["NiftiFeature"].str.contains("TumorVolume_mm3", na=False)
    ].copy()

    def _tp(name):
        if "baseline"    in name: return 0
        for i in range(1, 6):
            if f"follow_up_{i}" in name: return i
        return None

    tv["timepoint"]  = tv["NiftiFeature"].apply(_tp)
    tv = tv.dropna(subset=["timepoint"])
    tv["timepoint"]  = tv["timepoint"].astype(int)
    tv = tv.rename(columns={"NiftiValue": "Volume_mm3"})
    tv = tv[["Patient_ID", "timepoint", "Volume_mm3"]]
    # Moyenne si doublons
    tv = tv.groupby(["Patient_ID", "timepoint"], as_index=False)["Volume_mm3"].mean()
    return tv.sort_values(["Patient_ID", "timepoint"])


def build_longitudinal_volumes(data_dir: Path, nifti_df: pd.DataFrame,
                                dt_days: float = 56.0) -> pd.DataFrame:
    """
    Construit (Patient_ID, tp_int, time_days, Volume_mm3, source).

    Priorité : masques tumor_segmentation/ → nifti_features.csv
    Les temps sont en jours depuis le baseline : tp_int × dt_days.
    """
    csv_vol  = _extract_longitudinal_from_csv(nifti_df)
    patients = list_patients(data_dir)
    records  = []

    if not patients:
        log.warning(f"{data_dir} vide — utilisation exclusive du CSV.")
        for _, row in csv_vol.iterrows():
            records.append(dict(Patient_ID=row["Patient_ID"],
                                tp_int=int(row["timepoint"]),
                                time_days=int(row["timepoint"]) * dt_days,
                                Volume_mm3=row["Volume_mm3"], source="csv"))
        return pd.DataFrame(records)

    log.info(f"Construction volumes longitudinaux ({len(patients)} patients)…")

    for pid in patients:
        pat_dir = data_dir / pid
        paths   = resolve_paths(pat_dir, pid)
        avail   = available_timepoints(paths)

        if not avail:
            # Fallback CSV complet pour ce patient
            sub = csv_vol[csv_vol["Patient_ID"] == pid]
            for _, row in sub.iterrows():
                records.append(dict(Patient_ID=pid,
                                    tp_int=int(row["timepoint"]),
                                    time_days=int(row["timepoint"]) * dt_days,
                                    Volume_mm3=row["Volume_mm3"], source="csv"))
            continue

        for tp_label in avail:
            tp_int = TIMEPOINT_TO_INT[tp_label]
            vol    = extract_volume_from_mask(paths["seg"][tp_label])

            if vol is None or vol <= 0:
                # Fallback CSV pour ce timepoint
                sub = csv_vol[
                    (csv_vol["Patient_ID"] == pid) &
                    (csv_vol["timepoint"]  == tp_int)
                ]
                if sub.empty:
                    continue
                vol    = float(sub.iloc[0]["Volume_mm3"])
                source = "csv"
            else:
                source = "nifti_seg"

            records.append(dict(Patient_ID=pid, tp_int=tp_int,
                                time_days=tp_int * dt_days,
                                Volume_mm3=vol, source=source))

    df = pd.DataFrame(records).sort_values(["Patient_ID", "tp_int"])
    n_nii = (df["source"] == "nifti_seg").sum()
    n_csv = (df["source"] == "csv").sum()
    log.info(f"Volumes : {df['Patient_ID'].nunique()} patients | "
             f"nifti_seg={n_nii} | csv={n_csv}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ESTIMATION DE LA VÉRITÉ TERRAIN (D, r, α)
# ─────────────────────────────────────────────────────────────────────────────

def _ode_kpp_phases(t, u, r, alpha, t_treat):
    """
    Phase 1 (t < t_treat) : prolifération logistique seule.
    Phase 2 (t ≥ t_treat) : idem + chimiothérapie, c(t) décalée depuis le début du traitement.
    """
    uu = float(u[0])
    if t < t_treat:
        dudt = r * uu * (1.0 - uu)
    else:
        c_t  = C0 * np.exp(-LAMBDA_DRUG * (t - t_treat))
        dudt = r * uu * (1.0 - uu) - alpha * c_t * uu
    return [dudt]


def _simulate_ode(times_days, u0, r, alpha, t_treat: float):
    """Intègre l'ODE à deux phases et retourne u aux instants demandés (strictement croissants)."""
    times = np.unique(np.asarray(times_days, dtype=float))
    if len(times) < 2:
        return np.array([u0])
    sol = solve_ivp(
        _ode_kpp_phases,
        t_span=(float(times[0]), float(times[-1])),
        y0=[u0], args=(r, alpha, float(t_treat)),
        t_eval=times.tolist(),
        method="RK45", rtol=1e-6, atol=1e-9, max_step=1.0,
    )
    return sol.y[0] if sol.success else np.full(len(times), np.nan)


def simulate_ode_volume_continuous(r, alpha, t_treat, V_baseline_mm3,
                                   t_end_days, n_points=400,
                                   V_cap_mm3=None, vols_hint=None):
    """
    Solution ODE 0D en volume [mm³] sur une grille temporelle dense.
    Même modèle que estimate_patient_parameters (1 cycle chimio après t_treat).
    """
    vols_hint = list(vols_hint or []) + [float(V_baseline_mm3)]
    V_cap = float(V_cap_mm3) if V_cap_mm3 is not None else max(
        max(vols_hint) * 2.5, V_baseline_mm3 * 2.5, 1.0,
    )
    u0 = float(np.clip(V_baseline_mm3 / V_cap, 1e-6, 1.0))
    t_end = max(float(t_end_days), 1e-6)
    n_points = max(int(n_points), 2)
    times = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        _ode_kpp_phases,
        t_span=(0.0, t_end),
        y0=[u0], args=(float(r), float(alpha), float(t_treat)),
        t_eval=times, method="RK45", rtol=1e-6, atol=1e-9, max_step=0.5,
    )
    if not sol.success:
        return times, np.full(len(times), np.nan)
    return sol.t, sol.y[0] * V_cap


def _estimate_D_from_shape(sphericity: float, max_diam_mm: float,
                            elongation: float) -> float:
    """
    Heuristique physique pour D [mm²/jour] depuis la morphologie.
    Étalonnage littérature : D ∈ [0.001, 15] mm²/jour (Swanson 2000).
    """
    irreg  = max(0.0, 1.0 - sphericity)
    extent = np.clip(max_diam_mm / 100.0, 0.05, 5.0)
    aniso  = max(0.0, 1.0 - elongation)
    return float(np.clip(0.05 * irreg * extent * (1.0 + 0.5 * aniso), 0.001, 15.0))


def estimate_patient_parameters(volumes_mm3, times_days,
                                 rad_baseline: dict = None) -> dict:
    """
    Estime D, r, α, t_treat pour un patient via ajustement de l'ODE normalisée
    par évolution différentielle (robuste aux minima locaux).

    D        : depuis morphologie radiomique (heuristique de Swanson)
    r, α, t_treat : optimisation sur les volumes normalisés longitudinaux
    """
    vols  = np.asarray(volumes_mm3, dtype=float)
    times = np.asarray(times_days,  dtype=float)

    base = dict(D=np.nan, r=np.nan, alpha=np.nan, t_treat=np.nan, r_squared=np.nan,
                v_front=np.nan, lambda_front=np.nan, success=False,
                n_timepoints=int(len(times)))

    if len(times) < 2 or vols[0] <= 0:
        return base

    # Normalisation u ∈ (0, 1]
    V_cap = max(vols.max() * 2.5, vols[0] * 2.5, 1.0)
    u_obs = np.clip(vols / V_cap, 1e-6, 1.0)
    u0    = float(u_obs[0])

    t_treat_hi = max(float(times[-1]) * 0.8, 1e-6)

    def objective(params):
        r_v, a_v, t_tr = params
        if r_v <= 0 or a_v < 0 or t_tr < 0:
            return 1e10
        u_sim = _simulate_ode(times, u0, r_v, a_v, t_tr)
        return float(np.sum((u_sim - u_obs) ** 2)) if not np.any(np.isnan(u_sim)) else 1e10

    try:
        opt = differential_evolution(
            objective,
            bounds=[
                (0.005, 0.1),           # r
                (0.0, 0.2),            # alpha
                (0.0, t_treat_hi),     # t_treat
            ],
            seed=42, maxiter=400, tol=1e-9,
            mutation=(0.6, 1.8), recombination=0.9,
            popsize=15, workers=1, polish=True,
        )
        r_est, alpha_est, t_treat_est = (
            float(opt.x[0]), float(opt.x[1]), float(opt.x[2])
        )
    except Exception as exc:
        log.debug(f"Optimisation : {exc}")
        # Fallback heuristique
        r_est     = float(np.clip(
            np.log(u_obs[-1] / u_obs[0]) / (times[-1] - times[0] + 1e-10),
            0.001, 0.5
        ))
        alpha_est = 0.01
        t_treat_est = float(times[int(np.argmax(vols))])

    # R²
    u_fit  = _simulate_ode(times, u0, r_est, alpha_est, t_treat_est)
    ss_res = float(np.sum((u_obs - u_fit) ** 2))
    ss_tot = float(np.sum((u_obs - u_obs.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # D depuis morphologie
    sph  = float((rad_baseline or {}).get("Sphericity",        0.5))
    diam = float((rad_baseline or {}).get("Maximum3DDiameter", 50.0))
    elo  = float((rad_baseline or {}).get("Elongation",        0.5))
    D    = _estimate_D_from_shape(sph, diam, elo)

    return dict(
        D=round(D, 5), r=round(r_est, 5), alpha=round(alpha_est, 5),
        t_treat=round(t_treat_est, 4),
        r_squared=round(r2, 4),
        v_front=round(2.0 * np.sqrt(r_est * D), 4),
        lambda_front=round(np.sqrt(D / max(r_est, 1e-10)), 4),
        u0=round(u0, 4), V_baseline=float(vols[0]), V_cap=float(V_cap),
        n_timepoints=int(len(times)),
        success=bool(r2 > 0.3 and not np.isnan(r_est)),
    )


def build_ground_truth(vol_df: pd.DataFrame, clin: pd.DataFrame,
                        rad: pd.DataFrame) -> pd.DataFrame:
    """Estime D, r, α pour chaque patient."""
    rad_t1c_base = rad[
        (rad["Mask"] == "mask_tumor") &
        (rad["Sequence"] == "t1c") &
        (rad["Timepoint_Label"] == "baseline")
    ].set_index("Patient_ID")

    records  = []
    patients = vol_df["Patient_ID"].unique()

    print(f"\n  {'Patient':<8} {'N':>3} {'D':>9} {'r':>9} "
          f"{'α':>9} {'t_tr':>8} {'R²':>7} {'OK':>3}")
    print("  " + "─" * 58)

    for pid in sorted(patients):
        pat    = vol_df[vol_df["Patient_ID"] == pid].sort_values("tp_int")
        vols   = pat["Volume_mm3"].values
        times  = pat["time_days"].values
        rad_row= rad_t1c_base.loc[pid].to_dict() if pid in rad_t1c_base.index else {}

        params = estimate_patient_parameters(vols, times, rad_row)
        params["Patient_ID"] = pid
        clin_r = clin[clin["Patient_ID"] == pid]
        params["Prior_Chemo"] = int(clin_r["Prior_Chemo"].values[0]) if len(clin_r) else 1
        params["Dose_Gy"]     = float(clin_r["Dose_Gy"].values[0])   if len(clin_r) else 20.0
        records.append(params)

        ok = "✓" if params["success"] else "✗"
        ttr = params.get("t_treat", float("nan"))
        print(f"  {pid:<8} {params['n_timepoints']:>3} "
              f"{params['D']:>9.4f} {params['r']:>9.5f} "
              f"{params['alpha']:>9.4f} {ttr:>8.2f} {params['r_squared']:>7.3f} {ok}")

    gt = pd.DataFrame(records)
    log.info(f"Vérité terrain : {gt['success'].sum()}/{len(gt)} patients valides (R²>0.3)")
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MATRICE DE FEATURES (X)
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(gt_df: pd.DataFrame, rad: pd.DataFrame,
                          clin: pd.DataFrame,
                          nifti_feats: pd.DataFrame = None):
    """
    Construit X (baseline uniquement) en fusionnant radiomics T1CE / FLAIR,
    ratio tumeur/oedème et cliniques, puis ne conserve que FEATURE_MATRIX_COLS
    (6 variables). Les autres colonnes sont exclues pour limiter le bruit.

    nifti_feats : conservé pour compatibilité d'appel ; non fusionné dans X
    (aucune des 6 variables ne provient de ce tableau).
    """
    # Radiomics T1CE tumeur
    r_t1c = rad[
        (rad["Mask"] == "mask_tumor") &
        (rad["Sequence"] == "t1c") &
        (rad["Timepoint_Label"] == "baseline")
    ][["Patient_ID"] + RADIOMICS_COLS].copy()
    r_t1c.columns = ["Patient_ID"] + [f"t1c_{c}" for c in RADIOMICS_COLS]

    # Radiomics FLAIR oedème
    r_fla = rad[
        (rad["Mask"] == "mask_oedema") &
        (rad["Sequence"] == "fla") &
        (rad["Timepoint_Label"] == "baseline")
    ][["Patient_ID"] + RADIOMICS_COLS].copy()
    r_fla.columns = ["Patient_ID"] + [f"fla_{c}" for c in RADIOMICS_COLS]

    ratio = r_t1c[["Patient_ID", "t1c_MeshVolume"]].merge(
        r_fla[["Patient_ID", "fla_MeshVolume"]], on="Patient_ID", how="left"
    )
    ratio["ratio_T_O_Volume"] = ratio["t1c_MeshVolume"] / (ratio["fla_MeshVolume"] + 1e-3)

    c_feats = clin[["Patient_ID"] + CLINICAL_COLS].copy()

    X = gt_df[["Patient_ID"]].copy()
    X = X.merge(r_t1c,                                    on="Patient_ID", how="left")
    X = X.merge(r_fla,                                    on="Patient_ID", how="left")
    X = X.merge(ratio[["Patient_ID","ratio_T_O_Volume"]], on="Patient_ID", how="left")
    X = X.merge(c_feats,                                  on="Patient_ID", how="left")
    if nifti_feats is not None and not nifti_feats.empty:
        log.info("nifti_feats ignoré pour X (matrice limitée à FEATURE_MATRIX_COLS).")

    missing = [c for c in FEATURE_MATRIX_COLS if c not in X.columns]
    if missing:
        log.warning(f"Colonnes attendues absentes (NaN) : {missing}")
        for c in missing:
            X[c] = np.nan

    feature_cols = list(FEATURE_MATRIX_COLS)
    X = X[["Patient_ID"] + feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[feature_cols] = X[feature_cols].apply(lambda col: col.fillna(col.median()))

    log.info(f"Matrice X : {X.shape[0]} patients × {len(feature_cols)} features "
             f"(colonnes : {', '.join(feature_cols)})")
    return X, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MODÈLE ML : RANDOM FOREST MULTI-SORTIES
# ─────────────────────────────────────────────────────────────────────────────

def load_augmented_training_tables(augmented_dir: Path):
    """Charge X_augmented.csv et y_augmented.csv (apres augment_biomedical_dataset.py)."""
    augmented_dir = Path(augmented_dir)
    x_path = augmented_dir / "X_augmented.csv"
    y_path = augmented_dir / "y_augmented.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Fichiers augmentes introuvables dans {augmented_dir.resolve()}/\n"
            "Lancez d'abord : python augment_biomedical_dataset.py"
        )
    return pd.read_csv(x_path), pd.read_csv(y_path)


def _build_augmented_train_test(
    X_aug: pd.DataFrame,
    y_aug: pd.DataFrame,
    feature_cols: list,
    test_size: float = 0.2,
):
    """
    Train = patients reels (hors test) + tous les synthetiques.
    Test  = uniquement patients reels (jamais de SYN_*).
    """
    y_cols = ["Patient_ID"] + TARGET_PARAMS
    merged = X_aug.merge(y_aug[y_cols], on="Patient_ID", how="inner")
    if "is_synthetic" not in merged.columns:
        raise ValueError("Colonne is_synthetic absente dans X_augmented.csv")

    feat = [c for c in feature_cols if c in merged.columns]
    if not feat:
        feat = [
            c for c in merged.columns
            if c not in ("Patient_ID", "is_synthetic", "source_id") + TARGET_PARAMS
        ]

    real = merged[merged["is_synthetic"].astype(int) == 0].copy()
    syn  = merged[merged["is_synthetic"].astype(int) == 1].copy()

    if len(real) < 6:
        raise ValueError(
            f"Pas assez de patients reels dans les donnees augmentees ({len(real)})."
        )

    X_real = real[feat].values.astype(float)
    Y_real = real[TARGET_PARAMS].values.astype(float)
    pids_real = real["Patient_ID"].values

    X_tr_r, X_te, Y_tr_r, Y_te, pid_tr_r, pid_te = train_test_split(
        X_real, Y_real, pids_real,
        test_size=test_size, random_state=42,
    )

    if len(syn) > 0:
        X_syn = syn[feat].values.astype(float)
        Y_syn = syn[TARGET_PARAMS].values.astype(float)
        X_tr = np.vstack([X_tr_r, X_syn])
        Y_tr = np.vstack([Y_tr_r, Y_syn])
    else:
        X_tr, Y_tr = X_tr_r, Y_tr_r

    meta = dict(
        n_real=len(real),
        n_synthetic=len(syn),
        n_train=len(X_tr),
        n_train_real=len(X_tr_r),
        n_train_synthetic=len(syn),
        n_test=len(X_te),
        feature_cols=feat,
        X_real_scaled_source=X_real,
        Y_real=Y_real,
    )
    return X_tr, Y_tr, X_te, Y_te, pid_te, meta


def build_rf_model(n_estimators: int = 300, max_depth: int = 5,
                   min_samples_leaf: int = 2) -> MultiOutputRegressor:
    """
    Random Forest enveloppé dans MultiOutputRegressor.

    Un modèle indépendant est ajusté pour chaque cible (D, r, α),
    ce qui permet d'utiliser les feature importances par paramètre.

    Hyperparamètres :
        n_estimators      : nombre d'arbres (300 = bon compromis biais/variance)
        max_depth         : profondeur max (5 évite le sur-apprentissage sur ~40 patients)
        min_samples_leaf  : régularisation supplémentaire
        max_features      : 2 (splits très simples, limite la mémorisation du bruit)
    """
    base = RandomForestRegressor(
        n_estimators    = n_estimators,
        max_depth       = max_depth,
        min_samples_leaf= min_samples_leaf,
        max_features    = 2,
        bootstrap       = True,
        oob_score       = True,   # score out-of-bag gratuit
        random_state    = 42,
        n_jobs          = -1,
    )
    return MultiOutputRegressor(base, n_jobs=-1)


def train_and_evaluate(
    X_df:         pd.DataFrame,
    gt_df:        pd.DataFrame,
    feature_cols: list,
    n_estimators: int   = 300,
    max_depth:    int   = 5,
    min_samples_leaf: int = 2,
    test_size:    float = 0.2,
    n_cv_folds:   int   = 5,
    output_dir:   Path  = Path("results_rfml"),
    augmented_dir: Path = None,
):
    """
    Entraîne le Random Forest multi-sorties et évalue ses performances.

    Étapes :
      1. Filtrage des patients avec estimation valide (success) + logs d'effectifs
      2. Normalisation StandardScaler
      3. Split train / test (+ tailles affichées)
      4. CV R² sur tout le jeu valide (stabilité, pas un seul split test)
      5. Entraînement RF sur le train ; OOB ; CV R² sur le train seul
      6. Baseline test : R² si on prédit la moyenne(train) partout sur le test
      7. Évaluation test (MAE, RMSE, R² RF vs baseline)
      8. Feature importances top-10 par paramètre
      9. Sauvegarde (pde_ml_model.pkl, metrics.csv enrichi)
    """
    use_aug = augmented_dir is not None
    n_gt_rows     = len(gt_df)
    n_success_gt  = int(gt_df["success"].sum()) if "success" in gt_df.columns else n_gt_rows
    aug_meta      = None

    # ── 1. Dataset (standard ou augmente) ─────────────────────────────────────
    if use_aug:
        X_aug, y_aug = load_augmented_training_tables(augmented_dir)
        X_tr, Y_tr, X_te, Y_te, pid_te, aug_meta = _build_augmented_train_test(
            X_aug, y_aug, feature_cols, test_size=test_size,
        )
        feature_cols = aug_meta["feature_cols"]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        X_real = aug_meta["X_real_scaled_source"]
        Y_real = aug_meta["Y_real"]
        X_scaled_real = scaler.transform(X_real)
        n_valid_ml = aug_meta["n_real"]
        n_train = aug_meta["n_train"]
        n_test = aug_meta["n_test"]
        log.info(
            "Mode AUGMENTE : train=%d (%d reels + %d synth.) | test=%d (reels uniquement)",
            n_train, aug_meta["n_train_real"], aug_meta["n_train_synthetic"], n_test,
        )
        print(
            f"\n  Entrainement augmente : {n_train} lignes train "
            f"({aug_meta['n_train_real']} reels + {aug_meta['n_train_synthetic']} synthetiques)"
        )
        print(f"  Test (reels seulement) : {n_test} patients")
    else:
        valid         = gt_df[gt_df["success"]][["Patient_ID"] + TARGET_PARAMS].dropna()
        n_valid_ml    = len(valid)
        X_sub         = X_df[X_df["Patient_ID"].isin(valid["Patient_ID"])]
        X_sub         = X_sub.set_index("Patient_ID").reindex(valid["Patient_ID"])
        X_arr         = X_sub[feature_cols].values
        Y_arr         = valid[TARGET_PARAMS].values

        log.info(
            f"Effectifs GT -> total : {n_gt_rows} | success=True : {n_success_gt} | "
            f"ML valides : {n_valid_ml}"
        )
        log.info(f"Dataset ML : {X_arr.shape[0]} patients x {X_arr.shape[1]} features")

        if X_arr.shape[0] < 6:
            log.error(
                "< 6 patients valides — entrainement impossible.\n"
                "Conseils : reduire --dt_days, verifier les masques, "
                "ou lancer augment_biomedical_dataset.py puis --use_augmented."
            )
            return None, None, None, None

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
        X_tr, X_te, Y_tr, Y_te, pid_tr, pid_te = train_test_split(
            X_scaled, Y_arr, valid["Patient_ID"].values,
            test_size=test_size, random_state=42,
        )
        X_scaled_real = X_scaled
        Y_real = Y_arr
        n_train, n_test = X_tr.shape[0], X_te.shape[0]
        log.info(f"Split train/test : {n_train} train | {n_test} test (test_size={test_size})")

    # ── 3b. CV R² sur patients REELS uniquement (pas les synthetiques) ─────────
    rf_template = build_rf_model(n_estimators, max_depth, min_samples_leaf)
    n_splits_full = min(max(2, n_cv_folds), X_scaled_real.shape[0])
    kfold_full    = KFold(n_splits=n_splits_full, shuffle=True, random_state=43)
    cv_full_results = {}
    label_cv = "reels" if use_aug else "valides"
    print(f"\n  Cross-validation {n_splits_full}-fold (patients {label_cv}, "
          f"{X_scaled_real.shape[0]} lignes) :")
    for i, param in enumerate(TARGET_PARAMS):
        scores_f = cross_val_score(
            clone(rf_template.estimator),
            X_scaled_real, Y_real[:, i],
            cv=kfold_full, scoring="r2",
        )
        cv_full_results[param] = scores_f
        print(f"    {param:>6} : R² = {scores_f.mean():.3f} ± {scores_f.std():.3f}")

    # ── 4. Entraînement ───────────────────────────────────────────────────────
    model = build_rf_model(n_estimators, max_depth, min_samples_leaf)
    model.fit(X_tr, Y_tr)

    # ── 5. Score OOB (calculé pendant l'entraînement, sans data de test) ──────
    print("\n  Scores OOB par paramètre (biais interne Random Forest) :")
    for i, param in enumerate(TARGET_PARAMS):
        rf = model.estimators_[i]
        if hasattr(rf, "oob_score_"):
            print(f"    {param:>6} : OOB R² = {rf.oob_score_:.3f}")

    # ── 6. Cross-validation K-fold sur le train set ───────────────────────────
    n_splits = min(max(2, n_cv_folds), X_tr.shape[0])
    kfold    = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"\n  Cross-validation {n_splits}-fold (sous-ensemble train seul, {n_train} patients) :")
    cv_results = {}
    for i, param in enumerate(TARGET_PARAMS):
        scores = cross_val_score(
            clone(model.estimators_[i]),
            X_tr, Y_tr[:, i],
            cv=kfold, scoring="r2",
        )
        cv_results[param] = scores
        print(f"    {param:>6} : R² = {scores.mean():.3f} ± {scores.std():.3f}")

    # ── 6b. Baseline : prédire la moyenne des cibles sur le train (constante sur le test)
    baseline_r2_test = {}
    print("\n  Baseline test — prédire constante = moyenne(train) (R² sur le test) :")
    for i, param in enumerate(TARGET_PARAMS):
        mu_tr   = float(Y_tr[:, i].mean())
        y_dummy = np.full(Y_te.shape[0], mu_tr)
        r2_b    = r2_score(Y_te[:, i], y_dummy)
        baseline_r2_test[param] = r2_b
        print(f"    {param:>6} : R² = {r2_b:+.3f}  (si > R² du RF, le modèle est pire que cette moyenne)")

    # ── 7. Évaluation test ────────────────────────────────────────────────────
    Y_pred = model.predict(X_te)

    print("\n" + "═" * 58)
    print("  RÉSULTATS — SET DE TEST")
    print("═" * 58)
    print(f"  {'Param':<8} {'MAE':>10} {'RMSE':>10} {'R² RF':>10} {'R² μ_tr':>10}")
    print("  " + "─" * 52)
    metrics = []
    for i, param in enumerate(TARGET_PARAMS):
        mae  = mean_absolute_error(Y_te[:, i], Y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(Y_te[:, i], Y_pred[:, i]))
        r2   = r2_score(Y_te[:, i], Y_pred[:, i])
        r2_b = baseline_r2_test[param]
        print(f"  {param:<8} {mae:>10.5f} {rmse:>10.5f} {r2:>10.3f} {r2_b:>10.3f}")
        metrics.append(dict(
            param=param, MAE=mae, RMSE=rmse, R2=r2,
            R2_baseline_train_mean=r2_b,
            OOB_R2=model.estimators_[i].oob_score_,
            cv_R2_train_mean=cv_results[param].mean(),
            cv_R2_train_std=cv_results[param].std(),
            cv_R2_full_mean=cv_full_results[param].mean(),
            cv_R2_full_std=cv_full_results[param].std(),
            n_gt_rows=n_gt_rows,
            n_success_gt=n_success_gt,
            n_valid_ml=n_valid_ml,
            n_train=n_train,
            n_test=n_test,
            n_train_synthetic=aug_meta["n_train_synthetic"] if use_aug else 0,
            training_mode="augmented" if use_aug else "standard",
        ))
    print("═" * 58)
    metrics_df = pd.DataFrame(metrics)

    # Prédictions détaillées test
    pred_rows = [
        {"Patient_ID": pid, "split": "test",
         **{f"{param}_true": Y_te[j, k] for k, param in enumerate(TARGET_PARAMS)},
         **{f"{param}_pred": Y_pred[j, k] for k, param in enumerate(TARGET_PARAMS)}}
        for j, pid in enumerate(pid_te)
    ]
    preds_df = pd.DataFrame(pred_rows)

    # ── 8. Feature importances top-10 par paramètre ───────────────────────────
    print("\n  TOP 10 FEATURES PAR PARAMÈTRE (Mean Decrease Impurity)")
    print("─" * 58)
    for i, param in enumerate(TARGET_PARAMS):
        imp = pd.Series(
            model.estimators_[i].feature_importances_,
            index=feature_cols,
        ).nlargest(10)
        print(f"\n  {param} :")
        for feat, val in imp.items():
            bar = "█" * max(1, int(val * 300))
            print(f"    {feat:<42} {val:.4f} {bar}")

    # ── 9. Sauvegarde ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model, "scaler": scaler, "feature_cols": feature_cols,
        "model_type": "random_forest",
        "training_mode": "augmented" if use_aug else "standard",
        "hyperparams": {"n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf},
    }
    joblib.dump(bundle, output_dir / "pde_ml_model.pkl")
    if use_aug:
        joblib.dump(bundle, output_dir / "pde_ml_model_augmented.pkl")
        metrics_df.to_csv(output_dir / "metrics_augmented.csv", index=False)
        preds_df.to_csv(output_dir / "predictions_test_augmented.csv", index=False)
    else:
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)
        preds_df.to_csv(output_dir / "predictions_test.csv", index=False)
    log.info("Modele Random Forest -> %s", output_dir / "pde_ml_model.pkl")

    return model, scaler, metrics_df, preds_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — SIMULATION EDP 3D
# ─────────────────────────────────────────────────────────────────────────────

# Étiquettes de tissus
TISSUE_BG, TISSUE_CSF, TISSUE_GM, TISSUE_WM = 0, 1, 2, 3


def segment_tissue_from_t1(t1_path: Path, brain_mask: np.ndarray = None,
                            ref_shape: tuple = None) -> np.ndarray:
    """
    Segmente le cerveau en 3 tissus (CSF / GM / WM) par K-means sur l'intensité T1.

    Retourne une carte d'étiquettes (même forme que le T1) :
        0 = hors cerveau, 1 = CSF, 2 = matière grise, 3 = matière blanche.
    Renvoie None si le T1 est absent/illisible ou incompatible avec ref_shape.
    """
    if t1_path is None or not Path(t1_path).exists():
        return None
    try:
        import nibabel as nib
        from sklearn.cluster import KMeans
    except ImportError as e:
        log.warning(f"Segmentation tissus indisponible ({e}).")
        return None

    try:
        t1 = nib.load(str(t1_path)).get_fdata()
    except Exception as e:
        log.warning(f"T1 illisible ({t1_path}) : {e}")
        return None

    if ref_shape is not None and t1.shape != tuple(ref_shape):
        log.warning(f"T1 {t1.shape} ≠ grille simulation {tuple(ref_shape)} — tissus ignorés.")
        return None

    # Masque cerveau : fourni, sinon seuil simple sur intensité positive
    if brain_mask is not None and brain_mask.shape == t1.shape:
        mask = brain_mask > 0
    else:
        mask = t1 > (0.05 * float(t1.max()) if t1.max() > 0 else 0.0)

    vals = t1[mask].reshape(-1, 1)
    if vals.size < 10:
        return None

    km = KMeans(n_clusters=3, n_init=4, random_state=0).fit(vals)
    centers = km.cluster_centers_.ravel()
    # Trie les clusters par intensité croissante : CSF < GM < WM
    order = np.argsort(centers)
    remap = {int(order[0]): TISSUE_CSF,
             int(order[1]): TISSUE_GM,
             int(order[2]): TISSUE_WM}

    tissue = np.zeros(t1.shape, dtype=np.uint8)
    labels_flat = np.vectorize(remap.get)(km.labels_)
    tissue[mask] = labels_flat.astype(np.uint8)

    n_csf = int((tissue == TISSUE_CSF).sum())
    n_gm  = int((tissue == TISSUE_GM).sum())
    n_wm  = int((tissue == TISSUE_WM).sum())
    log.info(f"Tissus (T1) : CSF={n_csf:,}  GM={n_gm:,}  WM={n_wm:,} voxels")
    return tissue


def build_param_fields(tissue: np.ndarray, D: float, r: float, alpha: float,
                        d_wm_gm_ratio: float = 5.0,
                        r_wm_gm_ratio: float = 1.0,
                        alpha_wm_gm_ratio: float = 1.0,
                        csf_factor: float = 0.1):
    """
    Construit les champs 3D D(x), r(x), alpha(x) à partir de la carte de tissus.

    Convention : la matière grise (GM) est la référence (multiplicateur 1.0).
        WM  : ratios *_wm_gm_ratio (diffusion typiquement 5–10× plus rapide).
        CSF : diffusion/prolifération fortement réduites (csf_factor).
        Hors cerveau / non classé : comportement GM par défaut.

    Retourne (D_field, r_field, alpha_field) en float64.
    """
    mult_D     = np.ones(tissue.shape, dtype=np.float64)
    mult_r     = np.ones(tissue.shape, dtype=np.float64)
    mult_alpha = np.ones(tissue.shape, dtype=np.float64)

    wm = tissue == TISSUE_WM
    mult_D[wm]     = d_wm_gm_ratio
    mult_r[wm]     = r_wm_gm_ratio
    mult_alpha[wm] = alpha_wm_gm_ratio

    csf = tissue == TISSUE_CSF
    mult_D[csf]     = csf_factor
    mult_r[csf]     = csf_factor
    mult_alpha[csf] = 1.0  # le traitement agit aussi sur d'éventuelles cellules en CSF

    return D * mult_D, r * mult_r, alpha * mult_alpha


def _divergence_D_grad(u: np.ndarray, D_field: np.ndarray, dx: float) -> np.ndarray:
    """
    Forme à flux du terme de diffusion : ∇·(D(x)·∇u), schéma volumes finis.

    D aux faces = moyenne arithmétique des voxels voisins.
    Conditions de Neumann (flux nul) imposées aux bords du domaine.
    """
    div = np.zeros_like(u)
    for ax in range(3):
        u_next = np.roll(u, -1, ax)
        D_next = np.roll(D_field, -1, ax)
        D_face = 0.5 * (D_field + D_next)
        flux   = D_face * (u_next - u) / dx          # flux à l'interface i+1/2

        # Flux nul à la frontière éloignée (pas d'effet de bord périodique)
        sl = [slice(None)] * 3; sl[ax] = -1
        flux[tuple(sl)] = 0.0

        flux_prev = np.roll(flux, 1, ax)             # flux à l'interface i-1/2
        div += (flux - flux_prev) / dx
    return div


def simulate_pde_3d(D: float, r: float, alpha: float,
                   u0_3d: np.ndarray, dx_mm: float,
                   t_days: float, c0: float = C0,
                   lam: float = LAMBDA_DRUG,
                   t_treat: float = 0.0,
                   voxel_volume_mm3: float = None,
                   n_snapshots: int = 0,
                   D_field: np.ndarray = None,
                   r_field: np.ndarray = None,
                   alpha_field: np.ndarray = None):
    """
    Version corrigée : 
    1. Utilise la somme intégrale de 'u' pour des courbes fluides (plus de plateau à zéro).
    2. Simule un protocole réel de 6 cycles (une dose tous les 28 jours).

    voxel_volume_mm3 : volume d'un voxel en mm³ (ex. prod(zooms) du NIfTI).
        Si fourni, les volumes rapportés coïncident avec ceux des masques (extract_volume_from_mask).
        Sinon : dx_mm**3 (grille isotrope).
    n_snapshots : si > 0, capture ~n_snapshots champs 3D de densité u répartis
        uniformément dans le temps (pour construire une série temporelle 4D).
    D_field/r_field/alpha_field : champs 3D optionnels (mêmes dims que u0_3d).
        Si D_field est fourni, la diffusion utilise la forme à flux ∇·(D(x)∇u)
        et r/alpha utilisent leurs champs s'ils sont fournis (sinon scalaires).

    Retourne (u_final, volumes, snapshots, snapshot_days) où :
        snapshots     : liste de champs 3D float32 (vide si n_snapshots <= 0)
        snapshot_days : instants (jours) correspondant à chaque snapshot
    """
    use_fields = D_field is not None
    D_max  = float(D_field.max()) if use_fields else float(D)
    dt_cfl  = (dx_mm ** 2) / (6.0 * D_max + 1e-12)
    dt      = min(1.0, dt_cfl * 0.9)
    n_steps = max(1, int(np.ceil(t_days / dt)))
    dt      = t_days / n_steps

    u       = u0_3d.astype(np.float64).copy()
    vox_vol = float(voxel_volume_mm3) if voxel_volume_mm3 is not None else float(dx_mm ** 3)
    
    # CORRECTION 1 : On utilise np.sum(u) dès le départ pour la fluidité
    volumes = [float(np.sum(u) * vox_vol)]

    # Indices de pas à capturer pour la série temporelle 4D (inclut l'état initial et final)
    snapshots, snapshot_days = [], []
    capture_steps = set()
    if n_snapshots and n_snapshots > 0:
        capture_steps = set(
            int(s) for s in np.linspace(0, n_steps, int(n_snapshots))
        )
        if 0 in capture_steps:
            snapshots.append(u.astype(np.float32).copy())
            snapshot_days.append(0.0)

    log.info(f"Simulation EDP 3D | Protocole Multi-cycles | D={D:.4f}  r={r:.5f}  α={alpha:.4f}")

    for step in range(n_steps):
        t_now = step * dt
        
        # CORRECTION 2 : Modélisation d'un protocole de 6 cycles (Témozolomide-like)
        c_t = 0.0
        if t_now >= t_treat:
            # On simule 6 cycles de traitement espacés de 40.0 jours
            for i in range(6):
                t_cycle_start = t_treat + (i *70.0)  # 40 jours = 4 semaines = 40 jours pour être conservateur)
                if t_now >= t_cycle_start:
                    # Chaque cycle ajoute sa propre concentration qui décroît
                    c_t += c0 * np.exp(-lam * (t_now - t_cycle_start))

        # Terme de diffusion
        if use_fields:
            # Forme à flux ∇·(D(x)∇u) — gère D variable dans l'espace (Neumann intégré)
            diff = _divergence_D_grad(u, D_field, dx_mm)
        else:
            # Laplacien homogène (D scalaire)
            lap = (
                  np.roll(u,-1,0)+np.roll(u,1,0)
                + np.roll(u,-1,1)+np.roll(u,1,1)
                + np.roll(u,-1,2)+np.roll(u,1,2)
                - 6.0 * u
            ) / dx_mm ** 2
            # Conditions de Neumann (bords du cerveau)
            for ax in range(3):
                s0 = [slice(None)]*3; s0[ax] = 0
                s1 = [slice(None)]*3; s1[ax] = 1
                sm = [slice(None)]*3; sm[ax] = -1
                s2 = [slice(None)]*3; s2[ax] = -2
                lap[tuple(s0)] = lap[tuple(s1)]
                lap[tuple(sm)] = lap[tuple(s2)]
            diff = D * lap

        # Termes de réaction et de traitement (champs si fournis, sinon scalaires)
        r_eff     = r_field     if r_field     is not None else r
        alpha_eff = alpha_field if alpha_field is not None else alpha

        # Mise à jour de la densité tumorale
        u = np.clip(u + dt*(diff + r_eff*u*(1.0-u) - alpha_eff*c_t*u), 0.0, 1.0)
        
        # CORRECTION 1 (suite) : Somme intégrale pour suivre l'évolution sans seuil brutal
        volumes.append(float(np.sum(u) * vox_vol))

        # Capture du champ 3D pour la série temporelle 4D
        if (step + 1) in capture_steps:
            snapshots.append(u.astype(np.float32).copy())
            snapshot_days.append((step + 1) * dt)

    return u, volumes, snapshots, snapshot_days


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PRÉDICTION + SIMULATION NOUVEAU PATIENT
# ─────────────────────────────────────────────────────────────────────────────

def predict_new_patient(model_pkl: Path, patient_dir: Path, patient_id: str,
                         simulate_days: float = 30.0,
                         output_dir: Path = Path("results_rfml"),
                         dt_days: float = 56.0,
                         bin_threshold: float = 0.5,
                         n_snapshots: int = 20,
                         use_tissue: bool = True,
                         d_wm_gm_ratio: float = 5.0,
                         r_wm_gm_ratio: float = 1.0,
                         alpha_wm_gm_ratio: float = 1.0,
                         csf_factor: float = 0.1) -> dict:
    """
    1. Extrait les features NIfTI depuis les fichiers réels du patient
    2. Prédit D, r, α avec le Random Forest sauvegardé
    3. (optionnel) Segmente GM/WM/CSF depuis le T1 → champs D(x), r(x), α(x)
    4. Simule l'évolution tumorale à T+simulate_days jours
    """
    bundle       = joblib.load(str(model_pkl))
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    paths = resolve_paths(patient_dir, patient_id)

    # Features NIfTI baseline
    nii = extract_nifti_features(
        paths["seg"]["baseline"],
        paths["t1c"]["baseline"],
        paths["flair"]["baseline"],
    )
    feat_map = {
        "nii_Sphericity":     "t1c_Sphericity",
        "nii_MaxDiameter_mm": "t1c_Maximum3DDiameter",
        "nii_Elongation":     "t1c_Elongation",
        "nii_Volume_mm3":     "t1c_MeshVolume",
        "nii_T1CE_Mean":      "t1c_VoxelValue_Mean",
        "nii_T1CE_Std":       "t1c_VoxelValue_SD",
        "nii_FLAIR_Mean":     "fla_VoxelValue_Mean",
        "nii_FLAIR_Std":      "fla_VoxelValue_SD",
    }
    pat_feats = {feat_map.get(k, k): v for k, v in nii.items()}
    defaults  = {
        "t1c_Sphericity": 0.5, "t1c_Maximum3DDiameter": 50.0,
        "t1c_Elongation": 0.5, "t1c_MeshVolume": 5000.0,
        "t1c_VoxelValue_Mean": 700.0, "t1c_VoxelValue_SD": 100.0,
        "fla_VoxelValue_Mean": 800.0, "Age": 65.0, "KPS": 80.0,
        "Dose_Gy": 20.0, "Fractions": 1.0, "Prior_Chemo": 1.0,
        "Histology_enc": 0.0, "N_metastases": 1.0, "WHO_PS": 1.0,
        "ratio_T_O_Volume": 0.5,
    }
    for k, v in defaults.items():
        pat_feats.setdefault(k, v)

    x_sc  = scaler.transform(
        np.array([[pat_feats.get(f, 0.0) for f in feature_cols]])
    )
    y_pred = model.predict(x_sc)[0]
    D_pred, r_pred, alpha_pred = (
        max(float(y_pred[0]), 0.001),
        max(float(y_pred[1]), 1e-5),
        max(float(y_pred[2]), 0.0),
    )
    t_treat_pred = float(y_pred[3]) if len(y_pred) >= 4 else 0.0
    t_treat_pred = max(0.0, t_treat_pred)

    print(f"\n  ── Paramètres prédits — {patient_id} ─────────────────────────")
    print(f"  D       = {D_pred:.5f}  mm²/jour")
    print(f"  r       = {r_pred:.6f}  /jour")
    print(f"  α       = {alpha_pred:.5f}  /jour")
    print(f"  t_treat = {t_treat_pred:.2f}  j (chimiothérapie active après ce délai)")
    print(f"  v_front = 2√(rD) = {2*np.sqrt(r_pred*D_pred):.4f} mm/jour")
    print(f"  λ_front = √(D/r) = {np.sqrt(D_pred/r_pred):.2f}  mm")

    # Condition initiale depuis masque baseline (u=1 dans la tumeur : même volumétrie que les masques)
    seg_array, dx_mm = None, 1.0
    vox_vol_mm3 = None
    if paths["seg"]["baseline"].exists():
        try:
            import nibabel as nib
            seg_img   = nib.load(str(paths["seg"]["baseline"]))
            zs        = np.asarray(seg_img.header.get_zooms()[:3], dtype=float)
            vox_vol_mm3 = float(np.prod(zs))
            dx_mm     = float(zs[0])
            seg_array = np.clip(
                (seg_img.get_fdata() > 0).astype(np.float64), 0.0, 1.0
            )
        except Exception as e:
            log.warning(f"Segmentation non chargeable : {e}")

    if seg_array is None:
        g = 40; c = g // 2
        zz, yy, xx = np.ogrid[:g, :g, :g]
        seg_array = np.where(
            np.sqrt((xx-c)**2 + (yy-c)**2 + (zz-c)**2) < 10/dx_mm,
            1.0, 0.0
        )
        log.info("Sphère synthétique utilisée (masque absent)")

    # Champs spatiaux D(x), r(x), α(x) depuis la carte de tissus (T1 baseline)
    D_field = r_field = alpha_field = None
    tissue_map = None
    if use_tissue:
        brain_arr = None
        bm_path = paths.get("brain_mask")
        if bm_path is not None and Path(bm_path).exists():
            try:
                import nibabel as nib
                bm = nib.load(str(bm_path)).get_fdata()
                if bm.shape == seg_array.shape:
                    brain_arr = (bm > 0).astype(np.uint8)
            except Exception as e:
                log.warning(f"brain_mask non chargeable : {e}")

        tissue_map = segment_tissue_from_t1(
            paths["t1"]["baseline"], brain_mask=brain_arr,
            ref_shape=seg_array.shape,
        )
        if tissue_map is not None:
            D_field, r_field, alpha_field = build_param_fields(
                tissue_map, D_pred, r_pred, alpha_pred,
                d_wm_gm_ratio=d_wm_gm_ratio,
                r_wm_gm_ratio=r_wm_gm_ratio,
                alpha_wm_gm_ratio=alpha_wm_gm_ratio,
                csf_factor=csf_factor,
            )
            log.info(
                f"Champs spatiaux activés | D_WM/GM={d_wm_gm_ratio}  "
                f"r_WM/GM={r_wm_gm_ratio}  α_WM/GM={alpha_wm_gm_ratio}  CSF×{csf_factor}"
            )
        else:
            log.info("Carte de tissus indisponible — simulation homogène (repli).")

    u_final, volumes, snapshots, snapshot_days = simulate_pde_3d(
        D_pred, r_pred, alpha_pred, seg_array, dx_mm, simulate_days,
        t_treat=t_treat_pred,
        voxel_volume_mm3=vox_vol_mm3,
        n_snapshots=n_snapshots,
        D_field=D_field, r_field=r_field, alpha_field=alpha_field,
    )
    vox_report = vox_vol_mm3 if vox_vol_mm3 is not None else float(dx_mm ** 3)
    V_i = float((seg_array > bin_threshold).sum() * vox_report)
    V_f = float((u_final > bin_threshold).sum() * vox_report)
    s   = "+" if V_f >= V_i else ""
    print(f"\n  ── Simulation T+{simulate_days:.0f}j ────────────────────────────────────")
    print(f"  Volume initial  : {V_i:>10.1f} mm³")
    print(f"  Volume final    : {V_f:>10.1f} mm³")
    print(f"  Variation       : {s}{V_f-V_i:.1f} mm³ ({s}{100*(V_f-V_i)/(V_i+1e-3):.1f}%)")

    # Sauvegarde
    output_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag([dx_mm, dx_mm, dx_mm, 1.0])
    try:
        import nibabel as nib
        # 1) Champ de densité final (continu 0–1)
        nib.save(
            nib.Nifti1Image(u_final.astype(np.float32), affine),
            str(output_dir / f"{patient_id}_u_T{int(simulate_days)}j.nii.gz"),
        )
        # 2) Masque binaire final (densité seuillée)
        final_mask = (u_final > bin_threshold).astype(np.uint8)
        nib.save(
            nib.Nifti1Image(final_mask, affine),
            str(output_dir / f"{patient_id}_mask_T{int(simulate_days)}j.nii.gz"),
        )
        # 3) Séries temporelles 4D (densité + masque binaire) si snapshots disponibles
        if snapshots:
            series_4d = np.stack(snapshots, axis=-1)  # (X, Y, Z, T)
            nib.save(
                nib.Nifti1Image(series_4d.astype(np.float32), affine),
                str(output_dir / f"{patient_id}_u_4D.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image((series_4d > bin_threshold).astype(np.uint8), affine),
                str(output_dir / f"{patient_id}_mask_4D.nii.gz"),
            )
            pd.DataFrame({
                "frame": range(len(snapshot_days)),
                "days":  snapshot_days,
            }).to_csv(output_dir / f"{patient_id}_4D_timepoints.csv", index=False)
            log.info(
                f"Série 4D : {series_4d.shape[-1]} frames "
                f"(densité + masque, seuil={bin_threshold})"
            )
        # 4) Carte de tissus (si segmentée) : 1=CSF 2=GM 3=WM
        if tissue_map is not None:
            nib.save(
                nib.Nifti1Image(tissue_map.astype(np.uint8), affine),
                str(output_dir / f"{patient_id}_tissue_map.nii.gz"),
            )
    except Exception as e:
        log.warning(f"Sauvegarde NIfTI partielle : {e}")
    pd.DataFrame({
        "step": range(len(volumes)),
        "days": np.linspace(0, simulate_days, len(volumes)),
        "volume_mm3": volumes,
    }).to_csv(output_dir / f"{patient_id}_volumes_sim.csv", index=False)

    obs_t, obs_v = collect_clinical_volumes(paths, dt_days, simulate_days)

    ode_days, ode_vols = simulate_ode_volume_continuous(
        r_pred, alpha_pred, t_treat_pred, V_i, simulate_days,
        vols_hint=list(obs_v) if obs_v.size else None,
    )
    pd.DataFrame({"days": ode_days, "volume_mm3": ode_vols}).to_csv(
        output_dir / f"{patient_id}_ode_volumes.csv", index=False,
    )
    save_ode_volume_plot(
        patient_id, ode_days, ode_vols,
        D_pred, r_pred, alpha_pred, t_treat_pred,
        output_dir,
        clinical_time_days=obs_t, clinical_volume_mm3=obs_v,
        param_source="coefficients Random Forest (patient)",
        simulate_days=simulate_days,
    )

    return dict(
        D_pred=D_pred, r_pred=r_pred, alpha_pred=alpha_pred,
        t_treat_pred=t_treat_pred,
        v_front=2*np.sqrt(r_pred*D_pred),
        lambda_front=np.sqrt(D_pred/r_pred),
        u_initial=seg_array, u_final=u_final,
        u_final_mask=(u_final > bin_threshold).astype(np.uint8),
        tissue_map=tissue_map,
        tissue_active=tissue_map is not None,
        d_wm_gm_ratio=d_wm_gm_ratio,
        r_wm_gm_ratio=r_wm_gm_ratio,
        alpha_wm_gm_ratio=alpha_wm_gm_ratio,
        csf_factor=csf_factor,
        bin_threshold=bin_threshold,
        n_snapshots=len(snapshots),
        snapshot_days=np.asarray(snapshot_days, dtype=float),
        volumes_evolution=volumes,
        V_initial_mm3=V_i, V_final_mm3=V_f,
        dx_mm=dx_mm, simulate_days=simulate_days,
        clinical_time_days=obs_t,
        clinical_volume_mm3=obs_v,
        ode_time_days=ode_days,
        ode_volume_mm3=ode_vols,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_ode_volume_plot(patient_id: str, days, volumes_mm3,
                         D, r, alpha, t_treat, output_dir: Path,
                         clinical_time_days=None, clinical_volume_mm3=None,
                         param_source: str = "fit ODE",
                         r_squared=None, simulate_days=None):
    """Courbe ODE 0D continue + points cliniques optionnels."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib absent — ode_volume.png non genere")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    days = np.asarray(days, dtype=float)
    vols = np.asarray(volumes_mm3, dtype=float)
    t_win = float(simulate_days if simulate_days is not None else days[-1])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(days, vols, color="#9B2226", lw=2.5,
            label="Volume ODE 0D (solution continue)")
    ax.fill_between(days, 0, vols, alpha=0.08, color="#9B2226")

    ct = cv = None
    if clinical_time_days is not None and clinical_volume_mm3 is not None:
        ct = np.asarray(clinical_time_days, dtype=float)
        cv = np.asarray(clinical_volume_mm3, dtype=float)
        m = ct <= t_win + 1e-9
        ct, cv = ct[m], cv[m]
        if ct.size > 0:
            o = np.argsort(ct)
            ax.scatter(ct[o], cv[o], color="#1D3557", s=85, zorder=5,
                       edgecolors="white", linewidths=1.0,
                       label="Evaluations reelles (masques)")
            ax.plot(ct[o], cv[o], color="#1D3557", ls="--", lw=1.2, alpha=0.55)

    line = (
        f"D={D:.4f} mm2/j  r={r:.5f}/j  alpha={alpha:.4f}/j  "
        f"t_treat={t_treat:.1f}j  |  {param_source}"
    )
    if r_squared is not None and not np.isnan(r_squared):
        line += f"  R2_ODE={r_squared:.3f}"

    ax.set(xlabel="Temps (jours)", ylabel="Volume tumoral (mm3)",
           title=f"Solution ODE — {patient_id} — T+{t_win:.0f}j\n{line}")
    ymin = float(np.nanmin(np.r_[vols, cv if cv is not None and cv.size else vols]))
    ymax = float(np.nanmax(np.r_[vols, cv if cv is not None and cv.size else vols]))
    pad = (ymax - ymin) * 0.08 + 1e-6
    ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax.set_xlim(0.0, t_win)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "ode_volume.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"{patient_id}: ode_volume.png -> {output_dir}")


def collect_clinical_volumes(paths: dict, dt_days: float, t_max: float):
    """Volumes masques baseline + FU pour overlay graphique."""
    obs_t, obs_v = [], []
    base_seg = paths["seg"]["baseline"]
    if base_seg.exists():
        vb = extract_volume_from_mask(base_seg)
        if vb is not None and vb > 0:
            obs_t.append(0.0)
            obs_v.append(float(vb))
    for tp_label in ("follow_up_1", "follow_up_2", "follow_up_3", "follow_up_4"):
        seg_p = paths["seg"][tp_label]
        if not seg_p.exists():
            continue
        vol_m = extract_volume_from_mask(seg_p)
        if vol_m is None or vol_m <= 0:
            continue
        t_day = float(TIMEPOINT_TO_INT[tp_label] * dt_days)
        if t_day <= t_max + 1e-9:
            obs_t.append(t_day)
            obs_v.append(float(vol_m))
    return np.asarray(obs_t, dtype=float), np.asarray(obs_v, dtype=float)


def plot_patient_ode_from_volumes(patient_id: str, vols, times, rad_baseline: dict,
                                  output_dir: Path, simulate_days: float = 200.0,
                                  n_ode_points: int = 400):
    """
    Coefficients uniques par patient via fit ODE sur tous ses timepoints
    (estimate_patient_parameters), puis courbe continue.
    """
    params = estimate_patient_parameters(vols, times, rad_baseline)
    if not params.get("success") and np.isnan(params.get("r", np.nan)):
        log.warning(f"{patient_id}: fit ODE echoue — graphique non genere")
        return params

    V0 = float(params["V_baseline"])
    days, v_ode = simulate_ode_volume_continuous(
        params["r"], params["alpha"], params["t_treat"], V0,
        simulate_days, n_points=n_ode_points,
        V_cap_mm3=params.get("V_cap"),
    )
    pd.DataFrame({"days": days, "volume_mm3": v_ode}).to_csv(
        output_dir / f"{patient_id}_ode_volumes.csv", index=False,
    )
    save_ode_volume_plot(
        patient_id, days, v_ode,
        params["D"], params["r"], params["alpha"], params["t_treat"],
        output_dir,
        clinical_time_days=times, clinical_volume_mm3=vols,
        param_source="coefficients fit ODE (patient)",
        r_squared=params.get("r_squared"),
        simulate_days=simulate_days,
    )
    return params


def generate_plots(gt_df: pd.DataFrame, preds_df: pd.DataFrame,
                   metrics_df: pd.DataFrame, sim_result: dict = None,
                   output_dir: Path = Path("results_rfml")):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib non disponible (pip install matplotlib).")
        return

    pal = {"D": "#E63946", "r": "#2A9D8F", "alpha": "#F4A261"}
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Vrai vs. Prédit ───────────────────────────────────────────────────────
    if not preds_df.empty and f"{TARGET_PARAMS[0]}_true" in preds_df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Random Forest — Vrai vs. Prédit (test set)",
                     fontsize=13, fontweight="bold")
        for ax, p in zip(axes, TARGET_PARAMS):
            yt   = preds_df[f"{p}_true"]
            yp   = preds_df[f"{p}_pred"]
            lims = [min(yt.min(), yp.min())*0.9, max(yt.max(), yp.max())*1.1]
            m    = metrics_df[metrics_df["param"] == p]
            r2s  = f"R²={m['R2'].values[0]:.3f}" if len(m) else ""
            ax.scatter(yt, yp, c=pal[p], s=90, alpha=0.85,
                       edgecolors="k", linewidths=0.5)
            ax.plot(lims, lims, "k--", lw=1, label="Parfait")
            ax.set(xlim=lims, ylim=lims,
                   xlabel=f"{p} réel", ylabel=f"{p} prédit",
                   title=f"{p}  {r2s}")
            ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "true_vs_predicted.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Distributions des paramètres GT ──────────────────────────────────────
    if not gt_df.empty and "success" in gt_df.columns:
        vg = gt_df[gt_df["success"]]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Distribution des paramètres PDE estimés (vérité terrain)",
                     fontsize=13, fontweight="bold")
        for ax, p in zip(axes, TARGET_PARAMS):
            vals = vg[p].dropna()
            if vals.empty:
                continue
            ax.hist(vals, bins=15, color=pal[p], edgecolor="white", alpha=0.9)
            ax.axvline(vals.median(), color="k", ls="--", lw=1.5,
                       label=f"Med={vals.median():.4f}")
            ax.set(xlabel=p, ylabel="Effectif", title=f"Distribution de {p}")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(output_dir / "parameter_distributions.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Évolution du volume simulé ────────────────────────────────────────────
    if sim_result and "volumes_evolution" in sim_result:
        vols = sim_result["volumes_evolution"]
        days = np.linspace(0, sim_result.get("simulate_days", len(vols)), len(vols))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(days, vols, color="#E63946", lw=2.5, label="Volume simulé (EDP 3D)")
        ax.axhline(vols[0], color="gray", ls=":", lw=1.5, label="Volume initial")
        ax.fill_between(days, vols[0], vols, alpha=0.1, color="#E63946")

        ode_t = sim_result.get("ode_time_days")
        ode_v = sim_result.get("ode_volume_mm3")
        if ode_t is not None and ode_v is not None:
            ax.plot(np.asarray(ode_t), np.asarray(ode_v), color="#9B2226", lw=2.0,
                    ls="-", alpha=0.9, label="ODE 0D (memes coefficients RF)")

        ct = sim_result.get("clinical_time_days")
        cv = sim_result.get("clinical_volume_mm3")
        t_win = float(sim_result.get("simulate_days", days[-1] if len(days) else 0.0))
        if ct is not None and cv is not None:
            ct = np.asarray(ct, dtype=float)
            cv = np.asarray(cv, dtype=float)
            if ct.size > 0 and cv.size == ct.size:
                m = ct <= t_win + 1e-9
                ct = ct[m]
                cv = cv[m]
            if ct.size > 0 and cv.size == ct.size:
                o = np.argsort(ct)
                ax.plot(ct[o], cv[o], color="#1D3557", ls="--", lw=1.3, alpha=0.65,
                        label="Suivi (baseline + FU, masques)")
                ax.scatter(ct, cv, color="#1D3557", s=85, zorder=5,
                           edgecolors="white", linewidths=1.0,
                           label="Volumes dossier (t ≤ T sim.)")

        if sim_result.get("tissue_active"):
            d_gm  = sim_result['D_pred']
            d_ratio = sim_result.get('d_wm_gm_ratio', 1.0)
            d_wm  = d_gm * d_ratio
            param_line = (
                f"D(x) hétérogène : GM={d_gm:.4f} → WM={d_wm:.4f} mm²/j (×{d_ratio:g})  "
                f"CSF×{sim_result.get('csf_factor', 0.1):g}\n"
                f"r={sim_result['r_pred']:.5f} /j  α={sim_result['alpha_pred']:.4f} /j"
            )
        else:
            param_line = (
                f"D={sim_result['D_pred']:.4f} mm²/j  "
                f"r={sim_result['r_pred']:.5f} /j  "
                f"α={sim_result['alpha_pred']:.4f} /j"
            )
        ax.set(xlabel="Temps (jours)", ylabel="Volume tumoral (mm³)",
               title=(f"Simulation EDP 3D — T+{sim_result['simulate_days']:.0f}j\n"
                      f"{param_line}"))

        ymin, ymax = float(np.min(vols)), float(np.max(vols))
        if ode_v is not None:
            ymin = min(ymin, float(np.nanmin(ode_v)))
            ymax = max(ymax, float(np.nanmax(ode_v)))
        if ct is not None and cv is not None and ct.size > 0 and cv.size == ct.size:
            ymin = min(ymin, float(np.min(cv)))
            ymax = max(ymax, float(np.max(cv)))
        pad_y = (ymax - ymin) * 0.06 + 1e-6
        ax.set_ylim(max(0.0, ymin - pad_y), ymax + pad_y)
        ax.set_xlim(0.0, max(t_win, 1e-6))

        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / "simulation_volume.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    log.info(f"Figures → {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline ML (Random Forest) — Paramètres PDE tumeur cérébrale",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",   default="17253793",
                   help="Dossier racine des patients (même dossier que le script)")
    p.add_argument("--dt_days",    type=float, default=56.0,
                   help="Intervalle en jours entre follow-ups consécutifs")
    # Hyperparamètres Random Forest
    p.add_argument("--n_estimators",     type=int, default=300)
    p.add_argument("--max_depth",        type=int, default=5)
    p.add_argument("--min_samples_leaf", type=int, default=2)
    # Évaluation
    p.add_argument("--test_size",  type=float, default=0.2)
    p.add_argument("--n_cv_folds", type=int,   default=5)
    p.add_argument("--output",     default="results_rfml")
    # Prédiction
    p.add_argument("--predict",      action="store_true")
    p.add_argument("--patient_id",   default=None, help="Ex: P04a")
    p.add_argument("--simulate_days",type=float, default=30.0)
    p.add_argument("--bin_threshold", type=float, default=0.5,
                   help="Seuil de densité u pour binariser le masque prédit")
    p.add_argument("--n_snapshots",  type=int, default=20,
                   help="Nombre de frames 3D capturées pour la série temporelle 4D "
                        "(0 = désactive la sortie 4D)")
    # Modélisation spatiale GM/WM/CSF (champs D(x), r(x), α(x))
    p.add_argument("--no_tissue", action="store_true",
                   help="Désactive la carte de tissus (simulation homogène classique)")
    p.add_argument("--d_wm_gm_ratio", type=float, default=5.0,
                   help="Ratio diffusion matière blanche / grise")
    p.add_argument("--r_wm_gm_ratio", type=float, default=1.0,
                   help="Ratio prolifération matière blanche / grise")
    p.add_argument("--alpha_wm_gm_ratio", type=float, default=1.0,
                   help="Ratio réponse au traitement matière blanche / grise")
    p.add_argument("--csf_factor", type=float, default=0.1,
                   help="Facteur réducteur de diffusion/prolifération dans le CSF")
    p.add_argument("--plot_ode", action="store_true",
                   help="Trace la solution ODE 0D avec coefficients fites sur le patient "
                        "(longitudinal) — requiert --patient_id")
    p.add_argument("--ode_points", type=int, default=400,
                   help="Nombre de points pour la courbe ODE continue")
    p.add_argument("--use_augmented", action="store_true",
                   help="Entraine le RF sur X_augmented/y_augmented (200 lignes). "
                        "Test toujours sur patients reels uniquement.")
    p.add_argument("--augmented_dir", default="results_rfml/augmented",
                   help="Dossier contenant X_augmented.csv et y_augmented.csv")
    p.add_argument("--train_augmented_only", action="store_true",
                   help="Entrainement augmente seul (sans recalculer GT/features). "
                        "Requiert --use_augmented et CSV deja generes.")
    return p.parse_args()


def main():
    args       = parse_args()
    script_dir = Path(__file__).parent.resolve()
    data_dir   = script_dir / args.data_dir
    output_dir = script_dir / args.output

    print("\n" + "═"*65)
    print("  PIPELINE ML (RANDOM FOREST) — PARAMÈTRES PDE TUMEUR")
    print("  ∂u/∂t = D·∇²u + r·u·(1−u) − α·c(t)·u")
    print("═"*65)
    print(f"  Script  : {script_dir}")
    print(f"  Données : {data_dir}  {'✓' if data_dir.exists() else '⚠ absent — CSV fallback'}")
    print(f"  Sorties : {output_dir}\n")

    if args.plot_ode:
        if not args.patient_id:
            log.error("--patient_id requis avec --plot_ode"); sys.exit(1)
        rad, clin, nifti_df = load_tabular_data(script_dir)
        vol_df = build_longitudinal_volumes(data_dir, nifti_df, dt_days=args.dt_days)
        pat = vol_df[vol_df["Patient_ID"] == args.patient_id].sort_values("tp_int")
        if pat.empty:
            log.error(f"Pas de volumes pour {args.patient_id}"); sys.exit(1)
        rad_t1c = rad[
            (rad["Mask"] == "mask_tumor") & (rad["Sequence"] == "t1c") &
            (rad["Timepoint_Label"] == "baseline")
        ]
        rad_row = {}
        if not rad_t1c.empty:
            row = rad_t1c[rad_t1c["Patient_ID"] == args.patient_id]
            if row.empty:
                row = rad_t1c.iloc[[0]]
            rad_row = row.iloc[0].to_dict()
        ode_out = output_dir / "ode_curves" / args.patient_id
        ode_out.mkdir(parents=True, exist_ok=True)
        t_end = max(args.simulate_days, float(pat["time_days"].max()))
        plot_patient_ode_from_volumes(
            args.patient_id,
            pat["Volume_mm3"].values, pat["time_days"].values,
            rad_row, ode_out, simulate_days=t_end, n_ode_points=args.ode_points,
        )
        print(f"\n  Courbe ODE -> {ode_out.resolve()}/ode_volume.png\n")
        return

    if args.predict:
        if not args.patient_id:
            log.error("--patient_id requis avec --predict"); sys.exit(1)
        model_pkl = output_dir / "pde_ml_model.pkl"
        if not model_pkl.exists():
            log.error(f"Modèle introuvable : {model_pkl}"); sys.exit(1)
        # Patients à prédire : directement dans prediction/<PID>/
        patient_dir = script_dir / "prediction" / args.patient_id
        if not patient_dir.exists():
            log.error(f"Dossier patient introuvable : {patient_dir}"); sys.exit(1)
        # Sorties de prédiction : prediction/<PID>/prediction/
        pred_out = patient_dir / "prediction"
        pred_out.mkdir(parents=True, exist_ok=True)
        sim = predict_new_patient(
            model_pkl, patient_dir,
            args.patient_id, args.simulate_days, pred_out,
            dt_days=args.dt_days,
            bin_threshold=args.bin_threshold,
            n_snapshots=args.n_snapshots,
            use_tissue=not args.no_tissue,
            d_wm_gm_ratio=args.d_wm_gm_ratio,
            r_wm_gm_ratio=args.r_wm_gm_ratio,
            alpha_wm_gm_ratio=args.alpha_wm_gm_ratio,
            csf_factor=args.csf_factor,
        )
        generate_plots(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                       sim_result=sim, output_dir=pred_out)
        rel = Path("prediction") / args.patient_id / "prediction"
        print(f"\n  Les sorties de prédiction sont dans le dossier : {rel.as_posix()}/")
        print(f"  (chemin complet : {pred_out.resolve()}/)\n")
        log.info("Prédictions écrites dans %s", pred_out.resolve())
        return

    augmented_dir = script_dir / args.augmented_dir if args.use_augmented else None
    if args.train_augmented_only and not args.use_augmented:
        log.error("--train_augmented_only requiert --use_augmented"); sys.exit(1)

    if args.train_augmented_only:
        log.info("Mode rapide : entrainement augmente depuis CSV existants")
        gt_path = output_dir / "ground_truth_parameters.csv"
        if not gt_path.exists():
            log.error("Manque ground_truth_parameters.csv — "
                      "lancez d'abord python random_forest_pipeline.py"); sys.exit(1)
        gt_df = pd.read_csv(gt_path)
        X_aug, _ = load_augmented_training_tables(augmented_dir)
        feat_cols = [
            c for c in X_aug.columns
            if c not in ("Patient_ID", "is_synthetic", "source_id")
        ]
        X_df = X_aug[X_aug["is_synthetic"].astype(int) == 0][["Patient_ID"] + feat_cols]
    else:
        log.info("ÉTAPE 1/5 — Chargement des données tabulaires")
        rad, clin, nifti_df = load_tabular_data(script_dir)

        log.info("ÉTAPE 2/5 — Volumes longitudinaux (NIfTI réels + CSV fallback)")
        vol_df = build_longitudinal_volumes(data_dir, nifti_df, dt_days=args.dt_days)
        output_dir.mkdir(parents=True, exist_ok=True)
        vol_df.to_csv(output_dir / "longitudinal_volumes.csv", index=False)

        log.info("ÉTAPE 3/5 — Features NIfTI baseline")
        nifti_feats = extract_all_nifti_features(data_dir)
        if not nifti_feats.empty:
            nifti_feats.to_csv(output_dir / "nifti_baseline_features.csv", index=False)

        log.info("ÉTAPE 4/5 — Vérité terrain (D, r, α) par patient")
        gt_df = build_ground_truth(vol_df, clin, rad)
        gt_df.to_csv(output_dir / "ground_truth_parameters.csv", index=False)

        log.info("ÉTAPE 5/5 — Matrice X + entraînement Random Forest")
        X_df, feat_cols = build_feature_matrix(gt_df, rad, clin, nifti_feats)
        X_df.to_csv(output_dir / "feature_matrix.csv", index=False)

    if args.use_augmented:
        load_augmented_training_tables(augmented_dir)  # verifie que les CSV existent

    model, scaler, metrics_df, preds_df = train_and_evaluate(
        X_df, gt_df, feat_cols,
        n_estimators    = args.n_estimators,
        max_depth       = args.max_depth,
        min_samples_leaf= args.min_samples_leaf,
        test_size       = args.test_size,
        n_cv_folds      = args.n_cv_folds,
        output_dir      = output_dir,
        augmented_dir   = augmented_dir,
    )

    if model is not None:
        generate_plots(gt_df, preds_df, metrics_df, output_dir=output_dir)

    print(f"\n  Fichiers générés dans {output_dir.resolve()}/")
    for f in sorted(output_dir.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
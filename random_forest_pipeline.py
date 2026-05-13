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

  # Prédiction nouveau patient
  python tumor_pde_ml_pipeline.py --predict --patient_id P04a --simulate_days 30
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
C0          = 1.0                 # concentration normalisée à t=0

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
    Cherche radiomics / clinical / nifti_features dans script_dir
    (priorité .xlsx puis .csv).
    """
    def _find(base):
        for ext in (".xlsx", ".csv", ".xls"):
            p = script_dir / (base + ext)
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Fichier {base}.xlsx / .csv introuvable dans {script_dir}"
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
        return float((data > 0).sum() * np.prod(img.header.get_zooms()))
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
    # ── 1. Dataset valide ─────────────────────────────────────────────────────
    n_gt_rows     = len(gt_df)
    n_success_gt  = int(gt_df["success"].sum())
    valid         = gt_df[gt_df["success"]][["Patient_ID"] + TARGET_PARAMS].dropna()
    n_valid_ml    = len(valid)
    X_sub         = X_df[X_df["Patient_ID"].isin(valid["Patient_ID"])]
    X_sub         = X_sub.set_index("Patient_ID").reindex(valid["Patient_ID"])
    X_arr         = X_sub[feature_cols].values
    Y_arr         = valid[TARGET_PARAMS].values

    log.info(
        f"Effectifs GT → total lignes : {n_gt_rows} | success=True : {n_success_gt} | "
        f"ML (success + D,r,α sans NaN) : {n_valid_ml}"
    )
    log.info(f"Dataset ML : {X_arr.shape[0]} patients × {X_arr.shape[1]} features → "
             f"3 cibles (D, r, α)")

    if X_arr.shape[0] < 6:
        log.error(
            "< 6 patients valides — entraînement impossible.\n"
            "Conseils : réduire --dt_days, vérifier les masques de segmentation, "
            "ou abaisser le seuil R² dans estimate_patient_parameters()."
        )
        return None, None, None, None

    # ── 2. Normalisation ──────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    # ── 3. Split train / test ──────────────────────────────────────────────────
    X_tr, X_te, Y_tr, Y_te, pid_tr, pid_te = train_test_split(
        X_scaled, Y_arr, valid["Patient_ID"].values,
        test_size=test_size, random_state=42,
    )
    n_train, n_test = X_tr.shape[0], X_te.shape[0]
    log.info(f"Split train/test : {n_train} train | {n_test} test (fraction test={test_size})")

    # ── 3b. CV R² sur tout le jeu valide (avant split — plus stable qu'un seul test)
    rf_template = build_rf_model(n_estimators, max_depth, min_samples_leaf)
    n_splits_full = min(max(2, n_cv_folds), X_arr.shape[0])
    kfold_full    = KFold(n_splits=n_splits_full, shuffle=True, random_state=43)
    cv_full_results = {}
    print(f"\n  Cross-validation {n_splits_full}-fold (tout le jeu valide, {X_arr.shape[0]} patients) :")
    for i, param in enumerate(TARGET_PARAMS):
        scores_f = cross_val_score(
            clone(rf_template.estimator),
            X_scaled, Y_arr[:, i],
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
    joblib.dump(
        {"model": model, "scaler": scaler, "feature_cols": feature_cols,
         "model_type": "random_forest",
         "hyperparams": {"n_estimators": n_estimators,
                         "max_depth": max_depth,
                         "min_samples_leaf": min_samples_leaf}},
        output_dir / "pde_ml_model.pkl",
    )
    metrics_df.to_csv(output_dir / "metrics.csv",            index=False)
    preds_df.to_csv(output_dir / "predictions_test.csv",     index=False)
    log.info(f"Modèle Random Forest → {output_dir / 'pde_ml_model.pkl'}")

    return model, scaler, metrics_df, preds_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — SIMULATION EDP 3D
# ─────────────────────────────────────────────────────────────────────────────

def simulate_pde_3d(D: float, r: float, alpha: float,
                   u0_3d: np.ndarray, dx_mm: float,
                   t_days: float, c0: float = C0,
                   lam: float = LAMBDA_DRUG,
                   t_treat: float = 0.0):
    """
    Version corrigée : 
    1. Utilise la somme intégrale de 'u' pour des courbes fluides (plus de plateau à zéro).
    2. Simule un protocole réel de 6 cycles (une dose tous les 28 jours).
    """
    dt_cfl  = (dx_mm ** 2) / (6.0 * D + 1e-12)
    dt      = min(1.0, dt_cfl * 0.9)
    n_steps = max(1, int(np.ceil(t_days / dt)))
    dt      = t_days / n_steps

    u       = u0_3d.astype(np.float64).copy()
    vox_vol = dx_mm ** 3
    
    # CORRECTION 1 : On utilise np.sum(u) dès le départ pour la fluidité
    volumes = [float(np.sum(u) * vox_vol)]

    log.info(f"Simulation EDP 3D | Protocole Multi-cycles | D={D:.4f}  r={r:.5f}  α={alpha:.4f}")

    for step in range(n_steps):
        t_now = step * dt
        
        # CORRECTION 2 : Modélisation d'un protocole de 6 cycles (Témozolomide-like)
        c_t = 0.0
        if t_now >= t_treat:
            # On simule 6 cycles de traitement espacés de 28 jours
            for i in range(6):
                t_cycle_start = t_treat + (i * 28)
                if t_now >= t_cycle_start:
                    # Chaque cycle ajoute sa propre concentration qui décroît
                    c_t += c0 * np.exp(-lam * (t_now - t_cycle_start))

        # Calcul du Laplacien (Diffusion)
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
        
        # Mise à jour de la densité tumorale
        u = np.clip(u + dt*(D*lap + r*u*(1.0-u) - alpha*c_t*u), 0.0, 1.0)
        
        # CORRECTION 1 (suite) : Somme intégrale pour suivre l'évolution sans seuil brutal
        volumes.append(float(np.sum(u) * vox_vol))

    return u, volumes


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PRÉDICTION + SIMULATION NOUVEAU PATIENT
# ─────────────────────────────────────────────────────────────────────────────

def predict_new_patient(model_pkl: Path, patient_dir: Path, patient_id: str,
                         simulate_days: float = 30.0,
                         output_dir: Path = Path("results_rfml")) -> dict:
    """
    1. Extrait les features NIfTI depuis les fichiers réels du patient
    2. Prédit D, r, α avec le Random Forest sauvegardé
    3. Simule l'évolution tumorale à T+simulate_days jours
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

    # Condition initiale depuis masque baseline
    seg_array, dx_mm = None, 1.0
    if paths["seg"]["baseline"].exists():
        try:
            import nibabel as nib
            seg_img   = nib.load(str(paths["seg"]["baseline"]))
            dx_mm     = float(seg_img.header.get_zooms()[0])
            seg_array = np.clip(
                (seg_img.get_fdata() > 0).astype(float) * 0.8, 0.0, 1.0
            )
        except Exception as e:
            log.warning(f"Segmentation non chargeable : {e}")

    if seg_array is None:
        g = 40; c = g // 2
        zz, yy, xx = np.ogrid[:g, :g, :g]
        seg_array = np.where(
            np.sqrt((xx-c)**2 + (yy-c)**2 + (zz-c)**2) < 10/dx_mm,
            0.8, 0.0
        )
        log.info("Sphère synthétique utilisée (masque absent)")

    u_final, volumes = simulate_pde_3d(
        D_pred, r_pred, alpha_pred, seg_array, dx_mm, simulate_days,
        t_treat=t_treat_pred,
    )
    V_i = float((seg_array > 0.5).sum() * dx_mm**3)
    V_f = float((u_final > 0.5).sum() * dx_mm**3)
    s   = "+" if V_f >= V_i else ""
    print(f"\n  ── Simulation T+{simulate_days:.0f}j ────────────────────────────────────")
    print(f"  Volume initial  : {V_i:>10.1f} mm³")
    print(f"  Volume final    : {V_f:>10.1f} mm³")
    print(f"  Variation       : {s}{V_f-V_i:.1f} mm³ ({s}{100*(V_f-V_i)/(V_i+1e-3):.1f}%)")

    # Sauvegarde
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import nibabel as nib
        nib.save(
            nib.Nifti1Image(u_final.astype(np.float32),
                            np.diag([dx_mm, dx_mm, dx_mm, 1.0])),
            str(output_dir / f"{patient_id}_u_T{int(simulate_days)}j.nii.gz"),
        )
    except Exception:
        pass
    pd.DataFrame({
        "step": range(len(volumes)),
        "days": np.linspace(0, simulate_days, len(volumes)),
        "volume_mm3": volumes,
    }).to_csv(output_dir / f"{patient_id}_volumes_sim.csv", index=False)

    return dict(
        D_pred=D_pred, r_pred=r_pred, alpha_pred=alpha_pred,
        t_treat_pred=t_treat_pred,
        v_front=2*np.sqrt(r_pred*D_pred),
        lambda_front=np.sqrt(D_pred/r_pred),
        u_initial=seg_array, u_final=u_final,
        volumes_evolution=volumes,
        V_initial_mm3=V_i, V_final_mm3=V_f,
        dx_mm=dx_mm, simulate_days=simulate_days,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

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
        ax.plot(days, vols, color="#E63946", lw=2.5, label="Volume simulé (EDP)")
        ax.axhline(vols[0], color="gray", ls=":", lw=1.5, label="Volume initial")
        ax.fill_between(days, vols[0], vols, alpha=0.1, color="#E63946")
        ax.set(xlabel="Temps (jours)", ylabel="Volume tumoral (mm³)",
               title=(f"Simulation EDP 3D — T+{sim_result['simulate_days']:.0f}j\n"
                      f"D={sim_result['D_pred']:.4f} mm²/j  "
                      f"r={sim_result['r_pred']:.5f} /j  "
                      f"α={sim_result['alpha_pred']:.4f} /j"))
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

    if args.predict:
        if not args.patient_id:
            log.error("--patient_id requis avec --predict"); sys.exit(1)
        model_pkl = output_dir / "pde_ml_model.pkl"
        if not model_pkl.exists():
            log.error(f"Modèle introuvable : {model_pkl}"); sys.exit(1)
        sim = predict_new_patient(
            model_pkl, data_dir / args.patient_id,
            args.patient_id, args.simulate_days, output_dir,
        )
        generate_plots(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                       sim_result=sim, output_dir=output_dir)
        return

    # ── Pipeline d'entraînement ───────────────────────────────────────────────
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

    model, scaler, metrics_df, preds_df = train_and_evaluate(
        X_df, gt_df, feat_cols,
        n_estimators    = args.n_estimators,
        max_depth       = args.max_depth,
        min_samples_leaf= args.min_samples_leaf,
        test_size       = args.test_size,
        n_cv_folds      = args.n_cv_folds,
        output_dir      = output_dir,
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
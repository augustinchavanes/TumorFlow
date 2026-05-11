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

Structure du dataset attendu (NIfTI)
──────────────────────────────────────
  data/
  ├── 17253793/                   ← identifiant étude (``--study_subdir``)
  │   └── P04a/
  │       └── BraTS/              ← ``--brats_subdir`` : NIfTI BraTS
  │           └── …
  ├── radiomics_features.xlsx     ← ou .csv
  ├── clinical_features.xlsx
  └── nifti_features.xlsx         ← feuille type ALL_PATIENTS (long format)

  Si ``data/17253793`` n'existe pas, le script parcourt ``data/PXX`` directement.

Usage rapide :
  python tumor_pde_ml_pipeline.py --data_dir ./data --study_subdir 17253793 \\
      --model_type mlp --dt_days 56 --output ./results

  # Prédiction pour un nouveau patient :
  python tumor_pde_ml_pipeline.py --predict --patient_dir ./data/NewPatient \
      --model_dir ./results --simulate_days 30
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import argparse
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
from scipy.ndimage import label, binary_erosion
from scipy.stats import pearsonr

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CHARGEMENT DES DONNÉES TABULAIRES
# ──────────────────────────────────────────────────────────────────────────────

def _read_tabular_file(path: str | Path, label: str = "") -> pd.DataFrame:
    """
    Lit un CSV ou un Excel (.xlsx / .xls). Pour Excel, essaie plusieurs feuilles.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label or 'Fichier'} introuvable : {path.resolve()}")

    # Certains fichiers sont "mal renommés" (ex: xlsx avec extension .csv).
    # Un .xlsx est un ZIP → header commence par "PK".
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
    except Exception:
        magic = b""
    looks_like_excel_zip = magic.startswith(b"PK")

    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls") or looks_like_excel_zip:
        for sheet in ("ALL_PATIENTS", "Sheet1", "Feuil1", 0):
            try:
                df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
                if df is not None and len(df) > 0:
                    return df
            except Exception:
                continue
        return pd.read_excel(path, sheet_name=0, engine="openpyxl")
    # CSV : encodages + séparateurs (exports Excel) → fallback robuste.
    # On tente d'abord : sep=None (détection) puis des séparateurs courants.
    last_exc: Exception | None = None
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin1")
    seps = (None, ",", ";", "\t", "|")

    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(
                    path,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                )
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                last_exc = exc
                continue

    # Dernier recours : remplacement des caractères non décodables.
    if last_exc is not None:
        return pd.read_csv(
            path,
            encoding="cp1252",
            sep=None,
            engine="python",
            encoding_errors="replace",
        )
    return pd.read_csv(
        path,
        encoding="cp1252",
        sep=None,
        engine="python",
        encoding_errors="replace",
    )


def load_tabular_data(radiomics_path, clinical_path, nifti_path):
    """
    Charge et nettoie les trois sources tabulaires (``.csv`` ou ``.xlsx``).

    Returns
    -------
    rad      : pd.DataFrame  — radiomics par patient/masque/séquence/timepoint
    clin     : pd.DataFrame  — données cliniques (une ligne par patient)
    nifti_df : pd.DataFrame  — volumes et dynamiques NIfTI (long format)
    """
    # ── Radiomics ────────────────────────────────────────────────────────────
    rad = _read_tabular_file(radiomics_path, "Radiomics")
    rad.columns = rad.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    # ── Clinical ─────────────────────────────────────────────────────────────
    clin = _read_tabular_file(clinical_path, "Clinical")
    clin.columns = clin.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    zenodo_clin = {
        "Patient ID (Zenodo)": "Patient_ID",
        "Tumour Histology": "Histology",
        "Prior Chemo": "Prior_Chemo",
        "Karnofsky PS (%)": "KPS",
        "Age (yrs)": "Age",
        "Number of metastases": "N_metastases",
        "Rx dose at tumor margins": "Dose_raw",
        "Fractions": "Fractions",
        "WHO PS": "WHO_PS",
    }
    clin = clin.rename(columns={k: v for k, v in zenodo_clin.items() if k in clin.columns})
    if "Patient_ID" not in clin.columns and "Patient ID" in clin.columns:
        clin = clin.rename(columns={"Patient ID": "Patient_ID"})
    if "Patient_ID" not in clin.columns:
        raise ValueError(
            "clinical : colonne patient introuvable "
            "(Patient_ID, « Patient ID (Zenodo) » ou « Patient ID »)."
        )
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
                pc.astype(str).str.strip().str.lower().isin(("yes", "y", "1", "true", "oui"))
            ).astype(int)
    else:
        clin["Prior_Chemo"] = 0
    if "Histology" in clin.columns:
        clin["Histology"] = clin["Histology"].astype(str).str.strip()
    else:
        clin["Histology"] = "unknown"

    hist_map = {"NSCLC": 0, "Breast Ca": 1, "SCLC": 2}
    clin["Histology_enc"] = clin["Histology"].map(hist_map).fillna(3).astype(int)

    # ── NIfTI features (table longue : Patient_ID, NiftiFeature, NiftiValue) ─
    nifti_df = _read_tabular_file(nifti_path, "NIfTI features")
    nifti_df.columns = nifti_df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    for a, b in (("NiftiFeature", "NiftiFeature"), ("Feature", "NiftiFeature")):
        if b not in nifti_df.columns and a in nifti_df.columns:
            nifti_df = nifti_df.rename(columns={a: b})
    for a, b in (("NiftiValue", "NiftiValue"), ("Value", "NiftiValue")):
        if b not in nifti_df.columns and a in nifti_df.columns:
            nifti_df = nifti_df.rename(columns={a: b})

    log.info(f"Radiomics : {rad.shape} | Clinique : {clin.shape} | NIfTI feat : {nifti_df.shape}")
    return rad, clin, nifti_df


def resolve_tabular_path(primary: str, *alternatives: str) -> str:
    """Retourne le premier chemin existant parmi primary et alternatives."""
    for p in (primary,) + alternatives:
        if p and Path(p).exists():
            return p
    return primary


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — EXTRACTION DES VOLUMES LONGITUDINAUX
# ──────────────────────────────────────────────────────────────────────────────

def extract_longitudinal_volumes(nifti_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait les volumes tumoraux T1CE par patient et timepoint.

    Returns
    -------
    DataFrame avec colonnes :
        Patient_ID | timepoint (int) | Volume_mm3 | GrowthRatio | DeltaVolume_mm3
    """
    need = {"Patient_ID", "NiftiFeature", "NiftiValue"}
    miss = need - set(nifti_df.columns)
    if miss:
        raise ValueError(
            f"nifti features : colonnes manquantes {miss}. "
            f"Colonnes trouvées : {list(nifti_df.columns)[:25]}…"
        )
    tv = nifti_df[nifti_df["NiftiFeature"].str.contains("TumorVolume_mm3", na=False)].copy()

    def _parse_tp(feat_name):
        if "baseline" in feat_name:
            return 0
        for i in range(1, 6):
            if f"follow_up_{i}" in feat_name:
                return i
        return None

    tv["timepoint"] = tv["NiftiFeature"].apply(_parse_tp)
    tv = tv.dropna(subset=["timepoint"])
    tv["timepoint"] = tv["timepoint"].astype(int)
    tv = tv.rename(columns={"NiftiValue": "Volume_mm3"})
    tv = tv[["Patient_ID", "timepoint", "Volume_mm3"]].sort_values(["Patient_ID", "timepoint"])

    # Ajouter GrowthRatio inter-timepoints
    growth = nifti_df[nifti_df["NiftiFeature"].str.contains("VolumeGrowthRatio", na=False)].copy()
    growth["timepoint"] = growth["NiftiFeature"].apply(_parse_tp)
    growth = growth.dropna(subset=["timepoint"])
    growth["timepoint"] = growth["timepoint"].astype(int)
    growth = growth.rename(columns={"NiftiValue": "GrowthRatio"})[["Patient_ID", "timepoint", "GrowthRatio"]]

    tv = tv.merge(growth, on=["Patient_ID", "timepoint"], how="left")

    # Doublons Excel / plusieurs features → une mesure par (patient, timepoint)
    agg = {"Volume_mm3": "mean", "GrowthRatio": "mean"}
    agg = {k: v for k, v in agg.items() if k in tv.columns}
    tv = tv.groupby(["Patient_ID", "timepoint"], as_index=False).agg(agg)
    tv = tv.sort_values(["Patient_ID", "timepoint"])

    log.info(f"Volumes extraits : {tv['Patient_ID'].nunique()} patients, "
             f"{len(tv)} mesures (timepoints 0–{tv['timepoint'].max()})")
    return tv


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — EXTRACTION DES FEATURES NIfTI DEPUIS LES MASQUES DE SEGMENTATION
# ──────────────────────────────────────────────────────────────────────────────

def extract_nifti_features_from_seg(seg_path: str, t1ce_path: str = None,
                                     flair_path: str = None) -> dict:
    """
    Calcule des features géométriques et d'intensité directement depuis
    les fichiers NIfTI (seg + t1ce + flair optionnels).

    Nécessite nibabel. Retourne un dict de scalaires.
    """
    try:
        import nibabel as nib
    except ImportError:
        log.warning("nibabel non installé — features NIfTI non calculées. "
                    "Installez avec : pip install nibabel")
        return {}

    results = {}

    # ── Segmentation ─────────────────────────────────────────────────────────
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    voxel_size = np.prod(seg_img.header.get_zooms())  # mm³/voxel

    # Volume tumoral (label > 0 = tumeur active T1CE)
    tumor_mask = (seg_data > 0).astype(float)
    volume_mm3 = tumor_mask.sum() * voxel_size
    results["seg_TumorVolume_mm3"] = volume_mm3

    # Sphéricité via l'approximation sphère équivalente
    if volume_mm3 > 0:
        r_sphere = ((3 * volume_mm3) / (4 * np.pi)) ** (1 / 3)
        sa_sphere = 4 * np.pi * r_sphere ** 2
        # Surface approximée par comptage de voxels de bord
        eroded = binary_erosion(tumor_mask)
        border = tumor_mask - eroded
        sa_voxel = border.sum() * (voxel_size ** (2 / 3))
        results["seg_Sphericity"] = (np.pi ** (1 / 3) * (6 * volume_mm3) ** (2 / 3)) / sa_voxel if sa_voxel > 0 else 0

        # Diamètre maximal (approximation bounding box)
        coords = np.argwhere(tumor_mask > 0)
        zooms = seg_img.header.get_zooms()
        bbox_mm = (coords.max(axis=0) - coords.min(axis=0)) * zooms
        results["seg_MaxDiameter_mm"] = float(np.linalg.norm(bbox_mm))
        results["seg_Elongation"] = float(bbox_mm.min() / (bbox_mm.max() + 1e-6))
    else:
        results.update({"seg_Sphericity": 0, "seg_MaxDiameter_mm": 0, "seg_Elongation": 0})

    # ── Intensités T1CE dans le masque ───────────────────────────────────────
    if t1ce_path and Path(t1ce_path).exists():
        t1ce_data = nib.load(t1ce_path).get_fdata()
        if tumor_mask.shape == t1ce_data.shape:
            vals = t1ce_data[tumor_mask > 0]
            results["seg_T1CE_Mean"]   = float(vals.mean()) if len(vals) > 0 else 0
            results["seg_T1CE_Std"]    = float(vals.std())  if len(vals) > 0 else 0
            results["seg_T1CE_Max"]    = float(vals.max())  if len(vals) > 0 else 0

    # ── Intensités FLAIR dans le masque ──────────────────────────────────────
    if flair_path and Path(flair_path).exists():
        flair_data = nib.load(flair_path).get_fdata()
        if tumor_mask.shape == flair_data.shape:
            vals = flair_data[tumor_mask > 0]
            results["seg_FLAIR_Mean"]  = float(vals.mean()) if len(vals) > 0 else 0
            results["seg_FLAIR_Std"]   = float(vals.std())  if len(vals) > 0 else 0

    return results


def _tumor_mask_from_t1c_proxy(t1c: np.ndarray) -> np.ndarray:
    """
    Masque tumoral approximatif sans segmentation : sur-rehaussement T1c
    (percentiles + plus grande composante connexe).
    """
    data = np.asarray(t1c, dtype=float)
    pos = data[data > 0]
    if pos.size < 100:
        return np.zeros_like(data, dtype=float)
    lo, hi = np.percentile(pos, [2.0, 99.8])
    brain = (data > lo) & (data < hi * 1.02)
    if brain.sum() < 500:
        brain = data > np.percentile(pos, 5.0)
    vals = data[brain]
    if vals.size < 50:
        return np.zeros_like(data, dtype=float)
    thr = max(
        float(np.percentile(vals, 97.5)),
        float(np.median(vals) + 2.0 * (np.std(vals) + 1e-6)),
    )
    m = ((data >= thr) & brain).astype(float)
    lab, n = label(m > 0)
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = int(np.argmax(sizes))
    return (lab == main).astype(float)


def extract_nifti_features_from_t1c_proxy(
    t1c_path: str,
    flair_path: str | None = None,
) -> dict:
    """
    Même sortie clés que ``extract_nifti_features_from_seg`` sans fichier seg :
    masque dérivé du T1c (baseline).
    """
    try:
        import nibabel as nib
    except ImportError:
        log.warning("nibabel non installé — features NIfTI non calculées.")
        return {}

    t1c_img = nib.load(t1c_path)
    t1ce_data = np.asarray(t1c_img.get_fdata(), dtype=float)
    zooms = t1c_img.header.get_zooms()
    voxel_size = float(np.prod(zooms))

    tumor_mask = _tumor_mask_from_t1c_proxy(t1ce_data)
    results: dict = {}

    volume_mm3 = float(tumor_mask.sum() * voxel_size)
    results["seg_TumorVolume_mm3"] = volume_mm3

    if volume_mm3 > 1e-6:
        eroded = binary_erosion(tumor_mask)
        border = tumor_mask - eroded
        sa_voxel = border.sum() * (voxel_size ** (2 / 3))
        results["seg_Sphericity"] = (
            (np.pi ** (1 / 3) * (6 * volume_mm3) ** (2 / 3)) / sa_voxel if sa_voxel > 0 else 0.0
        )
        coords = np.argwhere(tumor_mask > 0)
        bbox_mm = (coords.max(axis=0) - coords.min(axis=0)) * np.asarray(zooms[:3])
        results["seg_MaxDiameter_mm"] = float(np.linalg.norm(bbox_mm))
        results["seg_Elongation"] = float(bbox_mm.min() / (bbox_mm.max() + 1e-6))
    else:
        results.update({"seg_Sphericity": 0.0, "seg_MaxDiameter_mm": 0.0, "seg_Elongation": 0.0})

    vals = t1ce_data[tumor_mask > 0]
    results["seg_T1CE_Mean"] = float(vals.mean()) if vals.size > 0 else 0.0
    results["seg_T1CE_Std"] = float(vals.std()) if vals.size > 0 else 0.0
    results["seg_T1CE_Max"] = float(vals.max()) if vals.size > 0 else 0.0

    if flair_path and Path(flair_path).exists():
        flair_data = nib.load(flair_path).get_fdata()
        if tumor_mask.shape == flair_data.shape:
            fv = flair_data[tumor_mask > 0]
            results["seg_FLAIR_Mean"] = float(fv.mean()) if fv.size > 0 else 0.0
            results["seg_FLAIR_Std"] = float(fv.std()) if fv.size > 0 else 0.0
    else:
        results["seg_FLAIR_Mean"] = 0.0
        results["seg_FLAIR_Std"] = 0.0

    return results


def _collect_nifti_files(scan_root: Path) -> list[Path]:
    """Tous les volumes NIfTI sous ``scan_root`` (``.nii`` / ``.nii.gz``, casse quelconque)."""
    found: list[Path] = []
    seen: set[Path] = set()
    for p in scan_root.rglob("*"):
        if not p.is_file():
            continue
        suf = "".join(p.suffixes).lower()
        if not (suf.endswith(".nii.gz") or suf.endswith(".nii")):
            continue
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            found.append(p)
    return found


def _nii_blob(path: Path, scan_root: Path) -> str:
    """Chemin relatif + nom (minuscules) pour repérer ``…/t1c/image.nii.gz`` ou ``…-t1c.nii.gz``."""
    try:
        rel = path.relative_to(scan_root).as_posix().lower()
    except ValueError:
        rel = ""
    return f"{rel}//{path.name.lower()}"


def _parent_is_modality_only(parent: str) -> bool:
    """Dossier parent typé modalité (volume brut, pas la seg)."""
    p = parent.lower()
    return p in (
        "t1c", "t1ce", "t1n", "t2f", "t2flair", "flair", "t2w", "t1", "t2",
        "adc", "dwi", "pd", "pet", "mtv", "dce", "pwi", "spect",
    )


def _parent_is_seg_like(parent: str) -> bool:
    return parent.lower() in (
        "segmentation", "seg", "labels", "label", "mask", "tumor",
        "tumor_segmentation", "expert_seg", "manual", "gt",
        "ground_truth", "annot", "annotation",
    )


def _is_t1_weighted_gd(blob: str) -> bool:
    """T1 post-gadolinium (BraTS ``t1c`` / ``t1ce``)."""
    if any(x in blob for x in (
        "t1ce", "t1_ce", "t1-gd", "t1gd", "t1-gado", "t1_gd",
    )):
        return True
    nm = blob.split("//", 1)[-1]
    if "-t1c." in nm or "_t1c." in nm or nm.startswith("t1c."):
        return "t1n" not in nm.split(".")[0]
    if "/t1c/" in blob.replace("\\", "/") or "\\t1c\\" in blob.lower():
        return True
    return False


def _is_flair(blob: str) -> bool:
    if any(x in blob for x in (
        "flair", "t2_flair", "t2-flair", "t2f", "t2_f", "t2-f", "space-flair",
    )):
        return True
    nm = blob.split("//", 1)[-1]
    if "-t2f." in nm or "_t2f." in nm or nm.startswith("t2f."):
        return True
    # Jeu 17253793 : ``baseline/fla.nii.gz``
    b = blob.replace("\\", "/").lower()
    return nm.startswith("fla.") or "/fla/" in b


def _is_seg_volume(path: Path, blob: str) -> bool:
    """Masque / labels tumoraux (exclut les ``image.nii.gz`` dans ``t1c/`` seulement)."""
    parent = path.parent.name
    name_l = path.name.lower()

    if _parent_is_seg_like(parent) and (name_l.endswith(".nii.gz") or name_l.endswith(".nii")):
        return True

    if _parent_is_modality_only(parent) and name_l in (
        "image.nii.gz", "volume.nii.gz", "data.nii.gz",
        "image.nii", "volume.nii", "data.nii",
    ):
        return False

    if any(x in blob for x in (
        "tumor_mask", "tumor_seg", "tumor_vol", "segmentation", "segmentación",
        "-seg.", "_seg.", ".seg.", "/seg/", "\\seg\\",
        "labels", "label_map", "ground_truth", "gtseg", "truth", "gts",
        "gtv", "ctv", "ptv", "_roi", "-roi.", "roi_mask",
    )):
        return True
    if "label" in name_l and "t1" not in name_l and "t2" not in name_l and "flair" not in name_l:
        return True
    if "seg" in name_l or "seg" in blob.split("//", 1)[0]:
        return True
    if "mask" in name_l:
        return True
    return False


def _seg_sort_key(path: Path, scan_root: Path) -> tuple:
    """Préfère ``-seg`` / tumeur avant un simple ``brain_mask``."""
    blob = _nii_blob(path, scan_root)
    pri = 0
    if "-seg." in blob or "_seg." in blob or "/seg/" in blob.replace("\\", "/").lower():
        pri = 4
    elif "tumor" in blob or "segmentation" in blob:
        pri = 3
    elif "label" in blob:
        pri = 2
    elif "mask" in blob:
        pri = 1
    return (-pri, str(path))


def _folder_leaf_from_nii(path: Path, scan_root: Path) -> str:
    """Ex. ``baseline/t1c.nii.gz`` → ``baseline`` ; ``fu2/t1c.nii.gz`` → ``fu2``."""
    try:
        rel = path.relative_to(scan_root).as_posix()
    except ValueError:
        return ""
    parts = Path(rel).parts
    return parts[0].lower() if len(parts) >= 2 else ""


def _time_rank_from_blob(blob: str) -> tuple:
    """
    Ordonne les timepoints : ``baseline`` puis ``fu1``, ``fu2``, …
    détectés dans le nom/chemin (`baseline`, `fu1`, `fu2`, …).
    """
    import re
    b = blob.lower()
    if "baseline" in b:
        return (0, 0)
    m = re.search(r"\bf(u)?\s*[_-]?\s*(\d+)", b)
    # Ex: "fu1", "fu_2", "fu2.nii.gz"
    if m:
        try:
            return (1, int(m.group(2)))
        except ValueError:
            pass
    m2 = re.search(r"\bfu\s*(\d+)", b)
    if m2:
        try:
            return (1, int(m2.group(1)))
        except ValueError:
            pass
    return (3, 0)


def _iter_patient_brats_roots(
    data_dir: Path,
    study_subdir: str | None,
    brats_subdir: str,
) -> list[tuple[str, Path]]:
    """
    Retourne [(Patient_ID, dossier_à_scanner), ...].

    Layout attendu (ex.) : data_dir / 17253793 / P04a / BraTS / *.nii.gz
    Si ``data_dir / study_subdir`` n'existe pas, on suppose que ``data_dir``
    est déjà la racine étude (ex. .../17253793) avec PXX/BraTS en dessous.
    """
    roots: list[tuple[str, Path]] = []
    if study_subdir:
        study = data_dir / study_subdir
        if study.is_dir():
            for p in sorted(study.iterdir()):
                if not p.is_dir() or p.name.startswith("."):
                    continue
                brats = p / brats_subdir
                # On parcourt le dossier patient complet pour inclure
                # `tumor_segmentation/` qui est souvent un sibling de `BraTS/`.
                roots.append((p.name, p))
            return roots
        log.info(
            f"Dossier étude {study} introuvable — utilisation de {data_dir} "
            "comme racine (PXX/BraTS ou PXX seul)."
        )
    for p in sorted(data_dir.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        brats = p / brats_subdir
        roots.append((p.name, p))
    return roots


def scan_patient_folders(
    data_dir: str,
    study_subdir: str | None = "17253793",
    brats_subdir: str = "BraTS",
) -> pd.DataFrame:
    """
    Parcourt les dossiers patients et extrait les features depuis les NIfTI.

    Layout principal (BraTS) ::

        data_dir /
            17253793 /          ← ``study_subdir`` (identifiant étude / export)
                P04a /
                    BraTS /    ← ``brats_subdir`` : segmentations + modalités
                        *.nii.gz
                    Dicom /
                    ...

    Si ``study_subdir`` est absent sur le disque, chaque sous-dossier direct
    de ``data_dir`` est traité comme ``PXX`` (même logique qu'avant).

    Fichiers reconnus : noms type BraTS (``*-t1c.nii.gz``, ``*-t2f.nii.gz``, ``*-seg.nii.gz``),
    chemins du type ``…/t1c/image.nii.gz``, ``…/Segmentation/…``, etc.
    """

    data_dir = Path(data_dir)
    records = []

    if not data_dir.exists():
        log.warning(f"Le dossier {data_dir} n'existe pas.")
        return pd.DataFrame()

    patient_roots = _iter_patient_brats_roots(data_dir, study_subdir, brats_subdir)

    for pid, scan_root in patient_roots:

        # ─────────────────────────────────────────────
        # Recherche récursive des NIfTI sous BraTS (ou dossier patient)
        # ─────────────────────────────────────────────
        all_nii = _collect_nifti_files(scan_root)

        if len(all_nii) == 0:
            log.warning(f"Aucun fichier NIfTI trouvé pour {pid} ({scan_root})")
            continue

        # ─────────────────────────────────────────────
        # Classer SEG / T1CE / FLAIR (nom + dossiers parents, conventions BraTS)
        # ─────────────────────────────────────────────
        blobs = {f: _nii_blob(f, scan_root) for f in all_nii}

        seg_files = sorted(
            [f for f in all_nii if _is_seg_volume(f, blobs[f])],
            key=lambda p: (_time_rank_from_blob(blobs[p]), _seg_sort_key(p, scan_root)),
        )
        t1ce_files = [f for f in all_nii if _is_t1_weighted_gd(blobs[f])]
        flair_files = [f for f in all_nii if _is_flair(blobs[f])]
        t1ce_ranked = sorted(
            t1ce_files,
            key=lambda p: (_time_rank_from_blob(blobs[p]), str(p)),
        )
        flair_ranked = sorted(
            flair_files,
            key=lambda p: (_time_rank_from_blob(blobs[p]), str(p)),
        )

        # ─────────────────────────────────────────────
        # DEBUG
        # ─────────────────────────────────────────────
        print(f"\n{pid}")
        print(f"  Racine scan : {scan_root}")
        print(f"  NIfTI trouvés : {len(all_nii)}")
        print(f"  SEG   : {len(seg_files)}")
        print(f"  T1CE  : {len(t1ce_files)}")
        print(f"  FLAIR : {len(flair_files)}")

        use_proxy = len(seg_files) == 0 and len(t1ce_ranked) > 0
        if len(seg_files) == 0 and not use_proxy:
            try:
                ex = [str(f.relative_to(scan_root)) for f in all_nii[:15]]
            except ValueError:
                ex = [str(f.name) for f in all_nii[:15]]
            log.warning(
                f"Aucune segmentation ni T1c pour {pid} — "
                f"exemples de chemins NIfTI : {ex}"
            )
            continue

        log.info(f"Extraction NIfTI : {pid}")

        try:
            if not use_proxy:
                seg_p = seg_files[0]
                seg_path = str(seg_p)
                seg_tp = _time_rank_from_blob(blobs[seg_p])
                t1c_same = [
                    p for p in t1ce_ranked
                    if _time_rank_from_blob(blobs[p]) == seg_tp
                ]
                fla_same = [
                    p for p in flair_ranked
                    if _time_rank_from_blob(blobs[p]) == seg_tp
                ]
                t1ce_path = str(t1c_same[0]) if t1c_same else (
                    str(t1ce_ranked[0]) if t1ce_ranked else None
                )
                flair_path = str(fla_same[0]) if fla_same else (
                    str(flair_ranked[0]) if flair_ranked else None
                )
                feats = extract_nifti_features_from_seg(
                    seg_path,
                    t1ce_path,
                    flair_path,
                )
                log.info(f"✓ {pid} chargé (SEG={Path(seg_path).name})")
            else:
                t1c_base = str(t1ce_ranked[0])
                fla_base = str(flair_ranked[0]) if flair_ranked else None
                feats = extract_nifti_features_from_t1c_proxy(t1c_base, fla_base)
                fla_part = f", FLAIR={Path(fla_base).name}" if fla_base else ""
                log.info(
                    f"✓ {pid} chargé (mode sans seg — masque proxy T1c, "
                    f"T1c={Path(t1c_base).name}{fla_part})"
                )

            feats["Patient_ID"] = pid
            records.append(feats)

        except Exception as e:

            log.warning(
                f"Erreur extraction {pid} : {e}"
            )

    # ─────────────────────────────────────────────
    # Résultat final
    # ─────────────────────────────────────────────
    if len(records) == 0:

        log.warning("Aucun patient détecté.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    log.info(
        f"{len(df)} patients chargés avec succès."
    )

    return df
# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — ESTIMATION DE LA VÉRITÉ TERRAIN (D, r, α)
# ──────────────────────────────────────────────────────────────────────────────
# Approche :
#   1. Ajustement de l'ODE de Fisher-KPP intégrée (u normalisé, sans traitement)
#      sur les timepoints pré-traitement  →  D, r via v=2√(rD) et λ=√(D/r)
#   2. Ajustement du terme de traitement −α·c(t)·u sur les timepoints
#      post-traitement                    →  α
# ──────────────────────────────────────────────────────────────────────────────

# Constantes du modèle pharmacocinétique simplifié
LAMBDA_DRUG   = np.log(2) / 14.0   # demi-vie médicament ≈ 14 jours  [1/jour]
C0_NORMALIZED = 1.0                 # concentration normalisée à t=0


def _ode_kpp_chemo(t, u, r, alpha, c0, lam):
    """ODE scalaire : du/dt = r·u·(1−u) − α·c0·exp(−λ·t)·u"""
    c_t = c0 * np.exp(-lam * t)
    return r * u * (1.0 - u) - alpha * c_t * u


def _prepare_monotonic_timeseries(
    times_days: np.ndarray, volumes_mm3: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trie par temps croissant et moyenne les volumes aux instants dupliqués.

    SciPy solve_ivp impose t_eval strictement croissant (np.diff(t_eval) > 0).
    Plusieurs lignes NIfTI pour le même timepoint produisent sinon des doublons.
    """
    t = np.asarray(times_days, dtype=float).ravel()
    v = np.asarray(volumes_mm3, dtype=float).ravel()
    if t.size != v.size:
        raise ValueError("times_days et volumes_mm3 doivent avoir la même taille.")
    if t.size == 0:
        return t, v
    order = np.argsort(t, kind="mergesort")
    t, v = t[order], v[order]
    df = pd.DataFrame({"t": t, "v": v}).groupby("t", sort=True, as_index=False)["v"].mean()
    return df["t"].to_numpy(), df["v"].to_numpy()


def _simulate_ode(times_days, u0, r, alpha,
                  c0=C0_NORMALIZED, lam=LAMBDA_DRUG):
    """Intègre l'ODE et retourne u aux instants demandés."""
    t_eval = np.asarray(times_days, dtype=float).ravel()
    if t_eval.size < 2:
        return np.array([u0])
    t0, tf = float(t_eval[0]), float(t_eval[-1])
    sol = solve_ivp(
        _ode_kpp_chemo,
        t_span=(t0, tf),
        y0=[u0],
        args=(r, alpha, c0, lam),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6, atol=1e-9,
        max_step=1.0,
    )
    return sol.y[0] if sol.success else np.full(t_eval.size, np.nan)


def _estimate_D_from_shape(sphericity: float, max_diam_mm: float,
                            elongation: float) -> float:
    """
    Heuristique physique pour D depuis la morphologie tumorale.

    Raisonnement :
      - Faible sphéricité → front irrégulier → forte invasion → grand D
      - Grand diamètre → front déjà avancé → D plus élevé
      Étalonnage littérature gliomes : D ≈ 0.01–10 mm²/jour
      (Swanson et al. 2000, Harpold et al. 2007)
    """
    irregularity = max(0.0, 1.0 - sphericity)   # [0, 1]
    extent       = np.clip(max_diam_mm / 100.0, 0.1, 5.0)
    anisotropy   = max(0.0, 1.0 - elongation)

    D_proxy = 0.05 * irregularity * extent * (1.0 + 0.5 * anisotropy)
    return np.clip(D_proxy, 0.001, 15.0)


def estimate_pde_parameters_for_patient(
    volumes_mm3:   np.ndarray,
    times_days:    np.ndarray,
    prior_chemo:   int = 1,
    rad_baseline:  dict = None,
    dt_days_inter: float = 56.0,
) -> dict:
    """
    Estime D, r et α pour un patient donné.

    Paramètres
    ----------
    volumes_mm3    : volumes tumoraux mesurés aux timepoints [mm³]
    times_days     : instants correspondants [jours depuis baseline]
    prior_chemo    : 1 si le patient reçoit une chimiothérapie
    rad_baseline   : dict de features radiomiques au baseline (optionnel)
    dt_days_inter  : intervalle moyen entre follow-ups si les dates manquent

    Retourne
    --------
    dict : D, r, alpha, r_squared, v_front, lambda_front, success
    """
    volumes = np.asarray(volumes_mm3, dtype=float)
    times   = np.asarray(times_days,  dtype=float)
    times, volumes = _prepare_monotonic_timeseries(times, volumes)

    # ── Normalisation : u = V / V_carr (carrying capacity) ──────────────────
    # On estime la carrying capacity comme 2× le volume maximal observé,
    # ce qui assure u ≤ 1 et laisse de la place à la croissance
    V_max = volumes[0]  # baseline
    V_cap = max(volumes.max() * 2.5, V_max * 2.5, 1.0)
    u_obs = np.clip(volumes / V_cap, 1e-6, 1.0)
    u0    = float(u_obs[0])

    result_base = dict(D=np.nan, r=np.nan, alpha=np.nan,
                       r_squared=np.nan, v_front=np.nan,
                       lambda_front=np.nan, success=False,
                       n_timepoints=len(times))

    if len(times) < 2:
        return result_base

    # ── Estimation jointe de r et α par optimisation globale ────────────────
    def objective(params):
        r_val, alpha_val = params
        if r_val < 0 or alpha_val < 0:
            return 1e10
        u_sim = _simulate_ode(times, u0, r_val, alpha_val)
        if np.any(np.isnan(u_sim)):
            return 1e10
        return float(np.sum((u_sim - u_obs) ** 2))

    # Bornes : r ∈ [0.001, 2.0] /jour ; α ∈ [0, 3.0] /jour
    bounds = [(1e-4, 2.0), (0.0, 3.0)]

    try:
        opt = differential_evolution(
            objective, bounds,
            seed=42, maxiter=60, tol=1e-6,
            mutation=(0.5, 1.5), recombination=0.7,
            popsize=8, workers=1, polish=False,
        )
        r_est, alpha_est = float(opt.x[0]), float(opt.x[1])
        
        # Vérifier la validité des paramètres
        if not (0 < r_est < 2.0 and 0 <= alpha_est < 3.0):
            # Fallback : heuristique basée sur la croissance observée
            if len(u_obs) >= 2:
                r_est = np.log(u_obs[-1] / u_obs[0]) / (times[-1] - times[0] + 1e-10)
                r_est = np.clip(r_est, 0.01, 0.5)
                alpha_est = 0.05
            else:
                return result_base
    except Exception as exc:
        log.debug(f"Optimisation échouée : {exc}")
        # Fallback : estimation simple basée sur la croissance observée
        if len(u_obs) >= 2:
            r_est = np.log(u_obs[-1] / u_obs[0]) / (times[-1] - times[0] + 1e-10)
            r_est = np.clip(r_est, 0.01, 0.5)
            alpha_est = 0.01
        else:
            return result_base

    # ── Goodness-of-fit (R²) ─────────────────────────────────────────────────
    u_fit  = _simulate_ode(times, u0, r_est, alpha_est)
    ss_res = float(np.sum((u_obs - u_fit) ** 2))
    ss_tot = float(np.sum((u_obs - u_obs.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # ── Estimation de D depuis la morphologie ────────────────────────────────
    if rad_baseline:
        sph  = float(rad_baseline.get("Sphericity", 0.5))
        diam = float(rad_baseline.get("Maximum3DDiameter", 50.0))
        elo  = float(rad_baseline.get("Elongation", 0.5))
    else:
        sph, diam, elo = 0.5, 50.0, 0.5

    D_est = _estimate_D_from_shape(sph, diam, elo)

    # ── Relations de Fisher-KPP ──────────────────────────────────────────────
    v_front      = 2.0 * np.sqrt(r_est * D_est)       # mm/jour
    lambda_front = np.sqrt(D_est / (r_est + 1e-10))   # mm

    return dict(
        D            = round(D_est,      5),
        r            = round(r_est,      5),
        alpha        = round(alpha_est,  5),
        r_squared    = round(r2,         4),
        v_front      = round(v_front,    4),
        lambda_front = round(lambda_front, 4),
        u0           = round(u0,         4),
        V_baseline   = float(volumes[0]),
        V_cap        = float(V_cap),
        n_timepoints = len(times),
        success      = (r2 > 0.3 and not np.isnan(r_est)),
    )


def build_ground_truth(
    tumor_vol_df: pd.DataFrame,
    clin:         pd.DataFrame,
    rad:          pd.DataFrame,
    dt_days:      float = 56.0,
) -> pd.DataFrame:
    """
    Construit le tableau de vérité terrain (D, r, α) pour chaque patient.

    Paramètres
    ----------
    tumor_vol_df : DataFrame (Patient_ID, timepoint, Volume_mm3)
    clin         : DataFrame clinique
    rad          : DataFrame radiomics
    dt_days      : jours entre deux follow-ups consécutifs (si les vraies
                   dates ne sont pas disponibles)
    """
    rad_t1c_base = rad[
        (rad["Mask"] == "mask_tumor") &
        (rad["Sequence"] == "t1c") &
        (rad["Timepoint_Label"] == "baseline")
    ].set_index("Patient_ID")

    records = []
    patients = tumor_vol_df["Patient_ID"].unique()
    log.info(f"Estimation GT pour {len(patients)} patients "
             f"(intervalle = {dt_days} j entre follow-ups)…")

    for pid in sorted(patients):
        pat = tumor_vol_df[tumor_vol_df["Patient_ID"] == pid].sort_values("timepoint")
        vols   = pat["Volume_mm3"].values
        tp_idx = pat["timepoint"].values
        times  = tp_idx.astype(float) * dt_days  # jours

        # Données cliniques
        clin_row   = clin[clin["Patient_ID"] == pid]
        prior_chemo = int(clin_row["Prior_Chemo"].values[0]) if len(clin_row) else 1

        # Radiomics baseline
        rad_row = rad_t1c_base.loc[pid].to_dict() if pid in rad_t1c_base.index else {}

        params = estimate_pde_parameters_for_patient(
            volumes_mm3  = vols,
            times_days   = times,
            prior_chemo  = prior_chemo,
            rad_baseline = rad_row,
            dt_days_inter= dt_days,
        )
        params["Patient_ID"] = pid
        records.append(params)

        ok_sym = "✓" if params["success"] else "✗"
        log.info(
            f"  {pid:<8} tp={params['n_timepoints']}  "
            f"D={params['D']:.4f}  r={params['r']:.5f}  "
            f"α={params['alpha']:.4f}  R²={params['r_squared']:.3f}  {ok_sym}"
        )

    gt_df = pd.DataFrame(records)
    n_ok  = gt_df["success"].sum()
    log.info(f"GT complète : {n_ok}/{len(gt_df)} patients valides")
    return gt_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — INGÉNIERIE DES FEATURES (X)
# ──────────────────────────────────────────────────────────────────────────────

RADIOMICS_FEATURES = [
    "MeshVolume", "SurfaceArea", "Sphericity", "Maximum3DDiameter",
    "Elongation", "VoxelValue_Mean", "VoxelValue_Maximum", "VoxelValue_SD",
    "GLCM_Entropy", "GLCM_Contrast",
    "GLSZM_SizeZoneNonUniformity", "GLDM_DependenceNonUniformity",
]

CLINICAL_FEATURES = [
    "Age", "KPS", "WHO_PS", "Dose_Gy", "Fractions",
    "Prior_Chemo", "Histology_enc", "N_metastases",
]


def build_feature_matrix(
    gt_df:      pd.DataFrame,
    rad:        pd.DataFrame,
    clin:       pd.DataFrame,
    nifti_feats: pd.DataFrame = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Construit la matrice de features X (baseline uniquement) pour le ML.

    Stratégie :
      • Radiomics T1CE tumor baseline   (texture, forme, intensité)
      • Radiomics FLAIR oedema baseline (halo péritumoral)
      • Variables cliniques encodées
      • Features NIfTI (si dossiers seg disponibles)

    Retourne (X_df, feature_cols)
    """
    # ── Radiomics T1CE tumor ─────────────────────────────────────────────────
    r_t1c = rad[
        (rad["Mask"] == "mask_tumor") &
        (rad["Sequence"] == "t1c") &
        (rad["Timepoint_Label"] == "baseline")
    ][["Patient_ID"] + RADIOMICS_FEATURES].copy()
    r_t1c.columns = ["Patient_ID"] + [f"t1c_{c}" for c in RADIOMICS_FEATURES]

    # ── Radiomics FLAIR oedema ───────────────────────────────────────────────
    r_fla = rad[
        (rad["Mask"] == "mask_oedema") &
        (rad["Sequence"] == "fla") &
        (rad["Timepoint_Label"] == "baseline")
    ][["Patient_ID"] + RADIOMICS_FEATURES].copy()
    r_fla.columns = ["Patient_ID"] + [f"fla_{c}" for c in RADIOMICS_FEATURES]

    # ── Ratio tumeur/oedema (indicateur d'invasion) ───────────────────────────
    ratio_df = r_t1c[["Patient_ID", "t1c_MeshVolume"]].merge(
        r_fla[["Patient_ID", "fla_MeshVolume"]], on="Patient_ID", how="left"
    )
    ratio_df["ratio_T_O_Volume"] = (
        ratio_df["t1c_MeshVolume"] /
        (ratio_df["fla_MeshVolume"] + 1e-3)
    )

    # ── Clinical ─────────────────────────────────────────────────────────────
    c_feats = clin[["Patient_ID"] + CLINICAL_FEATURES].copy()

    # ── Fusion ───────────────────────────────────────────────────────────────
    X = gt_df[["Patient_ID"]].copy()
    X = X.merge(r_t1c,     on="Patient_ID", how="left")
    X = X.merge(r_fla,     on="Patient_ID", how="left")
    X = X.merge(ratio_df[["Patient_ID", "ratio_T_O_Volume"]], on="Patient_ID", how="left")
    X = X.merge(c_feats,   on="Patient_ID", how="left")

    if nifti_feats is not None and not nifti_feats.empty:
        X = X.merge(nifti_feats, on="Patient_ID", how="left")

    # Colonnes numériques uniquement
    feature_cols = [c for c in X.columns
                    if c != "Patient_ID" and X[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    # Imputation par médiane (valeurs manquantes)
    X[feature_cols] = X[feature_cols].apply(lambda col: col.fillna(col.median()))

    log.info(f"Matrice X : {X.shape[0]} patients × {len(feature_cols)} features")
    return X, feature_cols


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MODÈLE ML : RÉGRESSION MULTI-SORTIES
# ──────────────────────────────────────────────────────────────────────────────

TARGET_PARAMS = ["D", "r", "alpha"]


def build_ml_model(model_type: str = "mlp"):
    """
    Crée un estimateur pour D, r, α : forêt aléatoire, GBM ou MLP (multi-sorties).

    La normalisation des entrées est faite à part (StandardScaler) avant ``fit``.
    """
    if model_type == "random_forest":
        base = RandomForestRegressor(
            n_estimators   = 300,
            max_depth       = 5,
            min_samples_leaf= 2,
            max_features    = "sqrt",
            random_state    = 42,
            n_jobs          = -1,
        )
        return MultiOutputRegressor(base, n_jobs=-1)
    if model_type == "gbm":
        base = GradientBoostingRegressor(
            n_estimators   = 200,
            max_depth       = 4,
            learning_rate   = 0.05,
            subsample       = 0.8,
            random_state    = 42,
        )
        return MultiOutputRegressor(base, n_jobs=-1)
    if model_type == "mlp":
        # Un seul MLP : sortie (n_samples, 3) pour D, r, α
        return MLPRegressor(
            hidden_layer_sizes = (128, 64, 32),
            activation         = "relu",
            solver             = "adam",
            alpha              = 1e-4,
            learning_rate_init = 1e-3,
            max_iter           = 4000,
            early_stopping     = True,
            validation_fraction = 0.12,
            n_iter_no_change   = 50,
            random_state       = 42,
        )
    raise ValueError(f"model_type inconnu : {model_type}")


def train_and_evaluate(
    X_df:        pd.DataFrame,
    gt_df:       pd.DataFrame,
    feature_cols: list[str],
    model_type:   str  = "mlp",
    test_size:    float = 0.2,
    n_cv_folds:   int   = 5,
    output_dir:   str   = ".",
) -> tuple:
    """
    Entraîne le modèle et évalue ses performances.

    1. Train/test split stratifié
    2. Cross-validation K-fold sur le train set
    3. Évaluation finale sur le test set
    4. Importance des variables

    Retourne (model, scaler, metrics_df, predictions_df)
    """
    # ── Filtrer patients valides ──────────────────────────────────────────────
    valid = gt_df[gt_df["success"] == True][["Patient_ID"] + TARGET_PARAMS].dropna()
    X_all = X_df[X_df["Patient_ID"].isin(valid["Patient_ID"])]
    X_all = X_all.set_index("Patient_ID").reindex(valid["Patient_ID"])

    X = X_all[feature_cols].values
    Y = valid[TARGET_PARAMS].values

    log.info(f"Dataset ML : {X.shape[0]} patients, {X.shape[1]} features → "
             f"{len(TARGET_PARAMS)} cibles ({', '.join(TARGET_PARAMS)})")

    if X.shape[0] < 6:
        log.error("Pas assez de patients (< 6) pour le ML.")
        return None, None, None, None

    # ── Normalisation ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Split train/test ──────────────────────────────────────────────────────
    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
        X_scaled, Y, valid["Patient_ID"].values,
        test_size=test_size, random_state=42
    )

    # ── Modèle ───────────────────────────────────────────────────────────────
    model = build_ml_model(model_type)
    model.fit(X_train, Y_train)

    # ── Cross-validation (sur train uniquement) ───────────────────────────────
    log.info(f"Cross-validation {n_cv_folds}-fold sur le train set…")
    cv_results = {}
    if X_train.shape[0] < 2:
        for param in TARGET_PARAMS:
            cv_results[param] = np.array([np.nan])
            log.info(f"  CV R² {param:>6} : — (échantillon train insuffisant)")
    else:
        n_splits = min(max(2, min(n_cv_folds, X_train.shape[0])), X_train.shape[0])
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for i, param in enumerate(TARGET_PARAMS):
            if isinstance(model, MultiOutputRegressor):
                est = clone(model.estimators_[i])
            else:
                est = clone(model)
            cv_scores = cross_val_score(
                est,
                X_train,
                Y_train[:, i],
                cv=kfold,
                scoring="r2",
            )
            cv_results[param] = cv_scores
            log.info(f"  CV R² {param:>6} : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Évaluation test ───────────────────────────────────────────────────────
    Y_pred = model.predict(X_test)

    print("\n" + "═" * 60)
    print("  RÉSULTATS — SET DE TEST")
    print("═" * 60)
    print(f"  {'Param':<8} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("  " + "─" * 42)

    metrics = []
    for i, param in enumerate(TARGET_PARAMS):
        mae  = mean_absolute_error(Y_test[:, i], Y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i]))
        r2   = r2_score(Y_test[:, i], Y_pred[:, i])
        print(f"  {param:<8} {mae:>10.5f} {rmse:>10.5f} {r2:>10.3f}")
        metrics.append(dict(param=param, MAE=mae, RMSE=rmse, R2=r2,
                            cv_R2_mean=cv_results[param].mean(),
                            cv_R2_std=cv_results[param].std()))
    print("═" * 60)

    metrics_df = pd.DataFrame(metrics)

    # ── Prédictions complètes ─────────────────────────────────────────────────
    pred_records = []
    for j, pid in enumerate(idx_test):
        row = {"Patient_ID": pid, "split": "test"}
        for k, param in enumerate(TARGET_PARAMS):
            row[f"{param}_true"] = Y_test[j, k]
            row[f"{param}_pred"] = Y_pred[j, k]
        pred_records.append(row)
    predictions_df = pd.DataFrame(pred_records)

    # ── Importance des features (arbres seulement) ────────────────────────────
    print("\n  TOP 10 FEATURES PAR PARAMÈTRE")
    print("─" * 60)
    if isinstance(model, MultiOutputRegressor):
        for i, param in enumerate(TARGET_PARAMS):
            est = model.estimators_[i]
            if hasattr(est, "feature_importances_"):
                imp = pd.Series(est.feature_importances_, index=feature_cols)
                top = imp.nlargest(10)
                print(f"\n  {param} :")
                for feat, val in top.items():
                    bar = "█" * int(val * 200)
                    print(f"    {feat:<45} {val:.4f} {bar}")
            else:
                print(f"\n  {param} : (importances non disponibles)")
    else:
        print("  MLP : pas de feature_importances_ (réseau dense).")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler,
                 "feature_cols": feature_cols}, out / "pde_ml_model.pkl")
    metrics_df.to_csv(out / "metrics.csv", index=False)
    predictions_df.to_csv(out / "predictions_test.csv", index=False)
    log.info(f"Modèle sauvegardé → {out / 'pde_ml_model.pkl'}")

    return model, scaler, metrics_df, predictions_df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — SIMULATION EDP 3D (schéma ADI simplifié → splitting de Strang)
# ──────────────────────────────────────────────────────────────────────────────

def simulate_pde_3d(
    D:          float,
    r:          float,
    alpha:      float,
    u0_3d:      np.ndarray,
    dx_mm:      float,
    t_days:     float,
    dt_days:    float  = 0.5,
    c0:         float  = 1.0,
    lambda_drug: float = LAMBDA_DRUG,
) -> tuple[np.ndarray, list[float]]:
    """
    Résout l'EDP Fisher-KPP + chimiothérapie sur une grille 3D via
    splitting de Strang (diffusion explicite + réaction).

    ∂u/∂t = D·∇²u + r·u·(1−u) − α·c(t)·u

    Paramètres
    ----------
    D, r, alpha    : paramètres PDE estimés
    u0_3d          : densité initiale 3D, shape (Nx, Ny, Nz), valeurs ∈ [0, 1]
    dx_mm          : pas de grille isotrope [mm] (même dans les 3 directions)
    t_days         : durée de simulation [jours]
    dt_days        : pas de temps [jours]
    c0             : concentration médicament normalisée à t=0
    lambda_drug    : taux de décroissance du médicament [1/jour]

    Retourne
    --------
    u_final  : np.ndarray — densité finale, même shape que u0_3d
    volumes  : list[float] — volumes tumoraux (u > 0.5) en mm³ à chaque pas
    """
    # ── Stabilité de Courant–Friedrichs–Lewy (CFL) ───────────────────────────
    dt_cfl = (dx_mm ** 2) / (6.0 * D + 1e-12)  # facteur 6 pour 3D
    dt = min(dt_days, dt_cfl * 0.9)
    n_steps = max(1, int(np.ceil(t_days / dt)))
    dt = t_days / n_steps

    u = u0_3d.astype(np.float64).copy()
    voxel_vol = dx_mm ** 3
    volumes   = [float((u > 0.5).sum() * voxel_vol)]

    log.info(f"Simulation PDE 3D : {n_steps} pas de {dt:.3f} j "
             f"(CFL dt_max={dt_cfl:.3f} j)  "
             f"grille {u.shape}  D={D:.4f}  r={r:.5f}  α={alpha:.4f}")

    for step in range(n_steps):
        t_curr = step * dt
        c_t    = c0 * np.exp(-lambda_drug * t_curr)

        # ── Diffusion (Laplacien avec conditions de Neumann) ──────────────
        # On utilise np.roll pour les voisins avec réflexion aux bords
        lap = (
              np.roll(u, -1, axis=0) + np.roll(u, 1, axis=0)
            + np.roll(u, -1, axis=1) + np.roll(u, 1, axis=1)
            + np.roll(u, -1, axis=2) + np.roll(u, 1, axis=2)
            - 6.0 * u
        ) / dx_mm ** 2

        # Correction bords (Neumann = flux nul)
        lap[0, :, :]  = lap[1, :, :]
        lap[-1, :, :] = lap[-2, :, :]
        lap[:, 0, :]  = lap[:, 1, :]
        lap[:, -1, :] = lap[:, -2, :]
        lap[:, :, 0]  = lap[:, :, 1]
        lap[:, :, -1] = lap[:, :, -2]

        # ── Mise à jour Euler explicite ────────────────────────────────────
        dudt = D * lap + r * u * (1.0 - u) - alpha * c_t * u
        u    = np.clip(u + dt * dudt, 0.0, 1.0)

        volumes.append(float((u > 0.5).sum() * voxel_vol))

    return u, volumes


def predict_and_simulate_new_patient(
    model_path:    str,
    patient_features: dict,
    seg_array:     np.ndarray  = None,
    dx_mm:         float       = 1.0,
    simulate_days: float       = 30.0,
) -> dict:
    """
    Pipeline complet pour un nouveau patient :
    1. Prédiction de D, r, α depuis ses features
    2. Simulation EDP 3D sur simulate_days jours

    Paramètres
    ----------
    model_path        : chemin vers pde_ml_model.pkl
    patient_features  : dict {feature_name: value} au baseline
    seg_array         : array 3D de la segmentation (optionnel ; sinon sphère)
    dx_mm             : résolution spatiale [mm/voxel]
    simulate_days     : horizon de simulation [jours]

    Retourne
    --------
    dict avec D_pred, r_pred, alpha_pred, u_final, volumes_evolution
    """
    bundle       = joblib.load(model_path)
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    # ── Vecteur de features ───────────────────────────────────────────────────
    x_raw = np.array([[patient_features.get(f, 0.0) for f in feature_cols]])
    x_sc  = scaler.transform(x_raw)
    y_pred = model.predict(x_sc)[0]

    D_pred, r_pred, alpha_pred = y_pred
    D_pred     = max(D_pred,     0.001)
    r_pred     = max(r_pred,     0.0001)
    alpha_pred = max(alpha_pred, 0.0)

    log.info(f"Paramètres prédits : D={D_pred:.4f} mm²/j  "
             f"r={r_pred:.5f} /j  α={alpha_pred:.4f} /j")

    # ── Condition initiale 3D ─────────────────────────────────────────────────
    if seg_array is not None:
        u0 = np.clip(seg_array.astype(float), 0.0, 1.0)
        grid_shape = u0.shape
    else:
        # Sphère synthétique de rayon 10 mm
        grid_shape = (40, 40, 40)
        cx, cy, cz = [s // 2 for s in grid_shape]
        zz, yy, xx = np.ogrid[:grid_shape[0], :grid_shape[1], :grid_shape[2]]
        radius_px   = 10.0 / dx_mm
        dist        = np.sqrt((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2)
        u0          = np.where(dist < radius_px, 0.8, 0.0).astype(float)
        log.info("Aucune segmentation fournie — sphère synthétique utilisée")

    # ── Simulation ────────────────────────────────────────────────────────────
    u_final, volumes = simulate_pde_3d(
        D=D_pred, r=r_pred, alpha=alpha_pred,
        u0_3d=u0, dx_mm=dx_mm,
        t_days=simulate_days,
    )

    V_init  = float((u0 > 0.5).sum() * dx_mm ** 3)
    V_final = float((u_final > 0.5).sum() * dx_mm ** 3)

    print(f"\n  ── Simulation à T+{simulate_days:.0f} j ──────────────────────────────")
    print(f"  Volume initial  : {V_init:>10.1f} mm³")
    print(f"  Volume final    : {V_final:>10.1f} mm³")
    delta = V_final - V_init
    sign  = "+" if delta >= 0 else ""
    print(f"  Variation       : {sign}{delta:.1f} mm³  ({sign}{100*delta/(V_init+1e-3):.1f}%)")
    print(f"  Vitesse du front: {2*np.sqrt(r_pred*D_pred):.3f} mm/j")
    print(f"  Largeur du front: {np.sqrt(D_pred/(r_pred+1e-9)):.2f} mm")

    return dict(
        D_pred           = D_pred,
        r_pred           = r_pred,
        alpha_pred       = alpha_pred,
        v_front          = 2.0 * np.sqrt(r_pred * D_pred),
        lambda_front     = np.sqrt(D_pred / (r_pred + 1e-9)),
        u_initial        = u0,
        u_final          = u_final,
        volumes_evolution= volumes,
        V_initial_mm3    = V_init,
        V_final_mm3      = V_final,
        grid_shape       = grid_shape,
        dx_mm            = dx_mm,
        simulate_days    = simulate_days,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(
    gt_df:          pd.DataFrame,
    predictions_df: pd.DataFrame,
    metrics_df:     pd.DataFrame,
    sim_result:     dict = None,
    output_dir:     str  = ".",
):
    """
    Génère et sauvegarde les figures :
    1. Scatter true vs. predicted (D, r, α)
    2. Distribution des paramètres estimés
    3. Évolution du volume simulé vs. réel (si sim_result fourni)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        log.warning("matplotlib non installé — figures non générées.")
        return

    out = Path(output_dir)
    palette = {"D": "#E63946", "r": "#2A9D8F", "alpha": "#E9C46A"}

    # ── Figure 1 : True vs. Predicted ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Prédiction ML des paramètres PDE — Vrai vs. Prédit (test set)",
                 fontsize=13, fontweight="bold")

    for ax, param in zip(axes, TARGET_PARAMS):
        if f"{param}_true" not in predictions_df.columns:
            continue
        y_true = predictions_df[f"{param}_true"]
        y_pred = predictions_df[f"{param}_pred"]
        lims   = [min(y_true.min(), y_pred.min()) * 0.9,
                  max(y_true.max(), y_pred.max()) * 1.1]

        m = metrics_df[metrics_df["param"] == param].iloc[0]
        ax.scatter(y_true, y_pred, color=palette[param], s=80, alpha=0.8,
                   edgecolors="k", linewidths=0.5, zorder=3)
        ax.plot(lims, lims, "k--", lw=1, label="Parfait")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f"{param} (vrai)", fontsize=11)
        ax.set_ylabel(f"{param} (prédit)", fontsize=11)
        ax.set_title(f"{param}  |  R²={m['R2']:.3f}  MAE={m['MAE']:.4f}", fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "true_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2 : Distribution des paramètres GT ─────────────────────────────
    valid_gt = gt_df[gt_df["success"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Distribution des paramètres PDE estimés (vérité terrain)",
                 fontsize=13, fontweight="bold")

    for ax, param in zip(axes, TARGET_PARAMS):
        vals = valid_gt[param].dropna()
        ax.hist(vals, bins=15, color=palette[param], edgecolor="white",
                linewidth=0.8, alpha=0.9)
        ax.axvline(vals.median(), color="black", linestyle="--", lw=1.5,
                   label=f"Médiane={vals.median():.4f}")
        ax.set_xlabel(param, fontsize=11)
        ax.set_ylabel("Effectif", fontsize=11)
        ax.set_title(f"Distribution de {param}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out / "parameter_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3 : Évolution du volume simulé ─────────────────────────────────
    if sim_result and "volumes_evolution" in sim_result:
        vols   = sim_result["volumes_evolution"]
        n_days = sim_result["simulate_days"]
        t_ax   = np.linspace(0, n_days, len(vols))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t_ax, vols, color="#E63946", lw=2.5, label="Volume simulé (EDP)")
        ax.axhline(vols[0], color="gray", linestyle=":", lw=1.5, label="Volume initial")
        ax.fill_between(t_ax, vols[0], vols, alpha=0.12, color="#E63946")
        ax.set_xlabel("Temps (jours)", fontsize=12)
        ax.set_ylabel("Volume tumoral (mm³)", fontsize=12)
        ax.set_title(
            f"Simulation EDP — T+{n_days:.0f} j\n"
            f"D={sim_result['D_pred']:.4f} mm²/j  "
            f"r={sim_result['r_pred']:.5f} /j  "
            f"α={sim_result['alpha_pred']:.4f} /j",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out / "simulation_volume.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    log.info(f"Figures sauvegardées dans {out}/")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 — POINT D'ENTRÉE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline ML — Paramètres PDE croissance tumorale",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Données
    p.add_argument("--data_dir",      default="./data",
                   help="Racine des données (parent du dossier étude si présent)")
    p.add_argument("--study_subdir",  default="17253793",
                   help="Sous-dossier étude : data_dir/STUDY/PXX/BraTS. "
                        "Mettre une chaîne vide si data_dir pointe déjà vers l'étude.")
    p.add_argument("--brats_subdir",  default="BraTS",
                   help="Nom du dossier BraTS sous chaque PXX")
    p.add_argument("--flat_brats_layout", action="store_true",
                   help="Ignorer study_subdir : data_dir/PXX/BraTS (pas de couche 17253793)")
    p.add_argument("--radiomics",     default="radiomics_features.xlsx",
                   help="Radiomics (.xlsx ou .csv) ; si absent, essai .csv automatique")
    p.add_argument("--clinical",      default="clinical_features.xlsx")
    p.add_argument("--nifti_feats",   default="nifti_features.xlsx")
    # Temporel
    p.add_argument("--dt_days",       type=float, default=56.0,
                   help="Intervalle en jours entre deux follow-ups consécutifs")
    # ML
    p.add_argument("--model_type",    default="mlp",
                   choices=["mlp", "random_forest", "gbm"],
                   help="mlp = Perceptron multicouche (sklearn), régression multi-sorties D,r,α")
    p.add_argument("--test_size",     type=float, default=0.2)
    p.add_argument("--n_cv_folds",    type=int,   default=5)
    # Sorties
    p.add_argument("--output",        default="./results")
    # Mode prédiction
    p.add_argument("--predict",       action="store_true",
                   help="Mode prédiction pour un nouveau patient")
    p.add_argument("--patient_dir",   default=None,
                   help="Dossier BraTS du nouveau patient (mode --predict)")
    p.add_argument("--model_dir",     default="./results",
                   help="Dossier contenant pde_ml_model.pkl (mode --predict)")
    p.add_argument("--simulate_days", type=float, default=30.0,
                   help="Horizon de simulation [jours] (mode --predict)")
    return p.parse_args()


def run_training_pipeline(args):
    """Pipeline d'entraînement complet."""
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 65)
    print("   PIPELINE ML — PARAMÈTRES PDE TUMEUR CÉRÉBRALE")
    print("   Modèle : ∂u/∂t = D·∇²u + r·u·(1−u) − α·c(t)·u")
    print("═" * 65 + "\n")

    # ── 1. Données tabulaires (.xlsx prioritaire, repli .csv) ─────────────────
    log.info("ÉTAPE 1/5 — Chargement des données tabulaires")
    rad_p = resolve_tabular_path(
        args.radiomics, "radiomics_features.xlsx", "radiomics_features.csv",
    )
    clin_p = resolve_tabular_path(
        args.clinical, "clinical_features.xlsx", "clinical_features.csv",
    )
    nif_p = resolve_tabular_path(
        args.nifti_feats, "nifti_features.xlsx", "nifti_features.csv",
    )
    rad, clin, nifti_df = load_tabular_data(rad_p, clin_p, nif_p)

    # ── 2. Volumes longitudinaux ──────────────────────────────────────────────
    log.info("ÉTAPE 2/5 — Extraction des volumes longitudinaux")
    tumor_vol_df = extract_longitudinal_volumes(nifti_df)

    # ── 3. Features NIfTI depuis les segmentations (optionnel) ───────────────
    nifti_scan_df = None
    if Path(args.data_dir).exists():
        log.info("ÉTAPE 3/5 — Extraction features NIfTI depuis dossiers BraTS")
        if args.flat_brats_layout:
            study_arg = None
        else:
            study_arg = (args.study_subdir or "").strip() or None
        nifti_scan_df = scan_patient_folders(
            args.data_dir,
            study_subdir  = study_arg,
            brats_subdir  = args.brats_subdir,
        )
        if not nifti_scan_df.empty:
            nifti_scan_df.to_csv(out / "nifti_scan_features.csv", index=False)
    else:
        log.info("ÉTAPE 3/5 — Dossiers BraTS non trouvés (data_dir absent) — ignoré")

    # ── 4. Vérité terrain ────────────────────────────────────────────────────
    log.info("ÉTAPE 4/5 — Estimation de la vérité terrain (D, r, α)")
    gt_df = build_ground_truth(tumor_vol_df, clin, rad, dt_days=args.dt_days)
    gt_df.to_csv(out / "ground_truth_parameters.csv", index=False)
    log.info(f"Vérité terrain → {out / 'ground_truth_parameters.csv'}")

    # ── 5. Matrice de features ────────────────────────────────────────────────
    log.info("ÉTAPE 5/5 — Construction de la matrice X et entraînement ML")
    X_df, feature_cols = build_feature_matrix(gt_df, rad, clin, nifti_scan_df)
    X_df.to_csv(out / "feature_matrix.csv", index=False)

    model, scaler, metrics_df, pred_df = train_and_evaluate(
        X_df, gt_df, feature_cols,
        model_type  = args.model_type,
        test_size   = args.test_size,
        n_cv_folds  = args.n_cv_folds,
        output_dir  = str(out),
    )

    if model is None:
        log.error("Entraînement échoué — dataset insuffisant.")
        return

    # ── Visualisations ────────────────────────────────────────────────────────
    plot_results(gt_df, pred_df, metrics_df, output_dir=str(out))

    print(f"\n  Tous les résultats → {out.resolve()}/")
    print("  Fichiers générés :")
    for f in sorted(out.iterdir()):
        print(f"    {f.name}")


def run_prediction_pipeline(args):
    """Mode prédiction pour un nouveau patient."""
    model_path = Path(args.model_dir) / "pde_ml_model.pkl"
    if not model_path.exists():
        log.error(f"Modèle introuvable : {model_path}")
        sys.exit(1)

    print("\n" + "═" * 55)
    print("   MODE PRÉDICTION — NOUVEAU PATIENT")
    print("═" * 55 + "\n")

    # Features depuis dossier patient
    pat_features = {}
    seg_array = None

    if args.patient_dir and Path(args.patient_dir).exists():
        pat_dir = Path(args.patient_dir)
        brats = pat_dir / "BraTS"
        if brats.is_dir():
            pat_dir = brats
        pid = pat_dir.parent.name if pat_dir.name.lower() == "brats" else pat_dir.name

        seg_files   = list(pat_dir.glob("*_seg.nii.gz"))
        t1ce_files  = list(pat_dir.glob("*_t1ce.nii.gz"))
        flair_files = list(pat_dir.glob("*_flair.nii.gz"))

        if seg_files:
            log.info(f"Extraction features depuis {pid}/…")
            feats = extract_nifti_features_from_seg(
                str(seg_files[0]),
                str(t1ce_files[0])  if t1ce_files  else None,
                str(flair_files[0]) if flair_files else None,
            )
            # Mapper vers noms attendus par le modèle
            pat_features.update({
                "t1c_Sphericity":         feats.get("seg_Sphericity", 0.5),
                "t1c_Maximum3DDiameter":  feats.get("seg_MaxDiameter_mm", 50.0),
                "t1c_Elongation":         feats.get("seg_Elongation", 0.5),
                "t1c_MeshVolume":         feats.get("seg_TumorVolume_mm3", 5000.0),
                "t1c_VoxelValue_Mean":    feats.get("seg_T1CE_Mean", 700.0),
                "t1c_VoxelValue_SD":      feats.get("seg_T1CE_Std", 100.0),
                "fla_VoxelValue_Mean":    feats.get("seg_FLAIR_Mean", 800.0),
            })

            # Charger le volume de segmentation pour la simulation
            try:
                import nibabel as nib
                seg_img   = nib.load(str(seg_files[0]))
                seg_data  = seg_img.get_fdata()
                dx_mm     = float(seg_img.header.get_zooms()[0])
                seg_array = (seg_data > 0).astype(float) * 0.8
            except Exception:
                dx_mm = 1.0
        else:
            log.warning("Pas de *_seg.nii.gz dans patient_dir — simulation sur sphère synthétique")
            dx_mm = 1.0
    else:
        log.warning("patient_dir absent — utilisation de valeurs par défaut")
        dx_mm = 1.0

    # Valeurs cliniques par défaut (à renseigner manuellement)
    pat_features.setdefault("Age",           65.0)
    pat_features.setdefault("KPS",           80.0)
    pat_features.setdefault("Dose_Gy",       20.0)
    pat_features.setdefault("Fractions",      1.0)
    pat_features.setdefault("Prior_Chemo",    1.0)
    pat_features.setdefault("Histology_enc",  0.0)  # NSCLC
    pat_features.setdefault("N_metastases",   1.0)
    pat_features.setdefault("WHO_PS",         1.0)

    sim_result = predict_and_simulate_new_patient(
        model_path        = str(model_path),
        patient_features  = pat_features,
        seg_array         = seg_array,
        dx_mm             = dx_mm,
        simulate_days     = args.simulate_days,
    )

    # Sauvegarde de la simulation
    out = Path(args.model_dir)
    try:
        import nibabel as nib
        import numpy as np
        aff = np.eye(4) * dx_mm
        nib.save(
            nib.Nifti1Image(sim_result["u_final"].astype(np.float32), aff),
            str(out / "u_final_simulated.nii.gz")
        )
        log.info(f"u_final sauvegardé → {out / 'u_final_simulated.nii.gz'}")
    except Exception:
        log.warning("nibabel non dispo — u_final non sauvegardé en NIfTI")

    vol_df = pd.DataFrame({
        "step":     range(len(sim_result["volumes_evolution"])),
        "days":     np.linspace(0, args.simulate_days, len(sim_result["volumes_evolution"])),
        "volume_mm3": sim_result["volumes_evolution"],
    })
    vol_df.to_csv(out / "simulation_volumes.csv", index=False)
    log.info(f"Évolution des volumes → {out / 'simulation_volumes.csv'}")

    plot_results(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        sim_result=sim_result, output_dir=str(out)
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = parse_args()

    if args.predict:
        run_prediction_pipeline(args)
    else:
        run_training_pipeline(args)

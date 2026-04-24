"""
patient_to_simulation.py — Extraction des données patient pour simulation EF
=============================================================================
Transforme un dossier patient (P01/, P02/, P11/...) en matrices NumPy
prêtes à l'emploi pour un modèle de diffusion-réaction tumoral.

Sorties :
  <patient_id>_sim_data.npz  contenant :
    - brain_mask      : (X,Y,Z) uint8   — domaine de calcul (0/1)
    - D_field         : (X,Y,Z) float32 — coefficient de diffusion par voxel
    - u0              : (X,Y,Z) float32 — densité tumorale initiale (baseline)
    - u_fu1..u_fu4    : (X,Y,Z) float32 — densités de validation (follow-ups)
    - spacing         : (3,)    float32 — résolution spatiale (dx,dy,dz) en mm
    - tp_dates        : dict    str     — dates des timepoints
    - tp_delta_days   : dict    int     — jours depuis baseline
    - D_white / D_gray / rho / kill_rate : scalaires biologiques

Usage :
    python patient_to_simulation.py "chemin/vers/P01/"
    python patient_to_simulation.py "chemin/vers/P01/" --d-white 0.013 --d-gray 0.0013 --rho 0.025

Install :
    pip install nibabel numpy scipy scikit-image
"""

import os, sys, re, argparse, warnings
from datetime import datetime
import numpy as np

warnings.filterwarnings("ignore")

try:
    import nibabel as nib
except ImportError:
    print("[ERREUR] nibabel manquant : pip install nibabel")
    sys.exit(1)

try:
    from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
except ImportError:
    print("[ERREUR] scipy / scikit-image manquants : pip install scipy scikit-image")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
#  CONSTANTES BIOLOGIQUES PAR DÉFAUT
#  (littérature GBM — Swanson et al., 2000)
# ══════════════════════════════════════════════════════════════

DEFAULT_D_WHITE   = 0.013   # mm²/jour — diffusion dans la matière blanche
DEFAULT_D_GRAY    = 0.0013  # mm²/jour — diffusion dans la matière grise (×10 plus lent)
DEFAULT_RHO       = 0.025   # /jour    — taux de prolifération (doublement ~28j)
DEFAULT_KILL_RATE = 0.010   # /jour    — effet chimiothérapie (temozolomide)

TIMEPOINTS = ["baseline", "fu1", "fu2", "fu3", "fu4"]


# ══════════════════════════════════════════════════════════════
#  1. DÉTECTION AUTOMATIQUE DES DATES
# ══════════════════════════════════════════════════════════════

def detect_dates(root):
    """
    Lit les noms de sous-dossiers DICOM et extrait les dates (YYYY-MM-DD).
    Retourne tp_dates (str) et tp_delta_days (int) par timepoint.
    """
    dicom_dir = os.path.join(root, "DICOM")
    if not os.path.isdir(dicom_dir):
        print("[WARN] Dossier DICOM absent — dates non disponibles.")
        return {tp: "—" for tp in TIMEPOINTS}, {tp: 0 for tp in TIMEPOINTS}

    date_re     = re.compile(r"(\d{4}-\d{2}-\d{2})")
    dates_found = set()
    for name in os.listdir(dicom_dir):
        m = date_re.search(name)
        if m:
            try:
                dates_found.add(datetime.strptime(m.group(1), "%Y-%m-%d"))
            except ValueError:
                pass

    if not dates_found:
        print("[WARN] Aucune date trouvée dans DICOM/")
        return {tp: "—" for tp in TIMEPOINTS}, {tp: 0 for tp in TIMEPOINTS}

    sorted_dates = sorted(dates_found)
    ref          = sorted_dates[0]
    tp_dates, tp_delta_days = {}, {}

    for tp, dt in zip(TIMEPOINTS, sorted_dates):
        tp_dates[tp]      = dt.strftime("%d/%m/%Y")
        tp_delta_days[tp] = (dt - ref).days

    for tp in TIMEPOINTS:
        tp_dates.setdefault(tp, "—")
        tp_delta_days.setdefault(tp, -1)

    print(f"[OK] Dates détectées :")
    for tp in TIMEPOINTS:
        if tp_dates[tp] != "—":
            print(f"     {tp:10s} → {tp_dates[tp]}  (J{tp_delta_days[tp]})")

    return tp_dates, tp_delta_days


# ══════════════════════════════════════════════════════════════
#  2. CHARGEMENT NIfTI
# ══════════════════════════════════════════════════════════════

def load_nii(path):
    """Retourne (volume float32, spacing tuple)."""
    img     = nib.load(path)
    vol     = np.asarray(img.dataobj, dtype=np.float32)
    zooms   = img.header.get_zooms()[:3]
    spacing = tuple(float(z) for z in zooms) if len(zooms) == 3 else (1.0, 1.0, 1.0)
    return vol, spacing


def find_volume(root, tp, seq):
    for brats in [os.path.join(root, "BraTS", tp), os.path.join(root, tp)]:
        for ext in [f"{seq}.nii.gz", f"{seq}.nii"]:
            p = os.path.join(brats, ext)
            if os.path.isfile(p):
                return p
    return None


def find_tumor_mask(root, tp):
    seg_roots = [
        os.path.join(root, "tumor segmentation"),
        os.path.join(root, "tumor_segmentation"),
        os.path.join(root, "tumour segmentation"),
        os.path.join(root, "tumour_segmentation"),
    ]
    seg_root = next((d for d in seg_roots if os.path.isdir(d)), None)
    if seg_root is None:
        return None

    # Fichier direct dans seg_root
    for fname in sorted(os.listdir(seg_root)):
        if tp in fname.lower() and fname.lower().endswith((".nii.gz", ".nii")):
            p = os.path.join(seg_root, fname)
            if os.path.isfile(p):
                return p

    # Sous-dossier contenant tp
    for subfolder in sorted(os.listdir(seg_root)):
        if tp not in subfolder.lower():
            continue
        sub = os.path.join(seg_root, subfolder)
        if not os.path.isdir(sub):
            continue
        for fname in sorted(os.listdir(sub)):
            if fname.lower().endswith((".nii.gz", ".nii")):
                return os.path.join(sub, fname)

    return None


def find_brain_mask_file(root):
    for entry in os.listdir(root):
        if "brain_mask" not in entry.lower():
            continue
        entry_path = os.path.join(root, entry)
        if os.path.isdir(entry_path):
            for fname in os.listdir(entry_path):
                if fname.lower().endswith((".nii", ".nii.gz")):
                    return os.path.join(entry_path, fname)
        elif os.path.isfile(entry_path) and entry.lower().endswith((".nii", ".nii.gz")):
            return entry_path
    return None


# ══════════════════════════════════════════════════════════════
#  3. SEGMENTATION MATIÈRE BLANCHE / GRISE → CHAMP D
# ══════════════════════════════════════════════════════════════

def build_diffusion_field(t1_volume, brain_mask, d_white, d_gray):
    """
    Construit le champ de diffusion D(x) à partir du volume T1 :

    Principe :
      - Dans les volumes BraTS, le fond = 0 exactement.
      - La matière blanche apparaît plus brillante que la grise en T1.
      - On applique un seuil d'Otsu sur les voxels cérébraux uniquement
        pour séparer WM (bright) de GM (dark).

    Retourne :
      D_field (float32, même shape que t1_volume)
        0.0          → hors cerveau
        d_gray       → matière grise
        d_white      → matière blanche
    """
    print("[D] Construction du champ de diffusion depuis T1...")

    # Voxels cérébraux uniquement
    brain_voxels = t1_volume[brain_mask == 1]
    brain_voxels = brain_voxels[brain_voxels > 0]   # exclure les zéros résiduels

    if len(brain_voxels) == 0:
        print("[WARN] Masque cerveau vide — D uniforme (matière grise)")
        D = np.where(brain_mask == 1, d_gray, 0.0).astype(np.float32)
        return D, None, None

    # Seuil d'Otsu pour séparer WM / GM
    thresh = threshold_otsu(brain_voxels)
    print(f"[D] Seuil Otsu WM/GM : {thresh:.1f}  (intensité T1)")

    # Lissage léger pour éviter les artefacts aux frontières
    t1_smooth = gaussian_filter(t1_volume, sigma=1.0)

    # Classification
    wm_mask = ((t1_smooth >= thresh) & (brain_mask == 1)).astype(np.uint8)
    gm_mask = ((t1_smooth <  thresh) & (brain_mask == 1) & (t1_smooth > 0)).astype(np.uint8)

    # Supprimer les petits composants parasites
    wm_mask = remove_small_objects(wm_mask.astype(bool), min_size=500).astype(np.uint8)
    gm_mask = remove_small_objects(gm_mask.astype(bool), min_size=500).astype(np.uint8)

    n_wm = int(wm_mask.sum())
    n_gm = int(gm_mask.sum())
    pct  = 100 * n_wm / max(n_wm + n_gm, 1)
    print(f"[D] Matière blanche : {n_wm:>8,} vox ({pct:.1f}%)")
    print(f"[D] Matière grise   : {n_gm:>8,} vox ({100-pct:.1f}%)")

    # Champ D
    D = np.zeros(t1_volume.shape, dtype=np.float32)
    D[wm_mask == 1] = d_white
    D[gm_mask == 1] = d_gray

    # Voxels cérébraux non classifiés → matière grise par défaut
    unclassified = (brain_mask == 1) & (wm_mask == 0) & (gm_mask == 0)
    D[unclassified] = d_gray

    return D, wm_mask, gm_mask


# ══════════════════════════════════════════════════════════════
#  4. EXTRACTION PRINCIPALE
# ══════════════════════════════════════════════════════════════

def extract(root, d_white, d_gray, rho, kill_rate, output_dir=None):
    patient_id = os.path.basename(root.rstrip("/\\"))
    print(f"\n{'═'*60}")
    print(f"  Patient : {patient_id}")
    print(f"{'═'*60}")

    # ── Dates ──────────────────────────────────────────────────
    tp_dates, tp_delta_days = detect_dates(root)

    # ── Spacing (depuis le premier volume disponible) ──────────
    spacing = (1.0, 1.0, 1.0)
    for tp in TIMEPOINTS:
        for seq in ["t1", "t1c", "fla", "t2"]:
            p = find_volume(root, tp, seq)
            if p:
                _, spacing = load_nii(p)
                break
        else:
            continue
        break
    print(f"\n[OK] Spacing : {spacing[0]:.3f} × {spacing[1]:.3f} × {spacing[2]:.3f} mm")

    # ── Masque cerveau ─────────────────────────────────────────
    print("\n[1/4] Masque cerveau...")
    brain_path = find_brain_mask_file(root)
    if brain_path:
        brain_vol, _ = load_nii(brain_path)
        brain_mask   = (brain_vol > 0.5).astype(np.uint8)
        print(f"[OK] {os.path.basename(brain_path)}  →  {brain_mask.sum():,} voxels")
    else:
        print("[WARN] Masque cerveau introuvable — construction depuis T1 baseline")
        t1_path = find_volume(root, "baseline", "t1")
        if t1_path is None:
            print("[ERREUR] Impossible de construire le masque cerveau.")
            sys.exit(1)
        t1_vol, _ = load_nii(t1_path)
        brain_mask = (t1_vol > 10).astype(np.uint8)
        brain_mask = binary_closing(brain_mask, structure=np.ones((5,5,5))).astype(np.uint8)

    shape = brain_mask.shape
    print(f"[OK] Shape domaine : {shape}")

    # ── Champ de diffusion D ───────────────────────────────────
    print("\n[2/4] Champ de diffusion D(x)...")
    t1_path = find_volume(root, "baseline", "t1")
    if t1_path is None:
        t1_path = find_volume(root, "baseline", "t1c")
    if t1_path:
        t1_vol, _ = load_nii(t1_path)
        D_field, wm_mask, gm_mask = build_diffusion_field(t1_vol, brain_mask, d_white, d_gray)
    else:
        print("[WARN] T1 baseline introuvable — D uniforme (matière grise)")
        D_field  = np.where(brain_mask == 1, d_gray, 0.0).astype(np.float32)
        wm_mask  = None
        gm_mask  = None

    print(f"[OK] D_field : min={D_field.min():.4f}  max={D_field.max():.4f}  "
          f"(mm²/jour)")

    # ── État initial u0 (baseline) ─────────────────────────────
    print("\n[3/4] État initial u0 (baseline)...")
    mask0_path = find_tumor_mask(root, "baseline")
    if mask0_path is None:
        print("[ERREUR] Masque tumoral baseline introuvable.")
        sys.exit(1)
    mask0_vol, _ = load_nii(mask0_path)
    u0 = (mask0_vol > 0.5).astype(np.float32)   # densité : 0.0 ou 1.0
    print(f"[OK] u0 : {int(u0.sum()):,} voxels tumoraux  "
          f"({u0.sum() * np.prod(spacing) / 1000:.2f} cm³)")

    # ── Masques de validation (fu1 → fu4) ─────────────────────
    print("\n[4/4] Masques de validation (follow-ups)...")
    u_followups = {}
    for tp in ["fu1", "fu2", "fu3", "fu4"]:
        mask_path = find_tumor_mask(root, tp)
        if mask_path:
            mv, _ = load_nii(mask_path)
            u_followups[tp] = (mv > 0.5).astype(np.float32)
            n_vox = int(u_followups[tp].sum())
            vol   = n_vox * np.prod(spacing) / 1000
            print(f"[OK] {tp:6s} → {n_vox:>8,} vox  ({vol:.2f} cm³)  "
                  f"[{tp_dates[tp]}  J{tp_delta_days[tp]}]")
        else:
            print(f"[--] {tp:6s} → introuvable")

    # ── Résumé des paramètres biologiques ──────────────────────
    print(f"\n{'─'*60}")
    print(f"  Paramètres biologiques")
    print(f"{'─'*60}")
    print(f"  D_white    = {d_white}  mm²/jour  (matière blanche)")
    print(f"  D_gray     = {d_gray}  mm²/jour  (matière grise)")
    print(f"  rho        = {rho}  /jour     (prolifération)")
    print(f"  kill_rate  = {kill_rate}  /jour     (chimio TMZ)")
    print(f"{'─'*60}")

    # ── Sauvegarde .npz ────────────────────────────────────────
    if output_dir is None:
        output_dir = root
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{patient_id}_sim_data.npz")

    save_dict = {
        "brain_mask":    brain_mask,
        "D_field":       D_field,
        "u0":            u0,
        "spacing":       np.array(spacing, dtype=np.float32),
        # Scalaires biologiques
        "D_white":       np.float32(d_white),
        "D_gray":        np.float32(d_gray),
        "rho":           np.float32(rho),
        "kill_rate":     np.float32(kill_rate),
    }

    # Masques WM/GM (optionnels)
    if wm_mask is not None:
        save_dict["wm_mask"] = wm_mask
    if gm_mask is not None:
        save_dict["gm_mask"] = gm_mask

    # Follow-ups
    for tp, u in u_followups.items():
        save_dict[f"u_{tp}"] = u

    # Métadonnées temporelles (encodées en tableaux)
    tp_list  = list(tp_dates.keys())
    dates_arr = np.array([tp_dates[tp]      for tp in tp_list], dtype=object)
    delta_arr = np.array([tp_delta_days[tp] for tp in tp_list], dtype=np.int32)
    save_dict["tp_names"]      = np.array(tp_list, dtype=object)
    save_dict["tp_dates"]      = dates_arr
    save_dict["tp_delta_days"] = delta_arr

    np.savez_compressed(out_path, **save_dict)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n[OK] Fichier sauvegardé : {out_path}  ({size_mb:.1f} MB)")

    # ── Résumé lisible ─────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Contenu de {patient_id}_sim_data.npz")
    print(f"{'═'*60}")
    rows = [
        ("brain_mask",   str(brain_mask.shape), "uint8",   "Domaine (0/1)"),
        ("D_field",      str(D_field.shape),    "float32", f"Diffusion mm²/j (WM={d_white}, GM={d_gray})"),
        ("u0",           str(u0.shape),          "float32", "Densité tumorale initiale"),
    ]
    for tp, u in u_followups.items():
        rows.append((f"u_{tp}", str(u.shape), "float32", f"Validation {tp_dates[tp]} (J{tp_delta_days[tp]})"))
    rows += [
        ("spacing",      "(3,)",  "float32", f"{spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm"),
        ("D_white",      "scalar","float32", f"{d_white} mm²/j"),
        ("D_gray",       "scalar","float32", f"{d_gray} mm²/j"),
        ("rho",          "scalar","float32", f"{rho} /j"),
        ("kill_rate",    "scalar","float32", f"{kill_rate} /j"),
        ("tp_names",     "(N,)",  "str",     "Noms des timepoints"),
        ("tp_dates",     "(N,)",  "str",     "Dates acquisitions"),
        ("tp_delta_days","(N,)",  "int32",   "Jours depuis baseline"),
    ]
    print(f"  {'Clé':<16} {'Shape':<14} {'Type':<9} Description")
    print(f"  {'─'*16} {'─'*14} {'─'*9} {'─'*30}")
    for key, shape, dtype, desc in rows:
        print(f"  {key:<16} {shape:<14} {dtype:<9} {desc}")

    print(f"\n  → Pour charger : data = np.load('{patient_id}_sim_data.npz', allow_pickle=True)")
    print(f"  → Accès        : brain = data['brain_mask']  |  D = data['D_field']  |  u0 = data['u0']")
    print()

    return out_path


# ══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TumorFlow — Extraction données patient pour simulation EF"
    )
    parser.add_argument("folder",
        help="Dossier patient (ex: P01/)")
    parser.add_argument("--d-white", type=float, default=DEFAULT_D_WHITE,
        help=f"Coefficient diffusion matière blanche mm²/j (défaut: {DEFAULT_D_WHITE})")
    parser.add_argument("--d-gray",  type=float, default=DEFAULT_D_GRAY,
        help=f"Coefficient diffusion matière grise mm²/j (défaut: {DEFAULT_D_GRAY})")
    parser.add_argument("--rho",     type=float, default=DEFAULT_RHO,
        help=f"Taux de prolifération /j (défaut: {DEFAULT_RHO})")
    parser.add_argument("--kill",    type=float, default=DEFAULT_KILL_RATE,
        help=f"Taux de mort chimio /j (défaut: {DEFAULT_KILL_RATE})")
    parser.add_argument("--output",  default=None,
        help="Dossier de sortie (défaut: dossier patient)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"[ERREUR] Dossier introuvable : {args.folder}")
        sys.exit(1)

    extract(
        root       = args.folder,
        d_white    = args.d_white,
        d_gray     = args.d_gray,
        rho        = args.rho,
        kill_rate  = args.kill,
        output_dir = args.output,
    )

if __name__ == "__main__":
    main()
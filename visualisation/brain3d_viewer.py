"""
brain3d_viewer.py — Viewer 3D Cérébral IRM + Tumeur (PyVista + NIfTI)
======================================================================
Compatible P01, P02, P11 — détection automatique des dates et masques.

Usage :
    python brain3d_viewer.py "chemin/vers/P01/"
    python brain3d_viewer.py "chemin/vers/P01/" --tp baseline
    python brain3d_viewer.py "chemin/vers/P01/" --tp fu2 --seq t1c
    python brain3d_viewer.py "chemin/vers/P01/" --all
    python brain3d_viewer.py "chemin/vers/P01/" --offscreen --output cerveau.png

Install :
    pip install pyvista nibabel numpy scipy scikit-image
"""

import os, sys, re, argparse, warnings
from datetime import datetime
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_closing, gaussian_filter
from skimage import measure

warnings.filterwarnings("ignore")

try:
    import nibabel as nib
except ImportError:
    print("[ERREUR] nibabel manquant : pip install nibabel")
    sys.exit(1)

try:
    import pyvista as pv
except ImportError:
    print("[ERREUR] pyvista manquant : pip install pyvista")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════

TIMEPOINTS = ["baseline", "fu1", "fu2", "fu3", "fu4"]
SEQUENCES  = ["fla", "t1", "t1c", "t2"]
SEQ_LABELS = {"fla": "FLAIR", "t1": "T1", "t1c": "T1c", "t2": "T2"}


# ══════════════════════════════════════════════════════════════
#  1. DÉTECTION AUTOMATIQUE DES DATES
# ══════════════════════════════════════════════════════════════

def detect_dates(root):
    """
    Lit les noms de sous-dossiers dans <root>/DICOM/ et extrait les dates.
    Format : <SEQ>_<YYYY-MM-DD> ou <SEQ>_<YYYY-MM-DD> (1)  — doublons ignorés.
    Associe chaque date unique (ordre chrono) à baseline, fu1, fu2...
    """
    fallback = ({tp: "—" for tp in TIMEPOINTS}, {tp: "—" for tp in TIMEPOINTS})

    dicom_dir = os.path.join(root, "DICOM")
    if not os.path.isdir(dicom_dir):
        print("[WARN] Dossier DICOM introuvable — dates non disponibles.")
        return fallback

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
        print("[WARN] Aucune date trouvée dans les noms DICOM.")
        return fallback

    sorted_dates = sorted(dates_found)
    ref_date     = sorted_dates[0]
    tp_dates, tp_delta = {}, {}

    for tp, dt in zip(TIMEPOINTS, sorted_dates):
        d            = (dt - ref_date).days
        tp_dates[tp] = dt.strftime("%d/%m/%Y")
        if d == 0:
            tp_delta[tp] = "J0 — référence"
        elif d < 60:
            tp_delta[tp] = f"J{d} — +{d // 7} sem."
        elif d < 400:
            tp_delta[tp] = f"J{d} — +{round(d / 30.44, 1)} mois"
        else:
            tp_delta[tp] = f"J{d} — +{round(d / 365.25, 1)} ans"

    for tp in TIMEPOINTS:
        tp_dates.setdefault(tp, "—")
        tp_delta.setdefault(tp, "—")

    print(f"[OK] {len(sorted_dates)} date(s) détectée(s) :")
    for tp in TIMEPOINTS:
        if tp_dates[tp] != "—":
            print(f"     {tp:10s} → {tp_dates[tp]}  ({tp_delta[tp]})")

    return tp_dates, tp_delta


# ══════════════════════════════════════════════════════════════
#  2. CHARGEMENT NIfTI
# ══════════════════════════════════════════════════════════════

def load_nii(path):
    img     = nib.load(path)
    vol     = np.asarray(img.dataobj, dtype=np.float32)
    zooms   = img.header.get_zooms()[:3]
    spacing = tuple(float(z) for z in zooms) if len(zooms) == 3 else (1.0, 1.0, 1.0)
    return vol, spacing


def find_volume(root, tp, seq):
    """Cherche <seq>.nii.gz dans BraTS/<tp>/."""
    for brats in [os.path.join(root, "BraTS", tp), os.path.join(root, tp)]:
        for ext in [f"{seq}.nii.gz", f"{seq}.nii"]:
            path = os.path.join(brats, ext)
            if os.path.isfile(path):
                return path
    return None


def find_tumor_mask(root, tp):
    """
    Cherche le masque tumoral pour un timepoint.
    Gère toutes les variantes :
      - P01 : tumor segmentation/<PXX>_tumor_mask_<tp>.nii/tumour_mask*.nii
      - P02/P11 : tumor_segmentation/<PXX>_tumor_mask_<tp>.nii.gz  (fichier direct)
    """
    seg_roots = [
        os.path.join(root, "tumor segmentation"),
        os.path.join(root, "tumor_segmentation"),
        os.path.join(root, "tumour segmentation"),
        os.path.join(root, "tumour_segmentation"),
    ]
    seg_root = next((d for d in seg_roots if os.path.isdir(d)), None)
    if seg_root is None:
        return None

    # Cas direct : fichier dans seg_root dont le nom contient tp
    for fname in sorted(os.listdir(seg_root)):
        if tp in fname.lower() and fname.lower().endswith((".nii.gz", ".nii")):
            path = os.path.join(seg_root, fname)
            if os.path.isfile(path):
                return path

    # Cas sous-dossier : <PXX>_tumor_mask_<tp>.nii/ contenant n'importe quel .nii
    for subfolder in sorted(os.listdir(seg_root)):
        if tp not in subfolder.lower():
            continue
        sub_path = os.path.join(seg_root, subfolder)
        if not os.path.isdir(sub_path):
            continue
        for fname in sorted(os.listdir(sub_path)):
            if fname.lower().endswith((".nii.gz", ".nii")):
                return os.path.join(sub_path, fname)

    return None


def find_brain_mask(root):
    """Cherche le masque cerveau global — compatible P01/P02/P11."""
    patient_id = os.path.basename(root.rstrip("/\\"))

    # Chercher d'abord dans les sous-dossiers *brain_mask*
    for entry in os.listdir(root):
        if "brain_mask" in entry.lower():
            entry_path = os.path.join(root, entry)
            if os.path.isdir(entry_path):
                for fname in os.listdir(entry_path):
                    if fname.lower().endswith((".nii", ".nii.gz")):
                        return os.path.join(entry_path, fname)
            elif os.path.isfile(entry_path) and entry.lower().endswith((".nii", ".nii.gz")):
                return entry_path

    return None


# ══════════════════════════════════════════════════════════════
#  3. SEGMENTATION CERVEAU (fallback)
# ══════════════════════════════════════════════════════════════

def segment_brain(volume):
    print("[SEG] Segmentation cerveau automatique...")
    brain = (volume > 10).astype(np.uint8)
    brain = binary_closing(brain, structure=np.ones((5, 5, 5))).astype(np.uint8)
    for i in range(brain.shape[2]):
        brain[:, :, i] = binary_fill_holes(brain[:, :, i]).astype(np.uint8)
    from scipy.ndimage import label as ndlabel
    lbl, n = ndlabel(brain)
    if n > 0:
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        brain = (lbl == np.argmax(sizes)).astype(np.uint8)
    print(f"[SEG] Cerveau : {brain.sum():,} voxels")
    return brain


# ══════════════════════════════════════════════════════════════
#  4. MAILLAGE 3D
# ══════════════════════════════════════════════════════════════

def mask_to_mesh(binary_mask, spacing, sigma=1.5, step=2, smooth_iter=60):
    if binary_mask.sum() == 0:
        return None
    sm = gaussian_filter(binary_mask.astype(np.float32), sigma=sigma)
    if sm.max() < 0.3:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(
            sm, level=0.4, spacing=spacing,
            step_size=step, allow_degenerate=False,
        )
    except Exception as e:
        print(f"[WARN] marching_cubes : {e}"); return None
    if len(faces) == 0:
        return None
    n_f  = len(faces)
    pv_f = np.hstack([np.full((n_f, 1), 3, dtype=np.int64),
                      faces.astype(np.int64)]).flatten()
    mesh = pv.PolyData(verts, pv_f)
    if smooth_iter > 0:
        mesh = mesh.smooth(n_iter=smooth_iter, relaxation_factor=0.15)
    mesh.compute_normals(inplace=True)
    print(f"[MESH] {mesh.n_points:,} pts, {mesh.n_cells:,} faces")
    return mesh


# ══════════════════════════════════════════════════════════════
#  5. STATISTIQUES TUMORALES
# ══════════════════════════════════════════════════════════════

def tumor_stats(volume, mask, spacing):
    sx, sy, sz  = spacing
    voxel_vol   = sx * sy * sz
    n_vox       = int(mask.sum())
    if n_vox == 0:
        return None
    vol_cm3 = round(n_vox * voxel_vol / 1000, 2)
    diam_mm = round(2 * ((3 * n_vox * voxel_vol) / (4 * np.pi)) ** (1/3), 1)
    hu      = volume[mask.astype(bool)]
    return {
        "n_vox": n_vox, "volume_cm3": vol_cm3, "diam_mm": diam_mm,
        "int_mean": round(float(hu.mean()), 1),
        "int_min":  round(float(hu.min()),  1),
        "int_max":  round(float(hu.max()),  1),
    }


# ══════════════════════════════════════════════════════════════
#  6. AFFICHAGE — UN TIMEPOINT
# ══════════════════════════════════════════════════════════════

def show_single(root, tp, seq, tp_dates, tp_delta, patient_id,
                offscreen=False, output_png=None):
    print(f"\n{'─'*60}")
    print(f"  Patient   : {patient_id}")
    print(f"  Timepoint : {tp.upper()}  |  Date : {tp_dates[tp]}  |  {tp_delta[tp]}")
    print(f"  Séquence  : {SEQ_LABELS[seq]}")
    print(f"{'─'*60}")

    vol_path = find_volume(root, tp, seq)
    if vol_path is None:
        print(f"[ERREUR] Volume {seq} introuvable pour {tp}"); sys.exit(1)
    volume, spacing = load_nii(vol_path)
    print(f"[OK] Volume {volume.shape}  "
          f"spacing {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm")

    mask_path = find_tumor_mask(root, tp)
    mask = None
    if mask_path:
        mask_vol, _ = load_nii(mask_path)
        mask = (mask_vol > 0.5).astype(np.uint8)
        print(f"[OK] Masque tumeur : {mask.sum():,} voxels  "
              f"({os.path.basename(mask_path)})")
    else:
        print(f"[WARN] Masque tumeur introuvable pour {tp}")

    brain_path = find_brain_mask(root)
    if brain_path:
        bv, _ = load_nii(brain_path)
        brain_mask = (bv > 0.5).astype(np.uint8)
        print(f"[OK] Masque cerveau : {brain_mask.sum():,} voxels  "
              f"({os.path.basename(brain_path)})")
    else:
        brain_mask = segment_brain(volume)

    print("\n[MESH] Construction surface cerveau...")
    brain_pd = mask_to_mesh(brain_mask, spacing, sigma=2.5, step=2, smooth_iter=100)

    tumor_pd = stats = None
    if mask is not None and mask.any():
        print("[MESH] Construction surface tumeur...")
        tumor_pd = mask_to_mesh(mask, spacing, sigma=1.0, step=1, smooth_iter=40)
        stats    = tumor_stats(volume, mask, spacing)

    _render(brain_pd, tumor_pd, stats,
            tp=tp, seq=seq,
            tp_dates=tp_dates, tp_delta=tp_delta, patient_id=patient_id,
            offscreen=offscreen, output_png=output_png)


# ══════════════════════════════════════════════════════════════
#  7. AFFICHAGE — TOUS LES TIMEPOINTS CÔTE À CÔTE
# ══════════════════════════════════════════════════════════════

def show_all(root, seq, tp_dates, tp_delta, patient_id,
             offscreen=False, output_png=None):
    available = [tp for tp in TIMEPOINTS if find_volume(root, tp, seq) is not None]
    n = len(available)
    if n == 0:
        print("[ERREUR] Aucun volume trouvé."); sys.exit(1)

    print(f"\n[INFO] Comparaison — {n} timepoints  |  {patient_id}  |  {SEQ_LABELS[seq]}")

    pv.set_plot_theme("dark")
    cols = min(n, 5)
    pl   = pv.Plotter(
        shape=(1, cols),
        title=f"Évolution tumorale {patient_id} — {SEQ_LABELS[seq]}",
        window_size=(300 * cols, 520),
        off_screen=offscreen,
    )
    pl.set_background("#04050e")

    brain_path = find_brain_mask(root)

    for col, tp in enumerate(available):
        pl.subplot(0, col)

        vol_path        = find_volume(root, tp, seq)
        volume, spacing = load_nii(vol_path)

        if brain_path:
            bv, _ = load_nii(brain_path)
            brain_mask = (bv > 0.5).astype(np.uint8)
        else:
            brain_mask = segment_brain(volume)

        brain_pd = mask_to_mesh(brain_mask, spacing, sigma=2.5, step=2, smooth_iter=80)

        tumor_pd = stats = None
        mask_path = find_tumor_mask(root, tp)
        if mask_path:
            mv, _ = load_nii(mask_path)
            mask  = (mv > 0.5).astype(np.uint8)
            if mask.any():
                tumor_pd = mask_to_mesh(mask, spacing, sigma=1.0, step=1, smooth_iter=30)
                stats    = tumor_stats(volume, mask, spacing)

        if brain_pd:
            pl.add_mesh(brain_pd, color="#b0c8e8", opacity=0.15,
                        smooth_shading=True, specular=0.5, diffuse=0.8)
        if tumor_pd:
            pl.add_mesh(tumor_pd, color="#ff2200", opacity=1.0,
                        smooth_shading=True, specular=1.0,
                        specular_power=80, ambient=0.3)
            pl.add_mesh(tumor_pd.copy(), color="#ff6600", opacity=0.25,
                        smooth_shading=True)

        vol_str = f"{stats['volume_cm3']} cm³" if stats else "—"
        pl.add_text(
            f"{tp.upper()}\n{tp_dates[tp]}\n{tp_delta[tp]}\nVol: {vol_str}",
            font_size=8, color="white", position="upper_left", font="courier",
        )

        pl.remove_all_lights()
        pl.add_light(pv.Light(position=( 300,  400, 300), color="white",   intensity=0.9))
        pl.add_light(pv.Light(position=(-200, -200, 200), color="#3355aa", intensity=0.5))
        pl.reset_camera()
        pl.camera.elevation = 10
        pl.camera.azimuth   = 20

    pl.add_text(
        f"Évolution tumorale — {patient_id}",
        position="lower_edge", font_size=8, color="#334466",
    )

    if offscreen and output_png:
        pl.screenshot(output_png); print(f"[OK] Screenshot : {output_png}")
    else:
        pl.show()


# ══════════════════════════════════════════════════════════════
#  8. RENDU PRINCIPAL
# ══════════════════════════════════════════════════════════════

def _render(brain_pd, tumor_pd, stats, tp, seq,
            tp_dates, tp_delta, patient_id,
            offscreen=False, output_png=None):
    pv.set_plot_theme("dark")
    pl = pv.Plotter(
        title=f"TumorFlow 3D — {patient_id} | {tp.upper()} | {SEQ_LABELS[seq]}",
        window_size=(1400, 900),
        off_screen=offscreen,
    )
    pl.set_background("#04050e")

    if brain_pd:
        pl.add_mesh(brain_pd, color="#c0d8f0", opacity=0.14,
                    smooth_shading=True, specular=0.7, specular_power=40,
                    diffuse=0.85, ambient=0.08, name="brain")
        pl.add_mesh(brain_pd, color="#4488cc", opacity=0.06,
                    style="wireframe", line_width=0.4, name="brain_wf")

    if tumor_pd:
        pl.add_mesh(tumor_pd, color="#ff1500", opacity=1.0,
                    smooth_shading=True, specular=1.0, specular_power=120,
                    diffuse=0.75, ambient=0.35, name="tumor")
        pl.add_mesh(tumor_pd.copy(), color="#ff6600", opacity=0.22,
                    smooth_shading=True, specular=0.3, name="tumor_glow")

    pl.remove_all_lights()
    pl.add_light(pv.Light(position=( 400,  500,  400), color="white",   intensity=0.85))
    pl.add_light(pv.Light(position=(-300, -200, -200), color="#3355aa", intensity=0.55))
    pl.add_light(pv.Light(position=(   0, -400,  300), color="white",   intensity=0.35))
    pl.add_light(pv.Light(position=(   0,    0, -500), color="#112266", intensity=0.45))

    pl.add_text(
        f"TumorFlow 3D — {patient_id}\n"
        f"Séquence : {SEQ_LABELS[seq]}\n"
        f"Timepoint : {tp.upper()}  ({tp_dates[tp]})\n"
        f"{tp_delta[tp]}",
        position="upper_left", font_size=11, color="white", font="courier",
    )

    if stats:
        pl.add_text(
            f"Tumeur\n"
            f"  Ø  {stats['diam_mm']} mm\n"
            f"  Vol {stats['volume_cm3']} cm³\n"
            f"  {stats['n_vox']:,} voxels\n"
            f"  Int. moy. {stats['int_mean']}\n"
            f"  Int. {stats['int_min']} → {stats['int_max']}",
            position="upper_right", font_size=11, color="#ff6622", font="courier",
        )
    else:
        pl.add_text("Aucun masque tumeur", position="upper_right",
                    font_size=11, color="#888888")

    pl.add_text(
        "Clic + glisser : rotation   |   Molette : zoom   |   Q : quitter",
        position="lower_edge", font_size=9, color="#334466",
    )

    pl.reset_camera()
    pl.camera.elevation = 12
    pl.camera.azimuth   = 25

    if offscreen and output_png:
        pl.screenshot(output_png, transparent_background=False)
        print(f"[OK] Screenshot : {output_png}")
    else:
        print("\n[Fenêtre 3D ouverte — fermer pour quitter]\n")
        pl.show()


# ══════════════════════════════════════════════════════════════
#  9. POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TumorFlow 3D — Viewer cérébral IRM + tumeur (PyVista)"
    )
    parser.add_argument("folder",      help="Dossier patient (ex: P01/)")
    parser.add_argument("--tp",        default="baseline", choices=TIMEPOINTS,
                        help="Timepoint (défaut: baseline)")
    parser.add_argument("--seq",       default="t1c", choices=SEQUENCES,
                        help="Séquence IRM (défaut: t1c)")
    parser.add_argument("--all",       action="store_true",
                        help="Tous les timepoints côte à côte")
    parser.add_argument("--offscreen", action="store_true",
                        help="Export PNG sans fenêtre")
    parser.add_argument("--output",    default="brain3d.png",
                        help="Fichier PNG de sortie (--offscreen)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"[ERREUR] Dossier introuvable : {args.folder}"); sys.exit(1)

    patient_id         = os.path.basename(args.folder.rstrip("/\\"))
    tp_dates, tp_delta = detect_dates(args.folder)

    if args.all:
        show_all(args.folder, args.seq,
                 tp_dates, tp_delta, patient_id,
                 offscreen=args.offscreen, output_png=args.output)
    else:
        show_single(args.folder, args.tp, args.seq,
                    tp_dates, tp_delta, patient_id,
                    offscreen=args.offscreen, output_png=args.output)


if __name__ == "__main__":
    main()
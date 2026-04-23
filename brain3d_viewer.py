"""
brain3d_viewer.py — Viewer 3D Cérébral IRM + Tumeur (PyVista + NIfTI)
======================================================================
Affiche une fenêtre 3D interactive avec :
  - Surface du cerveau (translucide, gris bleuté)
  - Tumeur en rouge vif avec halo lumineux
  - Navigation par timepoint (baseline → fu4) avec évolution tumorale

Usage :
    python brain3d_viewer.py "chemin/vers/P01/"
    python brain3d_viewer.py "chemin/vers/P01/" --tp baseline
    python brain3d_viewer.py "chemin/vers/P01/" --tp fu2 --seq t1c
    python brain3d_viewer.py "chemin/vers/P01/" --all      # tous les timepoints côte à côte
    python brain3d_viewer.py "chemin/vers/P01/" --offscreen --output cerveau.png

Install :
    pip install pyvista nibabel numpy scipy scikit-image
"""

import os, sys, argparse, warnings
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
#  CONSTANTES PATIENT P01
# ══════════════════════════════════════════════════════════════

TIMEPOINTS = ["baseline", "fu1", "fu2", "fu3", "fu4"]
SEQUENCES  = ["fla", "t1", "t1c", "t2"]
SEQ_LABELS = {"fla": "FLAIR", "t1": "T1", "t1c": "T1c", "t2": "T2"}

TP_DATES = {
    "baseline": "07/01/2020",
    "fu1":      "14/02/2020",
    "fu2":      "20/05/2020",
    "fu3":      "10/08/2020",
    "fu4":      "10/11/2020",
}
TP_DELTA = {
    "baseline": "J0   — référence",
    "fu1":      "J38  — +5 semaines",
    "fu2":      "J134 — +4,5 mois",
    "fu3":      "J216 — +7 mois",
    "fu4":      "J308 — +10 mois",
}

TUMOR_FOLDERS = {
    "baseline": ["P01_tumor_mask_baseline.nii", "tumour_mask_corr.nii"],
    "fu1":      ["P01_tumor_mask_fu1.nii",      "tumour_mask.nii"],
    "fu2":      ["P01_tumor_mask_fu2.nii",      "tumour_mask.nii"],
    "fu3":      ["P01_tumor_mask_fu3.nii",      "tumour_mask.nii"],
    "fu4":      ["P01_tumor_mask_fu4.nii",      "tumour_mask.nii"],
}


# ══════════════════════════════════════════════════════════════
#  1. CHARGEMENT NIfTI
# ══════════════════════════════════════════════════════════════

def load_nii(path):
    img = nib.load(path)
    vol = np.asarray(img.dataobj, dtype=np.float32)
    # Récupère le voxel spacing depuis l'en-tête (en mm)
    hdr = img.header
    zooms = hdr.get_zooms()[:3]
    spacing = tuple(float(z) for z in zooms) if len(zooms) == 3 else (1.0, 1.0, 1.0)
    return vol, spacing


def find_volume(root, tp, seq):
    """Cherche fla/t1/t1c/t2.nii.gz dans BraTS/<tp>/."""
    brats = os.path.join(root, "BraTS", tp)
    if not os.path.isdir(brats):
        brats = os.path.join(root, tp)
    for ext in [f"{seq}.nii.gz", f"{seq}.nii"]:
        path = os.path.join(brats, ext)
        if os.path.isfile(path):
            return path
    return None


def find_mask(root, tp):
    """Cherche le masque tumoral dans tumor segmentation/<dossier>/<fichier>."""
    seg_roots = [
        os.path.join(root, "tumor segmentation"),
        os.path.join(root, "tumour segmentation"),
        root,
    ]
    folder_name, file_name = TUMOR_FOLDERS[tp]
    for seg_root in seg_roots:
        path = os.path.join(seg_root, folder_name, file_name)
        if os.path.isfile(path):
            return path
    return None


def find_brain_mask(root):
    """Cherche le masque cerveau global (optionnel)."""
    candidates = [
        os.path.join(root, "P01_brain_mask.nii", "RP01a_brain_mask.nii"),
        os.path.join(root, "RP01a_brain_mask.nii"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


# ══════════════════════════════════════════════════════════════
#  2. SEGMENTATION CERVEAU (si pas de masque fourni)
# ══════════════════════════════════════════════════════════════

def segment_brain(volume):
    """
    Segmente le cerveau depuis un volume T1/FLAIR normalisé BraTS.
    Les volumes BraTS sont déjà crânio-extraits à 80% → seuillage simple suffit.
    """
    print("[SEG] Segmentation cerveau...")

    # En BraTS, le fond = 0 exactement. Le cerveau = tout le reste.
    brain = (volume > 10).astype(np.uint8)

    # Fermeture pour combler les petits trous (ventricules, sulci)
    brain = binary_closing(brain, structure=np.ones((5, 5, 5))).astype(np.uint8)

    # Remplir les trous tranche par tranche
    for i in range(brain.shape[2]):
        brain[:, :, i] = binary_fill_holes(brain[:, :, i]).astype(np.uint8)

    # Garder uniquement le plus gros composant connexe (= cerveau)
    from scipy.ndimage import label as ndlabel
    lbl, n = ndlabel(brain)
    if n > 0:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        brain = (lbl == np.argmax(sizes)).astype(np.uint8)

    print(f"[SEG] Cerveau : {brain.sum():,} voxels")
    return brain


# ══════════════════════════════════════════════════════════════
#  3. MAILLAGE 3D (marching cubes → PyVista PolyData)
# ══════════════════════════════════════════════════════════════

def mask_to_mesh(binary_mask, spacing, sigma=1.5, step=2, smooth_iter=60):
    """
    Masque binaire numpy → PyVista PolyData lissé.
    spacing = (sx, sy, sz) en mm (ordre NIfTI x,y,z).
    """
    if binary_mask.sum() == 0:
        return None

    # Lissage gaussien pour une surface propre
    sm = gaussian_filter(binary_mask.astype(np.float32), sigma=sigma)
    if sm.max() < 0.3:
        return None

    try:
        verts, faces, _, _ = measure.marching_cubes(
            sm, level=0.4,
            spacing=spacing,
            step_size=step,
            allow_degenerate=False,
        )
    except Exception as e:
        print(f"[WARN] marching_cubes : {e}")
        return None

    if len(faces) == 0:
        return None

    # Format PyVista : [3, i0, i1, i2, ...]
    n_faces  = len(faces)
    pv_faces = np.hstack([
        np.full((n_faces, 1), 3, dtype=np.int64),
        faces.astype(np.int64)
    ]).flatten()

    mesh = pv.PolyData(verts, pv_faces)

    if smooth_iter > 0:
        mesh = mesh.smooth(n_iter=smooth_iter, relaxation_factor=0.15)

    mesh.compute_normals(inplace=True)
    print(f"[MESH] {mesh.n_points:,} pts, {mesh.n_cells:,} faces")
    return mesh


# ══════════════════════════════════════════════════════════════
#  4. STATISTIQUES TUMORALES
# ══════════════════════════════════════════════════════════════

def tumor_stats(volume, mask, spacing):
    sx, sy, sz = spacing
    voxel_vol  = sx * sy * sz          # mm³ par voxel
    n_vox      = int(mask.sum())
    if n_vox == 0:
        return None
    vol_cm3    = round(n_vox * voxel_vol / 1000, 2)
    # Diamètre équivalent d'une sphère de même volume
    diam_mm    = round(2 * ((3 * n_vox * voxel_vol) / (4 * np.pi)) ** (1/3), 1)
    hu         = volume[mask.astype(bool)]
    return {
        "n_vox":      n_vox,
        "volume_cm3": vol_cm3,
        "diam_mm":    diam_mm,
        "int_mean":   round(float(hu.mean()), 1),
        "int_min":    round(float(hu.min()),  1),
        "int_max":    round(float(hu.max()),  1),
    }


# ══════════════════════════════════════════════════════════════
#  5. AFFICHAGE 3D (un seul timepoint)
# ══════════════════════════════════════════════════════════════

def show_single(root, tp, seq, offscreen=False, output_png=None):
    print(f"\n{'─'*60}")
    print(f"  Timepoint : {tp.upper()}  |  Date : {TP_DATES[tp]}  |  {TP_DELTA[tp]}")
    print(f"  Séquence  : {SEQ_LABELS[seq]}")
    print(f"{'─'*60}")

    # ── Chargement volume IRM ────────────────────────────────
    vol_path = find_volume(root, tp, seq)
    if vol_path is None:
        print(f"[ERREUR] Fichier {seq}.nii.gz introuvable pour {tp}")
        sys.exit(1)
    volume, spacing = load_nii(vol_path)
    print(f"[OK] Volume {volume.shape}  spacing {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm")

    # ── Masque tumeur ────────────────────────────────────────
    mask_path = find_mask(root, tp)
    mask = None
    if mask_path:
        mask_vol, _ = load_nii(mask_path)
        mask = (mask_vol > 0.5).astype(np.uint8)
        print(f"[OK] Masque tumeur : {mask.sum():,} voxels")
    else:
        print(f"[WARN] Masque tumeur introuvable pour {tp}")

    # ── Masque cerveau ───────────────────────────────────────
    brain_path = find_brain_mask(root)
    if brain_path:
        brain_vol, _ = load_nii(brain_path)
        brain_mask = (brain_vol > 0.5).astype(np.uint8)
        print(f"[OK] Masque cerveau fourni : {brain_mask.sum():,} voxels")
    else:
        brain_mask = segment_brain(volume)

    # ── Construction des maillages ───────────────────────────
    print("\n[MESH] Construction surface cerveau...")
    brain_pd  = mask_to_mesh(brain_mask, spacing, sigma=2.5, step=2, smooth_iter=100)

    tumor_pd = None
    stats    = None
    if mask is not None and mask.any():
        print("[MESH] Construction surface tumeur...")
        tumor_pd = mask_to_mesh(mask, spacing, sigma=1.0, step=1, smooth_iter=40)
        stats    = tumor_stats(volume, mask, spacing)

    # ── Rendu PyVista ────────────────────────────────────────
    _render(
        brain_pd, tumor_pd, stats,
        tp=tp, seq=seq,
        offscreen=offscreen, output_png=output_png,
    )


# ══════════════════════════════════════════════════════════════
#  6. AFFICHAGE 3D (tous les timepoints côte à côte)
# ══════════════════════════════════════════════════════════════

def show_all(root, seq, offscreen=False, output_png=None):
    """Affiche les 5 timepoints dans une grille 1×5 pour comparer l'évolution."""
    available = [tp for tp in TIMEPOINTS if find_volume(root, tp, seq) is not None]
    n = len(available)
    if n == 0:
        print("[ERREUR] Aucun volume trouvé.")
        sys.exit(1)

    print(f"\n[INFO] Mode comparaison — {n} timepoints  |  séquence {SEQ_LABELS[seq]}")

    pv.set_plot_theme("dark")
    cols = min(n, 5)
    pl = pv.Plotter(
        shape=(1, cols),
        title=f"Évolution tumorale P01 — {SEQ_LABELS[seq]}",
        window_size=(300 * cols, 520),
        off_screen=offscreen,
    )
    pl.set_background("#04050e")

    for col, tp in enumerate(available):
        pl.subplot(0, col)

        # Chargement
        vol_path  = find_volume(root, tp, seq)
        volume, spacing = load_nii(vol_path)
        mask_path = find_mask(root, tp)

        brain_path = find_brain_mask(root)
        if brain_path:
            bv, _ = load_nii(brain_path)
            brain_mask = (bv > 0.5).astype(np.uint8)
        else:
            brain_mask = segment_brain(volume)

        brain_pd = mask_to_mesh(brain_mask, spacing, sigma=2.5, step=2, smooth_iter=80)

        tumor_pd = None
        stats    = None
        if mask_path:
            mv, _ = load_nii(mask_path)
            mask  = (mv > 0.5).astype(np.uint8)
            if mask.any():
                tumor_pd = mask_to_mesh(mask, spacing, sigma=1.0, step=1, smooth_iter=30)
                stats    = tumor_stats(volume, mask, spacing)

        # Cerveau
        if brain_pd:
            pl.add_mesh(brain_pd, color="#b0c8e8", opacity=0.15,
                        smooth_shading=True, specular=0.5, diffuse=0.8)

        # Tumeur
        if tumor_pd:
            pl.add_mesh(tumor_pd, color="#ff2200", opacity=1.0,
                        smooth_shading=True, specular=1.0,
                        specular_power=80, ambient=0.3)
            pl.add_mesh(tumor_pd.copy(), color="#ff6600", opacity=0.25,
                        smooth_shading=True)  # halo chaud

        # Titre sous-fenêtre
        vol_str = f"{stats['volume_cm3']} cm³" if stats else "—"
        pl.add_text(
            f"{tp.upper()}\n{TP_DATES[tp]}\n{TP_DELTA[tp]}\nVol: {vol_str}",
            font_size=8, color="white", position="upper_left", font="courier",
        )

        # Lumières
        pl.remove_all_lights()
        pl.add_light(pv.Light(position=(300, 400, 300), color="white",    intensity=0.9))
        pl.add_light(pv.Light(position=(-200, -200, 200), color="#3355aa", intensity=0.5))

        pl.reset_camera()
        pl.camera.elevation = 10
        pl.camera.azimuth   = 20

    pl.add_text(
        "Mode comparaison — évolution tumorale sur 10 mois",
        position="lower_edge", font_size=8, color="#334466",
    )

    if offscreen and output_png:
        pl.screenshot(output_png)
        print(f"[OK] Screenshot : {output_png}")
    else:
        pl.show()


# ══════════════════════════════════════════════════════════════
#  7. RENDU PRINCIPAL (un timepoint)
# ══════════════════════════════════════════════════════════════

def _render(brain_pd, tumor_pd, stats, tp, seq, offscreen=False, output_png=None):
    pv.set_plot_theme("dark")

    pl = pv.Plotter(
        title=f"TumorFlow 3D — P01 | {tp.upper()} | {SEQ_LABELS[seq]}",
        window_size=(1400, 900),
        off_screen=offscreen,
    )
    pl.set_background("#04050e")

    # ── Cerveau — translucide bleu-gris ─────────────────────
    if brain_pd:
        pl.add_mesh(
            brain_pd,
            color="#c0d8f0",
            opacity=0.14,
            smooth_shading=True,
            specular=0.7,
            specular_power=40,
            diffuse=0.85,
            ambient=0.08,
            name="brain",
        )
        # Rim light effect (second pass wireframe très fin)
        pl.add_mesh(
            brain_pd,
            color="#4488cc",
            opacity=0.06,
            style="wireframe",
            line_width=0.4,
            name="brain_wf",
        )

    # ── Tumeur — rouge vif + halo orange ────────────────────
    if tumor_pd:
        # Corps principal
        pl.add_mesh(
            tumor_pd,
            color="#ff1500",
            opacity=1.0,
            smooth_shading=True,
            specular=1.0,
            specular_power=120,
            diffuse=0.75,
            ambient=0.35,
            name="tumor",
        )
        # Halo glow : copie légèrement dilatée, orange semi-transparent
        glow = tumor_pd.copy()
        pl.add_mesh(
            glow,
            color="#ff6600",
            opacity=0.22,
            smooth_shading=True,
            specular=0.3,
            name="tumor_glow",
        )

    # ── Lumières ─────────────────────────────────────────────
    pl.remove_all_lights()
    pl.add_light(pv.Light(position=( 400,  500,  400), color="white",   intensity=0.85))
    pl.add_light(pv.Light(position=(-300, -200, -200), color="#3355aa", intensity=0.55))
    pl.add_light(pv.Light(position=(   0, -400,  300), color="white",   intensity=0.35))
    pl.add_light(pv.Light(position=(   0,    0, -500), color="#112266", intensity=0.45))

    # Pas de point light colorée sur la tumeur — évite le reflet rouge sur le cerveau

    # ── Texte informatif ─────────────────────────────────────
    header = (
        f"TumorFlow 3D — Patient P01\n"
        f"Séquence : {SEQ_LABELS[seq]}\n"
        f"Timepoint : {tp.upper()}  ({TP_DATES[tp]})\n"
        f"{TP_DELTA[tp]}"
    )
    pl.add_text(header, position="upper_left", font_size=11,
                color="white", font="courier")

    if stats:
        info = (
            f"Tumeur\n"
            f"  Ø  {stats['diam_mm']} mm\n"
            f"  Vol {stats['volume_cm3']} cm³\n"
            f"  {stats['n_vox']:,} voxels\n"
            f"  Int. moy. {stats['int_mean']}\n"
            f"  Int. {stats['int_min']} → {stats['int_max']}"
        )
        pl.add_text(info, position="upper_right", font_size=11,
                    color="#ff6622", font="courier")
    else:
        pl.add_text("Aucun masque tumeur", position="upper_right",
                    font_size=11, color="#888888")

    # Légende bas de page
    pl.add_text(
        "Clic + glisser : rotation   |   Molette : zoom   |   Shift+clic : déplacer   |   Q : quitter",
        position="lower_edge", font_size=9, color="#334466",
    )

    # ── Vue initiale ─────────────────────────────────────────
    pl.reset_camera()
    pl.camera.elevation = 12
    pl.camera.azimuth   = 25

    # ── Rendu ────────────────────────────────────────────────
    if offscreen and output_png:
        pl.screenshot(output_png, transparent_background=False)
        print(f"[OK] Screenshot sauvegardé : {output_png}")
    else:
        print("\n[Fenêtre 3D ouverte — fermer pour quitter]\n")
        pl.show()


# ══════════════════════════════════════════════════════════════
#  8. POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TumorFlow 3D — Viewer cérébral IRM + tumeur (PyVista)"
    )
    parser.add_argument("folder",
        help="Dossier patient (ex: P01/)")
    parser.add_argument("--tp", default="baseline",
        choices=TIMEPOINTS,
        help="Timepoint à afficher (défaut: baseline)")
    parser.add_argument("--seq", default="t1c",
        choices=SEQUENCES,
        help="Séquence IRM (défaut: t1c — montre le mieux la tumeur active)")
    parser.add_argument("--all", action="store_true",
        help="Afficher tous les timepoints côte à côte (évolution)")
    parser.add_argument("--offscreen", action="store_true",
        help="Rendu sans fenêtre (export PNG)")
    parser.add_argument("--output", default="brain3d.png",
        help="Nom du fichier PNG de sortie (avec --offscreen)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"[ERREUR] Dossier introuvable : {args.folder}")
        sys.exit(1)

    if args.all:
        show_all(args.folder, args.seq,
                 offscreen=args.offscreen, output_png=args.output)
    else:
        show_single(args.folder, args.tp, args.seq,
                    offscreen=args.offscreen, output_png=args.output)


if __name__ == "__main__":
    main()
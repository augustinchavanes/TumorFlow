"""
animate_prediction.py — Animation 3D de l'évolution tumorale simulée (série 4D)
================================================================================

Lit un NIfTI 4D produit par random_forest_pipeline.py (--predict) :
  - prediction/<PID>/prediction/<PID>_u_4D.nii.gz      (densité continue 0–1)
  - prediction/<PID>/prediction/<PID>_mask_4D.nii.gz   (masque binaire 0/1)

et affiche une iso-surface 3D animée image par image (PyVista).

Usage :
  # Fenêtre interactive
  python visualisation/animate_prediction.py prediction/P03/prediction/P03_u_4D.nii.gz

  # Seuil personnalisé + boucle de lecture
  python visualisation/animate_prediction.py prediction/P03/prediction/P03_u_4D.nii.gz --threshold 0.5 --loops 3

  # Export d'une animation (gif ou mp4)
  python visualisation/animate_prediction.py prediction/P03/prediction/P03_u_4D.nii.gz --output evolution.gif --fps 5

  # Avec libellés temporels issus du CSV de frames
  python visualisation/animate_prediction.py prediction/P03/prediction/P03_u_4D.nii.gz \
      --timepoints prediction/P03/prediction/P03_4D_timepoints.csv

Fenêtre interactive (sans --output) :
  - par défaut : un CURSEUR temporel permet de parcourir l'évolution image par image,
    la souris pivote/zoome la scène (comme brain3d_viewer) ;
  - --autoplay : lecture automatique en boucle.

Le cerveau du patient (brain_mask) est auto-détecté dans le dossier patient et
affiché en surface translucide de contexte. Options :
  --brain <chemin>      forcer un masque cerveau précis
  --brain_opacity 0.12  régler la transparence
  --no_brain            désactiver l'affichage du cerveau
================================================================================
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pyvista as pv
from skimage import measure
from scipy.ndimage import gaussian_filter


def field_to_mesh(field, spacing, level, sigma=1.0, step=1, smooth_iter=30):
    """Iso-surface (marching cubes) d'un champ 3D. None si vide."""
    sm = gaussian_filter(field.astype(np.float32), sigma=sigma)
    if float(sm.max()) < level:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(
            sm, level=level, spacing=spacing, step_size=step,
            allow_degenerate=False,
        )
    except Exception as e:
        print(f"[WARN] marching_cubes : {e}")
        return None
    if len(faces) == 0:
        return None
    pv_faces = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)]
    ).flatten()
    mesh = pv.PolyData(verts, pv_faces)
    if smooth_iter > 0:
        mesh = mesh.smooth(n_iter=smooth_iter, relaxation_factor=0.15)
    return mesh


def load_4d(path):
    """Charge un NIfTI, retourne (data4d [X,Y,Z,T], spacing). Promeut 3D -> 4D(T=1)."""
    img = nib.load(str(path))
    data = img.get_fdata()
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    if data.ndim == 3:
        data = data[..., None]
    elif data.ndim != 4:
        print(f"[ERREUR] NIfTI {data.ndim}D non supporté (3D ou 4D attendu).")
        sys.exit(1)
    return data, spacing


def load_days(timepoints_csv, n_frames):
    """Libellés temporels (jours) depuis le CSV de frames, sinon indices."""
    if timepoints_csv and Path(timepoints_csv).exists():
        try:
            import pandas as pd
            df = pd.read_csv(timepoints_csv)
            if "days" in df.columns:
                return [f"{d:.0f} j" for d in df["days"].tolist()]
        except Exception as e:
            print(f"[WARN] Lecture timepoints : {e}")
    return [f"frame {i}" for i in range(n_frames)]


def find_brain_mask(nifti_4d_path: Path, explicit: str = None):
    """
    Localise le masque cerveau du patient.
    Cherche dans le dossier du patient (parent et grand-parent du NIfTI 4D),
    par ex. prediction/<PID>/<PID>_brain_mask.nii.gz.
    """
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    candidates_dirs = [nifti_4d_path.parent, nifti_4d_path.parent.parent]
    for d in candidates_dirs:
        if not d.exists():
            continue
        for pat in ("*brain_mask*.nii.gz", "*brain_mask*.nii"):
            hits = sorted(d.glob(pat))
            if hits:
                return hits[0]
    return None


def build_brain_mesh(brain_path: Path, ref_shape: tuple, spacing):
    """Maillage translucide du cerveau (iso-surface du masque). None si incompatible."""
    try:
        bm = nib.load(str(brain_path)).get_fdata()
    except Exception as e:
        print(f"[WARN] brain_mask illisible : {e}")
        return None
    if bm.shape != tuple(ref_shape):
        print(f"[WARN] brain_mask {bm.shape} ≠ grille {tuple(ref_shape)} — cerveau ignoré.")
        return None
    mask = (bm > 0.5).astype(np.float32)
    if mask.sum() == 0:
        return None
    # step plus grossier + lissage fort : surface légère pour servir de contexte
    return field_to_mesh(mask, spacing, level=0.5, sigma=2.5, step=3, smooth_iter=80)


def main():
    ap = argparse.ArgumentParser(
        description="Animation 3D d'une série 4D de simulation tumorale.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("nifti_4d", help="Chemin du NIfTI 4D (u_4D ou mask_4D)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Niveau d'iso-surface (densité). Ignoré si masque binaire.")
    ap.add_argument("--timepoints", default=None,
                    help="CSV <PID>_4D_timepoints.csv pour les libellés en jours")
    ap.add_argument("--output", default=None,
                    help="Fichier d'export (.gif ou .mp4). Si absent : fenêtre interactive")
    ap.add_argument("--fps", type=int, default=5, help="Images par seconde")
    ap.add_argument("--autoplay", action="store_true",
                    help="Fenêtre : lecture automatique en boucle (sinon curseur temporel)")
    ap.add_argument("--loops", type=int, default=2,
                    help="Nombre de lectures en boucle (mode --autoplay)")
    ap.add_argument("--color", default="#ff1500", help="Couleur de la tumeur")
    ap.add_argument("--brain", default=None,
                    help="Masque cerveau à afficher en contexte (auto-détecté si absent)")
    ap.add_argument("--no_brain", action="store_true",
                    help="Ne pas afficher le cerveau")
    ap.add_argument("--brain_opacity", type=float, default=0.14,
                    help="Opacité de la surface cérébrale (0 = invisible, 1 = opaque)")
    args = ap.parse_args()

    path = Path(args.nifti_4d)
    if not path.exists():
        print(f"[ERREUR] Fichier introuvable : {path}")
        sys.exit(1)

    data, spacing = load_4d(path)
    n_frames = data.shape[-1]
    is_binary = set(np.unique(data)).issubset({0.0, 1.0})
    level = 0.5 if is_binary else args.threshold
    labels = load_days(args.timepoints, n_frames)

    print(f"[OK] {path.name} | {data.shape[:3]} | {n_frames} frames | "
          f"spacing {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm | "
          f"{'binaire' if is_binary else 'densité'} (level={level})")

    # Pré-calcul des maillages (topologie variable -> on les stocke)
    meshes = []
    for t in range(n_frames):
        m = field_to_mesh(data[..., t], spacing, level)
        meshes.append(m)
        print(f"  frame {t+1}/{n_frames} — "
              f"{'vide' if m is None else f'{m.n_points:,} pts'}")

    # Cerveau du patient (contexte anatomique), calculé une seule fois
    brain_mesh = None
    if not args.no_brain:
        brain_path = find_brain_mask(path, args.brain)
        if brain_path is not None:
            print(f"[OK] Cerveau : {brain_path.name}")
            brain_mesh = build_brain_mesh(brain_path, data.shape[:3], spacing)
        else:
            print("[INFO] Aucun brain_mask trouvé — animation sans cerveau "
                  "(utilise --brain pour le préciser).")

    # Bornes globales pour figer la caméra (tumeur + cerveau)
    all_bounds = [m.bounds for m in meshes if m is not None]
    if brain_mesh is not None:
        all_bounds.append(brain_mesh.bounds)
    if not all_bounds:
        print("[ERREUR] Aucune iso-surface non vide (seuil trop haut ?).")
        sys.exit(1)
    ab = np.array(all_bounds)
    global_bounds = [ab[:, 0].min(), ab[:, 1].max(),
                     ab[:, 2].min(), ab[:, 3].max(),
                     ab[:, 4].min(), ab[:, 5].max()]

    off = args.output is not None
    pv.set_plot_theme("dark")
    plotter = pv.Plotter(
        title=f"TumorFlow 3D — {path.stem}",
        window_size=(1400, 900),
        off_screen=off,
    )
    plotter.set_background("#04050e")

    # Surface cérébrale translucide + wireframe (esthétique brain3d_viewer)
    if brain_mesh is not None:
        plotter.add_mesh(brain_mesh, color="#c0d8f0", opacity=args.brain_opacity,
                         smooth_shading=True, specular=0.7, specular_power=40,
                         diffuse=0.85, ambient=0.08, name="brain")
        plotter.add_mesh(brain_mesh, color="#4488cc", opacity=0.06,
                         style="wireframe", line_width=0.4, name="brain_wf")

    # Pré-création de TOUS les actors une seule fois (tumeur + halo par frame),
    # masqués au départ. On ne fera ensuite que basculer leur visibilité : plus
    # aucune recompilation de shaders par frame -> rendu fluide.
    tumor_actors, glow_actors = [], []
    for i, mesh in enumerate(meshes):
        if mesh is None:
            tumor_actors.append(None)
            glow_actors.append(None)
            continue
        a = plotter.add_mesh(
            mesh, color=args.color, opacity=1.0,
            smooth_shading=True, specular=1.0, specular_power=120,
            diffuse=0.75, ambient=0.35, name=f"tumor_{i}")
        g = plotter.add_mesh(
            mesh.copy(), color="#ff6600", opacity=0.22,
            smooth_shading=True, specular=0.3, name=f"glow_{i}")
        a.visibility = False
        g.visibility = False
        tumor_actors.append(a)
        glow_actors.append(g)

    def show_frame(idx, actor_holder=None):
        for a, g in zip(tumor_actors, glow_actors):
            if a is not None:
                a.visibility = False
                g.visibility = False
        if tumor_actors[idx] is not None:
            tumor_actors[idx].visibility = True
            glow_actors[idx].visibility = True
        plotter.add_text(
            f"TumorFlow 3D — {path.stem}\n{labels[idx]}",
            position="upper_left", font_size=11, color="white", font="courier",
            name="label")

    holder = None

    # Éclairage esthétique brain3d_viewer, limité à 2 sources : au-delà, le
    # contexte OpenGL off-screen (export GIF/MP4) sature ses unités de texture
    # ("Hardware does not support the number of textures defined") et la
    # compilation des shaders échoue en boucle.
    plotter.remove_all_lights()
    plotter.add_light(pv.Light(position=( 400,  500,  400), color="white",   intensity=0.95))
    plotter.add_light(pv.Light(position=(-300, -200, -200), color="#3355aa", intensity=0.55))

    plotter.reset_camera(bounds=global_bounds)
    plotter.camera.elevation = 12
    plotter.camera.azimuth   = 25

    if off:
        if args.output.lower().endswith(".gif"):
            plotter.open_gif(args.output, fps=args.fps)
        else:
            plotter.open_movie(args.output, framerate=args.fps)
        plotter.show(auto_close=False)
        for idx in range(n_frames):
            show_frame(idx, holder)
            plotter.render()
            plotter.write_frame()
        plotter.close()
        print(f"[OK] Animation exportée → {args.output}")
    elif args.autoplay:
        # Lecture automatique en boucle (fenêtre interactive, rotation possible)
        plotter.show(interactive_update=True, auto_close=False)
        for _ in range(max(1, args.loops)):
            for idx in range(n_frames):
                show_frame(idx, holder)
                plotter.update()
                time.sleep(1.0 / max(1, args.fps))
        plotter.show()
    else:
        # Visualiseur interactif : curseur temporel + rotation/zoom souris
        show_frame(0, holder)

        def _on_slider(value):
            show_frame(int(round(value)), holder)

        plotter.add_slider_widget(
            _on_slider, [0, max(0, n_frames - 1)], value=0,
            title="Instant (frame)", fmt="%.0f",
            pointa=(0.25, 0.90), pointb=(0.75, 0.90),
            style="modern",
        )
        plotter.add_text(
            "Curseur : temps   |   Clic + glisser : rotation   |   Molette : zoom   |   Q : quitter",
            position="lower_edge", font_size=9, color="#334466", font="courier", name="hint",
        )
        print("[INFO] Fenêtre interactive ouverte — déplacez le curseur pour parcourir "
              "l'évolution, la souris pour pivoter/zoomer.")
        plotter.show()


if __name__ == "__main__":
    main()

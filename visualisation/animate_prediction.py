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

  # Rendu VOLUMIQUE (opacité ∝ densité) : suit fidèlement la courbe de volume
  # (grossit / dégrossit / repousse), y compris le halo diffus
  python visualisation/animate_prediction.py prediction/P03/prediction/P03_u_4D.nii.gz \
      --volume --timepoints prediction/P03/prediction/P03_4D_timepoints.csv

Modes de représentation (fichier densité u_4D) :
  - défaut         : iso-surface à seuil FIXE (--threshold) ; montre la régression réelle
  - --relative     : iso-surface à seuil adaptatif (toujours visible tant que densité > 0)
  - --volume       : rendu volumique (opacité ∝ densité) ; le plus proche de la courbe

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


def field_to_mesh(field, spacing, level=None, rel_frac=None,
                  sigma=1.0, step=1, smooth_iter=30):
    """
    Iso-surface (marching cubes) d'un champ 3D. None si vide.

    level     : niveau d'iso-surface absolu.
    rel_frac  : si fourni, le niveau est calculé relativement au pic du champ
                lissé (level = rel_frac × max). Tant qu'il reste de la densité
                (max > 0), une surface existe -> la tumeur reste visible.
    """
    sm = gaussian_filter(field.astype(np.float32), sigma=sigma)
    smax = float(sm.max())
    if rel_frac is not None:
        if smax <= 1e-9:
            return None
        level = max(smax * rel_frac, 1e-6)
    if level is None or smax < level:
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
    # float32 plutôt que float64 par défaut -> mémoire divisée par 2 (4D volumineux)
    data = np.asarray(img.dataobj, dtype=np.float32)
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


def _frame_grid(arr3d, spacing):
    """Enveloppe un champ 3D dans une grille VTK pour le rendu volumique."""
    grid = pv.ImageData()
    grid.dimensions = arr3d.shape
    grid.spacing = spacing
    grid.origin = (0.0, 0.0, 0.0)
    grid.point_data["u"] = np.ascontiguousarray(arr3d).flatten(order="F")
    return grid


def render_volume(args, path, data, spacing, labels, n_frames):
    """
    Rendu VOLUMIQUE animé : l'opacité de chaque voxel est proportionnelle à sa
    densité. Contrairement à l'iso-surface (tout-ou-rien à un seuil), on voit
    toute la densité -> l'animation suit fidèlement la courbe de volume
    (grossit / dégrossit / repousse), y compris le halo diffus de faible densité.
    """
    # Fonction de transfert d'opacité : 0 = transparent (air), montée douce avec u.
    base_tf = np.array([0.0, 0.12, 0.28, 0.5, 0.75, 1.0], dtype=float)
    opacity_tf = list(np.clip(base_tf * max(0.05, args.vol_gain), 0.0, 1.0))
    clim = [0.0, max(1e-3, args.vol_clim)]

    # Cerveau de contexte (iso-surface translucide, comme en mode surfaces)
    brain_mesh = None
    if not args.no_brain:
        brain_path = find_brain_mask(path, args.brain)
        if brain_path is not None:
            print(f"[OK] Cerveau : {brain_path.name}")
            brain_mesh = build_brain_mesh(brain_path, data.shape[:3], spacing)
        else:
            print("[INFO] Aucun brain_mask trouvé — rendu sans cerveau.")

    off = args.output is not None
    pv.set_plot_theme("dark")
    plotter = pv.Plotter(
        title=f"TumorFlow 3D (volume) — {path.stem}",
        window_size=(1400, 900), off_screen=off,
    )
    plotter.set_background("#04050e")

    if brain_mesh is not None:
        plotter.add_mesh(brain_mesh, color="#c0d8f0", opacity=args.brain_opacity,
                         smooth_shading=True, specular=0.7, specular_power=40,
                         diffuse=0.85, ambient=0.08, name="brain")
        plotter.add_mesh(brain_mesh, color="#4488cc", opacity=0.06,
                         style="wireframe", line_width=0.4, name="brain_wf")

    # Bornes globales (toute la grille) pour figer la caméra
    nx, ny, nz = data.shape[:3]
    global_bounds = [0, nx * spacing[0], 0, ny * spacing[1], 0, nz * spacing[2]]

    def show_frame(idx):
        grid = _frame_grid(data[..., idx], spacing)
        plotter.add_volume(
            grid, scalars="u", cmap=args.vol_cmap, clim=clim,
            opacity=opacity_tf, shade=True, name="vol",
        )
        plotter.add_text(
            f"TumorFlow 3D — {path.stem}\n{labels[idx]}",
            position="upper_left", font_size=11, color="white", font="courier",
            name="label")

    plotter.reset_camera(bounds=global_bounds)
    plotter.camera.elevation = args.elevation
    plotter.camera.azimuth   = args.azimuth
    if args.roll:
        plotter.camera.roll += args.roll

    if off:
        if args.output.lower().endswith(".gif"):
            plotter.open_gif(args.output, fps=args.fps)
        else:
            plotter.open_movie(args.output, framerate=args.fps)
        plotter.show(auto_close=False)
        for idx in range(n_frames):
            show_frame(idx)
            if args.spin:
                plotter.camera.azimuth += 360.0 / max(1, n_frames)
            plotter.render()
            plotter.write_frame()
            print(f"  écrit {idx+1}/{n_frames}", flush=True)
        plotter.close()
        print(f"[OK] Animation volumique exportee -> {args.output}")
        return

    # Interactif : curseur temporel
    show_frame(0)

    def _on_slider(value):
        show_frame(int(round(value)))

    plotter.add_slider_widget(
        _on_slider, [0, max(0, n_frames - 1)], value=0,
        title="Instant (frame)", fmt="%.0f",
        pointa=(0.25, 0.90), pointb=(0.75, 0.90), style="modern",
    )
    plotter.add_text(
        "Curseur : temps   |   Clic + glisser : rotation   |   Molette : zoom   |   Q : quitter",
        position="lower_edge", font_size=9, color="#334466", font="courier", name="hint",
    )
    print("[INFO] Fenêtre interactive (rendu volumique) — curseur = temps, souris = pivoter/zoomer.")
    plotter.show()


def main():
    ap = argparse.ArgumentParser(
        description="Animation 3D d'une série 4D de simulation tumorale.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("nifti_4d", help="Chemin du NIfTI 4D (u_4D ou mask_4D)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Niveau d'iso-surface (densité) FIXE. Ignoré si masque binaire ou --relative.")
    ap.add_argument("--relative", action="store_true",
                    help="Seuil ADAPTATIF par frame : la tumeur reste visible tant "
                         "que la densité est non nulle (recommandé pour suivre la régression).")
    ap.add_argument("--rel_frac", type=float, default=0.2,
                    help="Fraction du pic de densité de chaque frame utilisée comme niveau "
                         "d'iso-surface en mode --relative (0.2 = 20%% du max de l'image).")
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
    ap.add_argument("--volume", action="store_true",
                    help="Rendu VOLUMIQUE (opacité ∝ densité) au lieu d'une iso-surface : "
                         "montre en 3D toute la dynamique de la courbe (grossit/dégrossit/repousse).")
    ap.add_argument("--vol_cmap", default="inferno",
                    help="Colormap du rendu volumique (ex. inferno, hot, magma, viridis).")
    ap.add_argument("--vol_clim", type=float, default=1.0,
                    help="Densité correspondant à l'opacité max (baisser à ~0.3 pour mieux "
                         "voir les faibles densités résiduelles).")
    ap.add_argument("--vol_gain", type=float, default=1.0,
                    help="Gain global d'opacité du rendu volumique (>1 = plus dense/visible).")
    ap.add_argument("--brain", default=None,
                    help="Masque cerveau à afficher en contexte (auto-détecté si absent)")
    ap.add_argument("--no_brain", action="store_true",
                    help="Ne pas afficher le cerveau")
    ap.add_argument("--brain_opacity", type=float, default=0.14,
                    help="Opacité de la surface cérébrale (0 = invisible, 1 = opaque)")
    ap.add_argument("--azimuth", type=float, default=25.0,
                    help="Angle de vue horizontal (rotation gauche/droite autour du cerveau).")
    ap.add_argument("--elevation", type=float, default=12.0,
                    help="Angle de vue vertical (rotation haut/bas).")
    ap.add_argument("--roll", type=float, default=0.0,
                    help="Rotation de la caméra autour de son axe (inclinaison).")
    ap.add_argument("--spin", action="store_true",
                    help="Export : fait tourner la caméra de 360° pendant l'animation.")
    args = ap.parse_args()

    path = Path(args.nifti_4d)
    if not path.exists():
        print(f"[ERREUR] Fichier introuvable : {path}")
        sys.exit(1)

    data, spacing = load_4d(path)
    n_frames = data.shape[-1]
    is_binary = set(np.unique(data)).issubset({0.0, 1.0})
    use_relative = args.relative and not is_binary
    level = 0.5 if is_binary else args.threshold
    labels = load_days(args.timepoints, n_frames)

    use_volume = args.volume and not is_binary
    if is_binary:
        mode_str = "binaire (level=0.5)"
    elif use_volume:
        mode_str = (f"VOLUMIQUE (cmap={args.vol_cmap}, clim=[0,{args.vol_clim}], "
                    f"gain={args.vol_gain})")
    elif use_relative:
        mode_str = f"densité — seuil RELATIF ({args.rel_frac:.0%} du pic par frame)"
    else:
        mode_str = f"densité — seuil FIXE (level={level})"
    print(f"[OK] {path.name} | {data.shape[:3]} | {n_frames} frames | "
          f"spacing {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm | {mode_str}")

    if use_volume:
        render_volume(args, path, data, spacing, labels, n_frames)
        return

    # Pré-calcul des maillages (topologie variable -> on les stocke)
    meshes = []
    for t in range(n_frames):
        fmax = float(data[..., t].max())
        if is_binary:
            m = field_to_mesh(data[..., t], spacing, level=0.5)
        elif use_relative:
            m = field_to_mesh(data[..., t], spacing, rel_frac=args.rel_frac)
        else:
            m = field_to_mesh(data[..., t], spacing, level=level)
        meshes.append(m)
        extra = f" | pic={fmax:.3f}" if not is_binary else ""
        print(f"  frame {t+1}/{n_frames} — "
              f"{'vide' if m is None else f'{m.n_points:,} pts'}{extra}", flush=True)

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

    # Le tableau 4D (souvent ~1 Go) n'est plus utile une fois les maillages
    # construits : on le libère pour soulager la RAM avant le rendu.
    import gc
    del data
    gc.collect()

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

    # Éclairage esthétique brain3d_viewer, limité à 2 sources : au-delà, le
    # contexte OpenGL off-screen (export GIF/MP4) sature ses unités de texture
    # ("Hardware does not support the number of textures defined") et la
    # compilation des shaders échoue en boucle.
    plotter.remove_all_lights()
    plotter.add_light(pv.Light(position=( 400,  500,  400), color="white",   intensity=0.95))
    plotter.add_light(pv.Light(position=(-300, -200, -200), color="#3355aa", intensity=0.55))

    plotter.reset_camera(bounds=global_bounds)
    plotter.camera.elevation = args.elevation
    plotter.camera.azimuth   = args.azimuth
    if args.roll:
        plotter.camera.roll += args.roll

    def _add_label(idx):
        plotter.add_text(
            f"TumorFlow 3D — {path.stem}\n{labels[idx]}",
            position="upper_left", font_size=11, color="white", font="courier",
            name="label")

    if off:
        # ── MODE EXPORT (faible empreinte mémoire) ────────────────────────────
        # On n'ajoute QUE la frame courante puis on la retire : indispensable sur
        # machine à RAM limitée (pré-charger 60+ actors sature la mémoire).
        holder = {"tumor": None, "glow": None}

        def show_frame(idx):
            for k in ("tumor", "glow"):
                if holder[k] is not None:
                    plotter.remove_actor(holder[k]); holder[k] = None
            mesh = meshes[idx]
            if mesh is not None:
                holder["tumor"] = plotter.add_mesh(
                    mesh, color=args.color, opacity=1.0,
                    smooth_shading=True, specular=1.0, specular_power=120,
                    diffuse=0.75, ambient=0.35, name="tumor")
                holder["glow"] = plotter.add_mesh(
                    mesh.copy(), color="#ff6600", opacity=0.22,
                    smooth_shading=True, specular=0.3, name="glow")
            _add_label(idx)

        if args.output.lower().endswith(".gif"):
            plotter.open_gif(args.output, fps=args.fps)
        else:
            plotter.open_movie(args.output, framerate=args.fps)
        plotter.show(auto_close=False)
        for idx in range(n_frames):
            show_frame(idx)
            if args.spin:
                plotter.camera.azimuth += 360.0 / max(1, n_frames)
            plotter.render()
            plotter.write_frame()
            print(f"  écrit {idx+1}/{n_frames}", flush=True)
        plotter.close()
        print(f"[OK] Animation exportee -> {args.output}")
        return

    # ── MODES INTERACTIFS (fluides) : pré-ajout de tous les actors + visibilité
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

    def show_frame(idx):
        for a, g in zip(tumor_actors, glow_actors):
            if a is not None:
                a.visibility = False
                g.visibility = False
        if tumor_actors[idx] is not None:
            tumor_actors[idx].visibility = True
            glow_actors[idx].visibility = True
        _add_label(idx)

    if args.autoplay:
        # Lecture automatique en boucle (fenêtre interactive, rotation possible)
        plotter.show(interactive_update=True, auto_close=False)
        for _ in range(max(1, args.loops)):
            for idx in range(n_frames):
                show_frame(idx)
                plotter.update()
                time.sleep(1.0 / max(1, args.fps))
        plotter.show()
    else:
        # Visualiseur interactif : curseur temporel + rotation/zoom souris
        show_frame(0)

        def _on_slider(value):
            show_frame(int(round(value)))

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

"""
dicom_viewer.py — Visionneuse CT simple (sans segmentation)
Usage : python dicom_viewer.py "chemin/vers/dossier_patient/"
Require : pip install pydicom numpy matplotlib
"""

import os, sys, warnings
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
#  1. CHARGEMENT CT
# ══════════════════════════════════════════════════════════════

def find_ct_files(root_folder):
    ct_files = []
    for root, _, files in os.walk(root_folder):
        for f in files:
            if not f.lower().endswith(".dcm"):
                continue
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                mod = getattr(ds, "Modality", "")
                sop = str(getattr(ds, "SOPClassUID", ""))
                # Exclure les SEG
                if "66.4" in sop or mod == "SEG":
                    continue
                if hasattr(ds, "pixel_array") or True:
                    ct_files.append(path)
            except Exception:
                pass
    if not ct_files:
        raise FileNotFoundError("Aucun fichier DICOM trouvé dans ce dossier.")
    print(f"[OK] {len(ct_files)} fichier(s) DICOM trouvé(s)")
    return ct_files


def load_volume(ct_files):
    slices = []
    for path in ct_files:
        try:
            ds = pydicom.dcmread(path)
            if hasattr(ds, "pixel_array"):
                slices.append(ds)
        except Exception:
            pass

    if not slices:
        raise ValueError("Impossible de lire les pixels des fichiers DICOM.")

    # Grouper par SeriesInstanceUID + shape pour éviter les mélanges
    from collections import defaultdict
    groups = defaultdict(list)
    for s in slices:
        try:
            shape = s.pixel_array.shape
        except Exception:
            continue
        series_uid = str(getattr(s, "SeriesInstanceUID", "unknown"))
        key = (series_uid, shape)
        groups[key].append(s)

    if not groups:
        raise ValueError("Aucune slice lisible.")

    # Choisir le groupe le plus grand (= série principale)
    best_key = max(groups, key=lambda k: len(groups[k]))
    slices = groups[best_key]
    series_uid, shape = best_key
    print(f"[OK] Série sélectionnée : {series_uid[:40]}...  ({len(slices)} coupes, {shape[0]}×{shape[1]}px)")
    if len(groups) > 1:
        print(f"[INFO] {len(groups)} série(s) détectée(s) au total — affichage de la plus grande.")

    # Tri par position Z puis InstanceNumber
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda s: int(s.InstanceNumber))
        except AttributeError:
            pass

    volume = np.stack([
        s.pixel_array.astype(np.float32)
        * float(getattr(s, "RescaleSlope", 1))
        + float(getattr(s, "RescaleIntercept", 0))
        for s in slices
    ])

    # Espacement
    try:
        ps = slices[0].PixelSpacing
        sy, sx = float(ps[0]), float(ps[1])
    except AttributeError:
        sy, sx = 1.0, 1.0
    try:
        sz = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
    except (AttributeError, IndexError):
        sz = float(getattr(slices[0], "SliceThickness", 1.0))

    # Infos patient
    info = {
        "patient": getattr(slices[0], "PatientName", "Inconnu"),
        "date":    getattr(slices[0], "StudyDate", "—"),
        "desc":    getattr(slices[0], "StudyDescription", "—"),
        "modality":getattr(slices[0], "Modality", "—"),
        "spacing": (sz, sy, sx),
        "shape":   volume.shape,
    }

    print(f"[OK] Volume : {volume.shape}  |  spacing {sz:.2f}×{sy:.2f}×{sx:.2f} mm")
    return volume, info


# ══════════════════════════════════════════════════════════════
#  2. VIEWER
# ══════════════════════════════════════════════════════════════

WINDOWS = {
    "Poumon":      (-600, 1500),
    "Médiastin":   (40,   400),
    "Abdomen":     (60,   400),
    "Os":          (400,  1800),
    "Cerveau":     (40,   80),
    "Auto":        None,
}

def launch_viewer(volume, info):
    n = volume.shape[0]
    win_names  = list(WINDOWS.keys())
    current    = {"idx": n // 2, "win": 0}

    def get_display(idx):
        wname = win_names[current["win"]]
        params = WINDOWS[wname]
        if params is None:
            lo, hi = float(volume[idx].min()), float(volume[idx].max())
        else:
            wc, ww = params
            lo, hi = wc - ww / 2, wc + ww / 2
        return np.clip((volume[idx] - lo) / max(hi - lo, 1e-6), 0, 1)

    # ── Layout ──────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor("#0d0d0d")
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    gs.update(left=0.03, right=0.97, bottom=0.15, top=0.93, wspace=0.06)

    ax      = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])
    ax.set_facecolor("#0d0d0d")
    ax_info.set_facecolor("#111")
    ax.axis("off")
    ax_info.axis("off")

    im = ax.imshow(get_display(current["idx"]), cmap="gray",
                   interpolation="bilinear", zorder=0)
    title = ax.set_title("", color="white", fontsize=11, pad=8)

    # ── Panneau info ─────────────────────────────────────────
    sz, sy, sx = info["spacing"]
    nz, ny, nx = info["shape"]
    info_str = (
        f"Patient\n  {info['patient']}\n\n"
        f"Date\n  {info['date']}\n\n"
        f"Modalité\n  {info['modality']}\n\n"
        f"Description\n  {info['desc']}\n\n"
        f"Dimensions\n  {nx} × {ny} × {nz}\n\n"
        f"Espacement\n  {sx:.2f} × {sy:.2f} × {sz:.2f} mm"
    )
    ax_info.set_title("Informations", color="#aaa", fontsize=10, pad=6)
    ax_info.text(0.08, 0.95, info_str, transform=ax_info.transAxes,
                 color="white", fontsize=9, va="top", family="monospace", linespacing=1.6)

    def draw_frame():
        idx = current["idx"]
        im.set_data(get_display(idx))
        title.set_text(
            f"Coupe {idx+1}/{n}  |  Fenêtre : {win_names[current['win']]}"
        )
        fig.canvas.draw_idle()

    draw_frame()

    # ── Slider ───────────────────────────────────────────────
    ax_sl = plt.axes([0.04, 0.07, 0.62, 0.028], facecolor="#1e1e1e")
    slider = Slider(ax_sl, "Coupe", 0, n - 1, valinit=current["idx"], valstep=1, color="#2196F3")
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    def on_slider(val):
        current["idx"] = int(slider.val)
        draw_frame()
    slider.on_changed(on_slider)

    # ── Bouton fenêtrage ─────────────────────────────────────
    ax_bw = plt.axes([0.70, 0.06, 0.14, 0.045])
    btn_win = Button(ax_bw, f"⊞ {win_names[current['win']]}", color="#1a1a1a", hovercolor="#333")
    btn_win.label.set_color("#2196F3")

    def on_win(e):
        current["win"] = (current["win"] + 1) % len(win_names)
        btn_win.label.set_text(f"⊞ {win_names[current['win']]}")
        draw_frame()
    btn_win.on_clicked(on_win)

    # ── Bouton colormap ──────────────────────────────────────
    cmaps = ["gray", "bone", "hot", "inferno", "viridis"]
    cmap_idx = [0]
    ax_bc = plt.axes([0.86, 0.06, 0.10, 0.045])
    btn_cm = Button(ax_bc, "🎨 gray", color="#1a1a1a", hovercolor="#333")
    btn_cm.label.set_color("#aaa")

    def on_cmap(e):
        cmap_idx[0] = (cmap_idx[0] + 1) % len(cmaps)
        cm = cmaps[cmap_idx[0]]
        im.set_cmap(cm)
        btn_cm.label.set_text(f"🎨 {cm}")
        fig.canvas.draw_idle()
    btn_cm.on_clicked(on_cmap)

    # ── Molette & clavier ────────────────────────────────────
    def on_scroll(event):
        idx = current["idx"]
        new = min(idx + 1, n - 1) if event.button == "up" else max(idx - 1, 0)
        slider.set_val(new)

    def on_key(event):
        idx = current["idx"]
        if   event.key in ("right", "up"):   idx = min(idx + 1, n - 1)
        elif event.key in ("left", "down"):  idx = max(idx - 1, 0)
        elif event.key == "w":
            current["win"] = (current["win"] + 1) % len(win_names)
            btn_win.label.set_text(f"⊞ {win_names[current['win']]}")
        slider.set_val(idx)

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.suptitle("🫁  Visionneuse DICOM", color="white", fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01,
             "Molette / ← →  pour naviguer  |  W = changer la fenêtre",
             ha="center", color="#444", fontsize=8.5)
    plt.show()


# ══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        folder = input("Chemin vers le dossier DICOM : ").strip().strip('"')
    else:
        folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(f"[ERREUR] Dossier introuvable : {folder}")
        sys.exit(1)

    ct_files = find_ct_files(folder)
    volume, info = load_volume(ct_files)
    launch_viewer(volume, info)
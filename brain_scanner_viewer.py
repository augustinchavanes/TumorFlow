"""
nifti_viewer.py — Visionneuse multi-séquence IRM cérébrale avec superposition tumorale
Usage : python nifti_viewer.py "chemin/vers/dossier_patient/"

Structure attendue :
  P01/
    BraTS/
      baseline/  fu1/  fu2/  fu3/  fu4/
        fla.nii.gz  t1.nii.gz  t1c.nii.gz  t2.nii.gz
    tumor segmentation/
      P01_tumor_mask_baseline.nii/tumour_mask_corr.nii
      P01_tumor_mask_fu1.nii/tumour_mask.nii
      ...

Requires : pip install nibabel numpy matplotlib
"""

import os, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

try:
    import nibabel as nib
except ImportError:
    print("[ERREUR] nibabel manquant. Installez-le : pip install nibabel")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
#  1. CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════

TIMEPOINTS = ["baseline", "fu1", "fu2", "fu3", "fu4"]
SEQUENCES  = ["fla", "t1", "t1c", "t2"]
SEQ_LABELS = {"fla": "FLAIR", "t1": "T1", "t1c": "T1c", "t2": "T2"}

# Dates réelles des acquisitions (correspondance BraTS <-> DICOM)
TP_DATES = {
    "baseline": "07/01/2020",
    "fu1":      "14/02/2020",
    "fu2":      "20/05/2020",
    "fu3":      "10/08/2020",
    "fu4":      "10/11/2020",
}
TP_DELTA = {
    "baseline": "J0   (référence)",
    "fu1":      "J38  (+5 semaines)",
    "fu2":      "J134 (+4,5 mois)",
    "fu3":      "J216 (+7 mois)",
    "fu4":      "J308 (+10 mois)",
}

# Description clinique courte de chaque séquence
SEQ_DESC = {
    "fla": "Lésions péri-tumorales,\n  œdème, infiltration",
    "t1":  "Anatomie de référence,\n  substance blanche claire",
    "t1c": "T1 avec injection de contraste,\n  rehaussement tumoral actif",
    "t2":  "Œdème et liquides,\n  zones brillantes = eau",
}

# Description des axes
AXIS_DESC = [
    "Vue de dessus — coupe horizontale",
    "Vue de face  — coupe frontale",
    "Vue de côté  — coupe latérale",
]

TUMOR_FOLDERS = {
    "baseline": ["P01_tumor_mask_baseline.nii", "tumour_mask_corr.nii"],
    "fu1":      ["P01_tumor_mask_fu1.nii",      "tumour_mask.nii"],
    "fu2":      ["P01_tumor_mask_fu2.nii",       "tumour_mask.nii"],
    "fu3":      ["P01_tumor_mask_fu3.nii",       "tumour_mask.nii"],
    "fu4":      ["P01_tumor_mask_fu4.nii",       "tumour_mask.nii"],
}


def load_nii(path):
    """Charge un volume NIfTI et retourne un array float32 normalisé."""
    img = nib.load(path)
    vol = np.asarray(img.dataobj, dtype=np.float32)
    return vol


def find_data(root):
    """
    Parcourt l'arborescence et retourne :
      data[tp][seq]  = volume numpy (H×W×Z) ou None
      masks[tp]      = masque tumeur numpy (H×W×Z) ou None
    """
    data  = {tp: {sq: None for sq in SEQUENCES} for tp in TIMEPOINTS}
    masks = {tp: None for tp in TIMEPOINTS}

    brats_root = os.path.join(root, "BraTS")
    if not os.path.isdir(brats_root):
        # Essai sans sous-dossier BraTS
        brats_root = root

    for tp in TIMEPOINTS:
        tp_dir = os.path.join(brats_root, tp)
        if not os.path.isdir(tp_dir):
            continue
        for sq in SEQUENCES:
            for ext in [f"{sq}.nii.gz", f"{sq}.nii"]:
                path = os.path.join(tp_dir, ext)
                if os.path.isfile(path):
                    try:
                        data[tp][sq] = load_nii(path)
                        print(f"[OK] {tp}/{sq}  →  {data[tp][sq].shape}")
                    except Exception as e:
                        print(f"[WARN] {tp}/{sq} non chargé : {e}")
                    break

    # Masques tumoraux
    seg_root_candidates = [
        os.path.join(root, "tumor segmentation"),
        os.path.join(root, "tumour segmentation"),
        os.path.join(root, "segmentation"),
        root,
    ]
    seg_root = next((d for d in seg_root_candidates if os.path.isdir(d)), None)

    if seg_root:
        for tp, (folder, fname) in TUMOR_FOLDERS.items():
            path = os.path.join(seg_root, folder, fname)
            if os.path.isfile(path):
                try:
                    masks[tp] = load_nii(path)
                    print(f"[OK] Masque {tp}  →  {masks[tp].shape}")
                except Exception as e:
                    print(f"[WARN] Masque {tp} non chargé : {e}")
    else:
        print("[WARN] Dossier 'tumor segmentation' introuvable.")

    return data, masks


# ══════════════════════════════════════════════════════════════
#  2. UTILITAIRES D'AFFICHAGE
# ══════════════════════════════════════════════════════════════

WINDOWS = {
    "Auto":    None,
    "Cerveau": (0.4, 0.7),   # en fraction normalisée [0,1]
}

# Colormap rouge semi-transparente pour la tumeur
red_cmap = mcolors.LinearSegmentedColormap.from_list(
    "tumor_red",
    [(0, 0, 0, 0), (1, 0.1, 0.05, 0.75)],   # transparent → rouge vif
)


def normalize_slice(sl, window=None):
    lo = sl.min(); hi = sl.max()
    if hi == lo:
        return np.zeros_like(sl)
    s = (sl - lo) / (hi - lo)
    if window:
        wlo, whi = window
        s = np.clip((s - wlo) / max(whi - wlo, 1e-6), 0, 1)
    return s


def get_slice_2d(volume, idx, axis=2):
    """Extrait une coupe selon l'axe (0=axial Z, 1=coronal Y, 2=sagittal X)."""
    if volume is None:
        return None
    axis = min(axis, volume.ndim - 1)
    idx  = min(idx, volume.shape[axis] - 1)
    if axis == 0: return volume[idx, :, :]
    if axis == 1: return volume[:, idx, :]
    return volume[:, :, idx]


# ══════════════════════════════════════════════════════════════
#  3. VIEWER
# ══════════════════════════════════════════════════════════════

def launch_viewer(data, masks):
    # Lister les timepoints disponibles
    available_tp = [tp for tp in TIMEPOINTS if any(data[tp][sq] is not None for sq in SEQUENCES)]
    if not available_tp:
        print("[ERREUR] Aucune donnée chargée.")
        sys.exit(1)

    state = {
        "tp":      0,          # index dans available_tp
        "seq":     0,          # index dans SEQUENCES
        "idx":     0,          # index de coupe
        "axis":    2,          # 0=axial, 1=coronal, 2=sagittal
        "show_tumor": True,
        "window":  0,          # index dans WINDOWS
    }

    def current_volume():
        tp = available_tp[state["tp"]]
        sq = SEQUENCES[state["seq"]]
        return data[tp][sq]

    def current_mask():
        if not state["show_tumor"]:
            return None
        tp = available_tp[state["tp"]]
        return masks.get(tp)

    def n_slices():
        vol = current_volume()
        if vol is None: return 1
        ax = state["axis"]
        return vol.shape[min(ax, vol.ndim - 1)]

    def clamp_idx():
        state["idx"] = max(0, min(state["idx"], n_slices() - 1))

    # ── Layout ────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("#080c10")

    gs_main = gridspec.GridSpec(1, 2, width_ratios=[3, 1.2],
                                left=0.02, right=0.98,
                                bottom=0.18, top=0.90, wspace=0.04)
    ax_img  = fig.add_subplot(gs_main[0])
    ax_info = fig.add_subplot(gs_main[1])
    for ax in [ax_img, ax_info]:
        ax.axis("off")
    ax_img.set_facecolor("#080c10")
    ax_info.set_facecolor("#0f1318")

    # Placeholder image
    blank = np.zeros((240, 240), dtype=np.float32)
    im_gray  = ax_img.imshow(blank, cmap="gray", vmin=0, vmax=1,
                             interpolation="bilinear", aspect="equal")
    im_tumor = ax_img.imshow(blank, cmap=red_cmap, vmin=0, vmax=1,
                             interpolation="nearest", aspect="equal",
                             alpha=1.0)
    title_ax = ax_img.set_title("", color="white", fontsize=12, pad=10,
                                fontfamily="monospace")

    # Frise chronologique
    tp_dates_short = ["Jan 2020", "Fév 2020", "Mai 2020", "Août 2020", "Nov 2020"]
    for i, (tp, date_s) in enumerate(zip(TIMEPOINTS, tp_dates_short)):
        x = 0.12 + i * 0.175
        fig.text(x, 0.935, date_s, ha="center", color="#445566", fontsize=7,
                 fontfamily="monospace")
        col = "#3a6ea5" if tp in available_tp else "#333"
        fig.text(x, 0.922, tp.upper(), ha="center", color=col, fontsize=7.5,
                 fontfamily="monospace", fontweight="bold")
    timeline_marker = fig.text(0.12, 0.910, "▲", ha="center",
                               color="#e53935", fontsize=9)

    # Panneau d'info
    ax_info.set_title("── Infos patient ──", color="#7a8fa6", fontsize=9,
                      pad=8, fontfamily="monospace")
    info_text = ax_info.text(0.06, 0.97, "", transform=ax_info.transAxes,
                             color="#ccd", fontsize=8, va="top",
                             fontfamily="monospace", linespacing=1.75)

    def refresh():
        vol  = current_volume()
        mask = current_mask()
        idx  = state["idx"]
        ax   = state["axis"]

        sl   = get_slice_2d(vol,  idx, ax)
        msl  = get_slice_2d(mask, idx, ax)

        wname = list(WINDOWS.keys())[state["window"]]
        win   = WINDOWS[wname]

        if sl is not None:
            sl_disp = np.rot90(normalize_slice(sl, win))
            im_gray.set_data(sl_disp)
            im_gray.set_extent([0, sl_disp.shape[1], 0, sl_disp.shape[0]])
        else:
            im_gray.set_data(blank)

        if msl is not None:
            msl_disp = np.rot90(msl.astype(np.float32))
            im_tumor.set_data(msl_disp)
            im_tumor.set_extent(im_gray.get_extent())
            im_tumor.set_visible(True)
        else:
            im_tumor.set_visible(False)

        ax_img.set_xlim(im_gray.get_extent()[:2])
        ax_img.set_ylim(im_gray.get_extent()[2:])

        tp_key    = available_tp[state["tp"]]
        tp_label  = tp_key.upper()
        seq_key   = SEQUENCES[state["seq"]]
        seq_label = SEQ_LABELS[seq_key]
        axis_names = ["Axial", "Coronal", "Sagittal"]
        has_mask  = masks.get(tp_key) is not None
        mask_note = "🔴 visible" if (has_mask and state["show_tumor"]) \
                    else ("⚪ masquée" if has_mask else "— absent")

        # Volume tumoral estimé (nombre de voxels non nuls × volume voxel)
        raw_mask = masks.get(tp_key)
        if raw_mask is not None:
            n_vox = int(np.sum(raw_mask > 0))
            # voxel ~ 1mm³ pour les volumes BraTS 1×1×1 mm
            tumor_vol_cm3 = n_vox / 1000.0
            tumor_str = f"{tumor_vol_cm3:.1f} cm³  ({n_vox:,} vox.)"
        else:
            tumor_str = "—"

        title_ax.set_text(
            f"{tp_label}  ·  {seq_label}  ·  {axis_names[ax]}  ·  coupe {idx+1}/{n_slices()}"
        )

        # Panneau info enrichi
        vol_shape = vol.shape if vol is not None else "—"
        info_text.set_text(
            f"TEMPS  ({tp_label})\n"
            f"  Date : {TP_DATES.get(tp_key, '—')}\n"
            f"  Suivi : {TP_DELTA.get(tp_key, '—')}\n"
            f"\n"
            f"SÉQUENCE\n"
            f"  {seq_label}\n"
            f"  {SEQ_DESC.get(seq_key, '')}\n"
            f"\n"
            f"AXE DE VUE\n"
            f"  {axis_names[ax]}\n"
            f"  {AXIS_DESC[ax]}\n"
            f"\n"
            f"VOLUME TUMEUR\n"
            f"  {tumor_str}\n"
            f"  Masque : {mask_note}\n"
            f"\n"
            f"DIMENSIONS\n"
            f"  {vol_shape[0]}×{vol_shape[1]}×{vol_shape[2]} vox.\n"
            f"  Résolution : 1×1×1 mm\n"
            f"\n"
            f"FENÊTRAGE\n"
            f"  {wname}"
        )

        # Mise à jour frise chronologique
        x_marker = 0.12 + state["tp"] * 0.175
        timeline_marker.set_position((x_marker, 0.910))

        # Mise à jour slider SANS déclencher le callback (évite la récursion infinie)
        new_val = min(idx, n_slices() - 1)
        slider.valmax = n_slices() - 1
        slider.ax.set_xlim(0, max(n_slices() - 1, 1))
        # Désactiver temporairement le callback puis forcer la valeur
        slider.eventson = False
        slider.set_val(new_val)
        slider.eventson = True

        fig.canvas.draw_idle()

    # ── Slider coupe ─────────────────────────────────────────
    ax_sl = plt.axes([0.03, 0.10, 0.65, 0.025], facecolor="#111")
    slider = Slider(ax_sl, "Coupe", 0, n_slices() - 1,
                    valinit=state["idx"], valstep=1, color="#e53935")
    slider.label.set_color("white")
    slider.valtext.set_color("#e53935")

    def on_slider(val):
        state["idx"] = int(slider.val)
        refresh()
    slider.on_changed(on_slider)

    # ── Boutons rang 1 ───────────────────────────────────────
    BTN_H, BTN_Y1, BTN_Y2 = 0.038, 0.055, 0.010
    BTN_COL   = "#0e1318"
    BTN_HOVER = "#1c2530"

    def make_btn(x, w, label, y=BTN_Y1, color="#1979b8"):
        a = plt.axes([x, y, w, BTN_H], facecolor=BTN_COL)
        b = Button(a, label, color=BTN_COL, hovercolor=BTN_HOVER)
        b.label.set_color(color)
        b.label.set_fontsize(8.5)
        return b

    # Timepoint
    btn_tp = make_btn(0.03, 0.10, f"◷ {available_tp[0].upper()}")
    def on_tp(e):
        state["tp"]  = (state["tp"] + 1) % len(available_tp)
        state["idx"] = n_slices() // 2
        btn_tp.label.set_text(f"◷ {available_tp[state['tp']].upper()}")
        refresh()
    btn_tp.on_clicked(on_tp)

    # Séquence
    btn_sq = make_btn(0.145, 0.10, f"⊡ {SEQ_LABELS[SEQUENCES[0]]}")
    def on_seq(e):
        state["seq"] = (state["seq"] + 1) % len(SEQUENCES)
        btn_sq.label.set_text(f"⊡ {SEQ_LABELS[SEQUENCES[state['seq']]]}")
        refresh()
    btn_sq.on_clicked(on_seq)

    # Axe
    axis_labels = ["Axial", "Coronal", "Sagittal"]
    btn_ax = make_btn(0.26, 0.09, f"⊕ {axis_labels[2]}")
    def on_axis(e):
        state["axis"] = (state["axis"] + 1) % 3
        state["idx"]  = n_slices() // 2
        btn_ax.label.set_text(f"⊕ {axis_labels[state['axis']]}")
        refresh()
    btn_ax.on_clicked(on_axis)

    # Fenêtrage
    win_names = list(WINDOWS.keys())
    btn_win = make_btn(0.365, 0.09, f"⊞ {win_names[0]}")
    def on_win(e):
        state["window"] = (state["window"] + 1) % len(win_names)
        btn_win.label.set_text(f"⊞ {win_names[state['window']]}")
        refresh()
    btn_win.on_clicked(on_win)

    # Toggle tumeur
    btn_tm = make_btn(0.47, 0.12, "🔴 Tumeur ON", color="#e53935")
    def on_tumor(e):
        state["show_tumor"] = not state["show_tumor"]
        btn_tm.label.set_text(
            "🔴 Tumeur ON" if state["show_tumor"] else "⚪ Tumeur OFF"
        )
        btn_tm.label.set_color("#e53935" if state["show_tumor"] else "#556")
        refresh()
    btn_tm.on_clicked(on_tumor)

    # ── Clavier & molette ─────────────────────────────────────
    def on_scroll(event):
        delta = 1 if event.button == "up" else -1
        state["idx"] = max(0, min(state["idx"] + delta, n_slices() - 1))
        slider.set_val(state["idx"])

    def on_key(event):
        if event.key in ("right", "up"):
            state["idx"] = min(state["idx"] + 1, n_slices() - 1)
        elif event.key in ("left", "down"):
            state["idx"] = max(state["idx"] - 1, 0)
        elif event.key == "t":
            state["tp"]  = (state["tp"] + 1) % len(available_tp)
            state["idx"] = n_slices() // 2
            btn_tp.label.set_text(f"◷ {available_tp[state['tp']].upper()}")
        elif event.key == "s":
            state["seq"] = (state["seq"] + 1) % len(SEQUENCES)
            btn_sq.label.set_text(f"⊡ {SEQ_LABELS[SEQUENCES[state['seq']]]}")
        elif event.key == "a":
            state["axis"] = (state["axis"] + 1) % 3
            state["idx"]  = n_slices() // 2
            btn_ax.label.set_text(f"⊕ {axis_labels[state['axis']]}")
        elif event.key == "m":
            state["show_tumor"] = not state["show_tumor"]
            btn_tm.label.set_text("🔴 Tumeur ON" if state["show_tumor"] else "⚪ Tumeur OFF")
        slider.set_val(state["idx"])

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Init
    state["idx"] = n_slices() // 2
    refresh()

    plt.suptitle("🧠  Visionneuse IRM Cérébrale  —  Patient P01", color="white",
                 fontsize=13, fontweight="bold", y=0.97, fontfamily="monospace")
    fig.text(
        0.5, 0.005,
        "Molette / ← →  naviguer  |  T = temps  |  S = séquence  |  A = axe  |  M = masque tumeur",
        ha="center", color="#334", fontsize=8
    )
    plt.show()


# ══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        folder = input("Chemin vers le dossier patient (P01/) : ").strip().strip('"')
    else:
        folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(f"[ERREUR] Dossier introuvable : {folder}")
        sys.exit(1)

    print(f"\n[INFO] Chargement depuis : {folder}")
    data, masks = find_data(folder)

    available = [tp for tp in TIMEPOINTS if any(data[tp][sq] is not None for sq in SEQUENCES)]
    if not available:
        print("[ERREUR] Aucun fichier NIfTI chargé. Vérifiez l'arborescence.")
        sys.exit(1)

    print(f"\n[OK] Timepoints disponibles : {available}")
    launch_viewer(data, masks)
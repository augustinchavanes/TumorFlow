"""
augment_biomedical_dataset.py
=============================
Augmentation intelligente pour petits cohorts oncologie / EDP Fisher-KPP.

Stratégie :
  1. Interpolation SMOTE-like en régression (k plus proches voisins dans l'espace X).
  2. Redressement physique : D cohérent avec la morphologie, (r, D) alignés sur
     v = 2√(rD) et λ = √(D/r) dérivés de la géométrie interpolée.

Usage :
  python augment_biomedical_dataset.py
  python augment_biomedical_dataset.py --x_csv results_rfml/feature_matrix.csv \\
      --y_csv results_rfml/ground_truth_parameters.csv --output results_rfml/augmented

API :
  from augment_biomedical_dataset import augment_biomedical_dataset
  X_aug, y_aug = augment_biomedical_dataset(X_df, y_df, target_count=200)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Bornes physiques / cliniques (cohérentes avec random_forest_pipeline.py) ──
BOUNDS = {
    "D": (0.001, 15.0),
    "r": (0.001, 0.15),
    "alpha": (0.0, 0.3),
    "t_treat": (0.0, 400.0),
    "Age": (18.0, 95.0),
    "KPS": (10.0, 100.0),
    "WHO_PS": (0.0, 4.0),
    "Dose_Gy": (0.0, 80.0),
    "Fractions": (1.0, 40.0),
    "Prior_Chemo": (0.0, 1.0),
    "Histology_enc": (0.0, 5.0),
    "N_metastases": (0.0, 10.0),
    "t1c_Sphericity": (0.0, 1.0),
    "t1c_Elongation": (0.0, 1.0),
    "t1c_Maximum3DDiameter": (5.0, 120.0),
    "t1c_MeshVolume": (100.0, 200_000.0),
    "ratio_T_O_Volume": (0.0, 5.0),
    "t1c_VoxelValue_Mean": (0.0, 3000.0),
}

TARGET_DEFAULT = ["D", "r", "alpha"]
GEOM_COLS = {
    "sphericity": "t1c_Sphericity",
    "diameter": "t1c_Maximum3DDiameter",
    "elongation": "t1c_Elongation",
}

RNG = np.random.default_rng(42)


def _clip_series(name: str, values: np.ndarray) -> np.ndarray:
    if name in BOUNDS:
        lo, hi = BOUNDS[name]
        return np.clip(values, lo, hi)
    return values


def estimate_D_from_shape(
    sphericity: np.ndarray,
    max_diam_mm: np.ndarray,
    elongation: np.ndarray,
) -> np.ndarray:
    """Heuristique D [mm²/j] — même logique que le pipeline principal."""
    irreg = np.maximum(0.0, 1.0 - sphericity)
    extent = np.clip(max_diam_mm / 100.0, 0.05, 5.0)
    aniso = np.maximum(0.0, 1.0 - elongation)
    D = 0.05 * irreg * extent * (1.0 + 0.5 * aniso)
    return np.clip(D, BOUNDS["D"][0], BOUNDS["D"][1])


def _geometry_arrays(X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sph_col = GEOM_COLS["sphericity"]
    diam_col = GEOM_COLS["diameter"]
    elo_col = GEOM_COLS["elongation"]
    sph = X[sph_col].values.astype(float) if sph_col in X.columns else np.full(len(X), 0.5)
    diam = X[diam_col].values.astype(float) if diam_col in X.columns else np.full(len(X), 50.0)
    if elo_col in X.columns:
        elo = X[elo_col].values.astype(float)
    else:
        elo = np.full(len(X), 0.5)
    return sph, diam, elo


def target_lambda_from_geometry(sphericity: np.ndarray, elongation: np.ndarray) -> np.ndarray:
    """
    Largeur de front λ = √(D/r) cible : tumeur irrégulière → front plus large.
    λ ∈ [1, 25] mm (ordre de grandeur Swanson / cohorte).
    """
    irregularity = 0.5 * (1.0 - sphericity) + 0.5 * (1.0 - elongation)
    lam = 2.0 + 18.0 * np.clip(irregularity, 0.0, 1.0)
    return np.clip(lam, 1.0, 25.0)


def target_v_front_from_geometry(
    sphericity: np.ndarray,
    elongation: np.ndarray,
    diameter_mm: np.ndarray,
) -> np.ndarray:
    """Vitesse de front v = 2√(rD) [mm/j] — plus invasive si forme irrégulière / grande."""
    irregularity = 0.5 * (1.0 - sphericity) + 0.5 * (1.0 - elongation)
    size_factor = np.clip(diameter_mm / 50.0, 0.3, 3.0)
    v = 0.05 + 1.2 * irregularity * size_factor
    return np.clip(v, 0.02, 3.0)


def redress_physics_targets(
    X_row: pd.Series,
    y_interp: dict,
    blend_geom: float = 0.35,
) -> dict:
    """
    Redresse (D, r, α) après interpolation statistique.

    Logique :
      1. D ← mélange D_interp et D_morpho(sphéricité, diamètre, élongation).
      2. λ_cible et v_cible depuis la géométrie interpolée (lois Fisher-KPP).
      3. r ← D / λ_cible² puis ajustement léger pour rapprocher v = 2√(rD).
      4. α, t_treat : clipping ; pas de recalcul ODE (coût / peu de TP synthétiques).
    """
    sph, diam, elo = _geometry_arrays(X_row.to_frame().T)
    sph, diam, elo = float(sph[0]), float(diam[0]), float(elo[0])

    D_i = float(y_interp.get("D", np.nan))
    r_i = float(y_interp.get("r", np.nan))
    alpha_i = float(y_interp.get("alpha", np.nan))

    D_geom = float(estimate_D_from_shape(np.array([sph]), np.array([diam]), np.array([elo]))[0])
    if np.isnan(D_i):
        D_final = D_geom
    else:
        D_final = (1.0 - blend_geom) * D_i + blend_geom * D_geom
    D_final = float(_clip_series("D", np.array([D_final]))[0])

    lam_target = float(target_lambda_from_geometry(np.array([sph]), np.array([elo]))[0])
    v_target = float(
        target_v_front_from_geometry(np.array([sph]), np.array([elo]), np.array([diam]))[0]
    )

    r_from_lambda = D_final / max(lam_target ** 2, 1e-8)
    r_from_v = (v_target / 2.0) ** 2 / max(D_final, 1e-8)
    r_final = 0.6 * r_from_lambda + 0.4 * r_from_v
    if not np.isnan(r_i):
        r_final = 0.5 * r_final + 0.5 * r_i
    r_final = float(_clip_series("r", np.array([r_final]))[0])

    v_check = 2.0 * np.sqrt(r_final * D_final)
    lam_check = np.sqrt(D_final / max(r_final, 1e-10))
    if abs(v_check - v_target) / (v_target + 1e-6) > 0.25:
        r_final = float(_clip_series("r", np.array([(v_target / 2.0) ** 2 / D_final]))[0])

    alpha_final = float(_clip_series("alpha", np.array([alpha_i if not np.isnan(alpha_i) else 0.05]))[0])
    out = {
        "D": D_final,
        "r": r_final,
        "alpha": alpha_final,
        "v_front": round(2.0 * np.sqrt(r_final * D_final), 5),
        "lambda_front": round(np.sqrt(D_final / r_final), 5),
        "D_geom": D_geom,
        "lambda_target": lam_target,
        "v_target": v_target,
    }
    if "t_treat" in y_interp and not np.isnan(y_interp.get("t_treat", np.nan)):
        out["t_treat"] = float(_clip_series("t_treat", np.array([y_interp["t_treat"]]))[0])
    return out


def _clip_feature_row(row: pd.Series, feature_cols: Sequence[str]) -> pd.Series:
    out = row.copy()
    for c in feature_cols:
        v = out[c]
        if c in BOUNDS:
            lo, hi = BOUNDS[c]
            out[c] = float(np.clip(v, lo, hi))
        elif c.endswith("_Volume") or "MeshVolume" in c:
            out[c] = float(np.clip(v, 1.0, None))
    return out


def smote_like_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_synthetic: int,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère n_synthetic échantillons par interpolation entre voisins proches (SMOTE-R).

    Pour chaque synthétique : choisit un point i, un voisin j, λ ~ U(0.15, 0.85),
    X_new = X_i + λ (X_j - X_i),  y_new = y_i + λ (y_j - y_i).
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    if n < 2:
        raise ValueError("Au moins 2 patients réels requis pour l'interpolation.")

    k = min(k_neighbors, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    X_syn = np.zeros((n_synthetic, X.shape[1]), dtype=float)
    y_syn = np.zeros((n_synthetic, y.shape[1]), dtype=float)

    for s in range(n_synthetic):
        i = int(rng.integers(0, n))
        neighbors = indices[i, 1:]
        j_idx = int(rng.choice(neighbors))
        lam = float(rng.uniform(0.15, 0.85))
        X_syn[s] = X[i] + lam * (X[j_idx] - X[i])
        y_syn[s] = y[i] + lam * (y[j_idx] - y[i])

    return X_syn, y_syn


def augment_biomedical_dataset(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    target_count: int = 200,
    target_cols: Optional[Sequence[str]] = None,
    id_col: str = "Patient_ID",
    k_neighbors: int = 5,
    blend_geom: float = 0.35,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enrichit le dataset (X, y) jusqu'à target_count lignes (réels + synthétiques).

    Parameters
    ----------
    X_df : DataFrame
        Features (doit contenir Patient_ID + colonnes numériques).
    y_df : DataFrame
        Cibles physiques (Patient_ID + D, r, alpha, …).
    target_count : int
        Effectif final souhaité (ex. 200).
    target_cols : list, optional
        Colonnes cibles ; défaut ['D','r','alpha'].

    Returns
    -------
    X_augmented, y_augmented : DataFrames avec colonnes is_synthetic, source_id.
    """
    target_cols = list(target_cols or TARGET_DEFAULT)
    for c in target_cols:
        if c not in y_df.columns:
            raise ValueError(f"Colonne cible manquante dans y_df : {c}")

    if id_col not in X_df.columns or id_col not in y_df.columns:
        raise ValueError(f"Colonne {id_col} requise dans X_df et y_df.")

    # Ne fusionner que les cibles (évite Prior_Chemo_x / _y si présent dans X et y)
    extra_y = [c for c in ("t_treat",) if c in y_df.columns and c not in X_df.columns]
    y_merge_cols = [id_col] + list(dict.fromkeys(target_cols + extra_y))
    merged = X_df.merge(y_df[y_merge_cols], on=id_col, how="inner")
    merged = merged.dropna(subset=target_cols)
    n_real = len(merged)
    if n_real < 2:
        raise ValueError(f"Pas assez de patients valides ({n_real}) après fusion X/y.")

    n_needed = max(0, target_count - n_real)
    log.info("Patients réels : %d | Synthétiques à générer : %d", n_real, n_needed)

    feature_cols = [
        c for c in X_df.columns
        if c != id_col and c in merged.columns
        and pd.api.types.is_numeric_dtype(X_df[c])
    ]
    if not feature_cols:
        feature_cols = [c for c in merged.columns if c not in [id_col] + target_cols + extra_y]

    X_real = merged[feature_cols].values.astype(float)
    y_real = merged[target_cols].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_real)

    X_out_rows = []
    y_out_rows = []

    # ── Réels ──
    for idx, row in merged.iterrows():
        xr = _clip_feature_row(row[feature_cols], feature_cols)
        yr = {c: float(row[c]) for c in target_cols}
        sph, diam, elo = _geometry_arrays(xr.to_frame().T)
        yr_red = redress_physics_targets(xr, yr, blend_geom=0.0)
        X_out_rows.append({id_col: row[id_col], **xr.to_dict(), "is_synthetic": 0, "source_id": row[id_col]})
        y_out_rows.append({id_col: row[id_col], **{c: yr_red[c] for c in target_cols},
                           "v_front": yr_red["v_front"], "lambda_front": yr_red["lambda_front"],
                           "is_synthetic": 0})

    # ── Synthétiques ──
    if n_needed > 0:
        X_syn_s, y_syn_s = smote_like_regression(
            X_scaled, y_real, n_needed, k_neighbors=k_neighbors, random_state=random_state,
        )
        X_syn_orig = scaler.inverse_transform(X_syn_s)

        for s in range(n_needed):
            x_dict = {feature_cols[i]: float(X_syn_orig[s, i]) for i in range(len(feature_cols))}
            xr = _clip_feature_row(pd.Series(x_dict), feature_cols)
            y_interp = {target_cols[i]: float(y_syn_s[s, i]) for i in range(len(target_cols))}
            yr = redress_physics_targets(xr, y_interp, blend_geom=blend_geom)
            syn_id = f"SYN_{s+1:04d}"
            X_out_rows.append({id_col: syn_id, **xr.to_dict(), "is_synthetic": 1, "source_id": "SMOTE_kNN"})
            row_y = {id_col: syn_id, **{c: yr[c] for c in target_cols},
                     "v_front": yr["v_front"], "lambda_front": yr["lambda_front"],
                     "is_synthetic": 1}
            if "t_treat" in yr:
                row_y["t_treat"] = yr["t_treat"]
            y_out_rows.append(row_y)

    X_aug = pd.DataFrame(X_out_rows)
    y_aug = pd.DataFrame(y_out_rows)
    log.info("Dataset augmenté : %d lignes (%d réelles, %d synthétiques)",
             len(X_aug), n_real, n_needed)
    return X_aug, y_aug


def plot_augmentation_diagnostic(
    X_real: pd.DataFrame,
    y_real: pd.DataFrame,
    X_aug: pd.DataFrame,
    y_aug: pd.DataFrame,
    output_path: Path,
    target_cols: Sequence[str] = TARGET_DEFAULT,
    id_col: str = "Patient_ID",
):
    """PCA 2D + nuage D vs r (réels vs synthétiques)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib absent — pas de figure de diagnostic.")
        return

    feature_cols = [c for c in X_real.columns if c != id_col and c not in ("is_synthetic", "source_id")]
    feat_aug = [c for c in feature_cols if c in X_aug.columns]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── D vs r ──
    ax = axes[0]
    m_real = y_aug["is_synthetic"] == 0
    m_syn = y_aug["is_synthetic"] == 1
    ax.scatter(y_aug.loc[m_real, "D"], y_aug.loc[m_real, "r"],
               c="#1D3557", s=70, alpha=0.9, edgecolors="white", label=f"Réels ({m_real.sum()})")
    ax.scatter(y_aug.loc[m_syn, "D"], y_aug.loc[m_syn, "r"],
               c="#E63946", s=35, alpha=0.5, label=f"Synthétiques ({m_syn.sum()})")
    ax.set(xlabel="D [mm²/j]", ylabel="r [/j]", title="Paramètres physiques (après redressement)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── PCA features ──
    ax = axes[1]
    X_comb = X_aug[feat_aug].values.astype(float)
    labels = y_aug["is_synthetic"].values
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(StandardScaler().fit_transform(X_comb))
    ax.scatter(Z[labels == 0, 0], Z[labels == 0, 1], c="#1D3557", s=70, alpha=0.9,
               edgecolors="white", label="Réels")
    ax.scatter(Z[labels == 1, 0], Z[labels == 1, 1], c="#E63946", s=35, alpha=0.45,
               label="Synthétiques")
    ax.set(xlabel=f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f} %)",
           ylabel=f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f} %)",
           title="PCA des features (X)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle("Augmentation biomédicale — réels vs synthétiques", fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Diagnostic sauvegardé : %s", output_path)


def load_pipeline_tables(script_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge feature_matrix + ground_truth depuis results_rfml si présents."""
    rf_dir = script_dir / "results_rfml"
    x_path = rf_dir / "feature_matrix.csv"
    y_path = rf_dir / "ground_truth_parameters.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Exécutez d'abord random_forest_pipeline.py ou fournissez --x_csv / --y_csv."
        )
    X_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    y_df = y_df[y_df["success"].astype(bool)] if "success" in y_df.columns else y_df
    return X_df, y_df


def main():
    p = argparse.ArgumentParser(description="Augmentation SMOTE-like + contraintes Fisher-KPP")
    p.add_argument("--x_csv", default=None, help="feature_matrix.csv")
    p.add_argument("--y_csv", default=None, help="ground_truth_parameters.csv")
    p.add_argument("--output", default="results_rfml/augmented")
    p.add_argument("--target_count", type=int, default=200)
    p.add_argument("--k_neighbors", type=int, default=5)
    p.add_argument("--blend_geom", type=float, default=0.35,
                   help="Poids du D morphologique dans le redressement")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.x_csv and args.y_csv:
        X_df = pd.read_csv(args.x_csv)
        y_df = pd.read_csv(args.y_csv)
        if "success" in y_df.columns:
            y_df = y_df[y_df["success"].astype(bool)]
    else:
        X_df, y_df = load_pipeline_tables(script_dir)

    X_aug, y_aug = augment_biomedical_dataset(
        X_df, y_df,
        target_count=args.target_count,
        k_neighbors=args.k_neighbors,
        blend_geom=args.blend_geom,
        random_state=args.seed,
    )

    out = script_dir / args.output
    out.mkdir(parents=True, exist_ok=True)
    X_aug.to_csv(out / "X_augmented.csv", index=False)
    y_aug.to_csv(out / "y_augmented.csv", index=False)

    y_real = y_df.merge(X_df[["Patient_ID"]], on="Patient_ID")
    plot_augmentation_diagnostic(
        X_df, y_real, X_aug, y_aug,
        out / "augmentation_diagnostic.png",
    )

    print(f"\n  Fichiers -> {out.resolve()}/")
    print(f"    X_augmented.csv ({len(X_aug)} lignes)")
    print(f"    y_augmented.csv")
    print(f"    augmentation_diagnostic.png\n")


if __name__ == "__main__":
    main()

"""
PROTEAS-MRI Radiomics Extractor
================================
Parcourt toutes les feuilles patients d'un fichier Excel radiomique,
extrait les 12 paramètres critiques pour chaque couple (Masque / Séquence / Timepoint),
et génère un fichier master_simulation_data.csv au format large (une ligne par combo).

Usage:
    python extract_radiomics.py

Dépendances:
    pip install pandas openpyxl
"""

import pandas as pd

# ── Chemins ────────────────────────────────────────────────────────────────────
XLSX_PATH   = '17253793\PROTEAS-MRI_radiomics_data.xlsx'   # adapte ce chemin si besoin
OUTPUT_PATH = 'master_simulation_data.csv'

# ── Filtres ────────────────────────────────────────────────────────────────────
TARGET_MASKS      = ['mask_tumor', 'mask_oedema']
TARGET_SEQS       = ['t1c', 'fla']
TARGET_TIMEPOINTS = ['baseline', 'follow_up_1', 'follow_up_2', 'follow_up_3', 'follow_up_4']

# ── Mapping paramètres ─────────────────────────────────────────────────────────
# Clé   = nom de colonne dans le CSV de sortie
# Valeur = suffixe exact dans la colonne RadiomicsFeature du fichier source
#
# Notes sur les proxies d'intensité :
#   VoxelValue_Maximum → 90e percentile  (le vrai maximum pixel n'est pas extrait)
#   VoxelValue_SD      → Variance         (prendre sqrt() si tu veux l'écart-type)
FEATURE_MAP = {
    # Morphologie
    'MeshVolume':                     'original_shape_MeshVolume',
    'SurfaceArea':                    'original_shape_SurfaceArea',
    'Sphericity':                     'original_shape_Sphericity',
    'Maximum3DDiameter':              'original_shape_Maximum3DDiameter',
    'Elongation':                     'original_shape_Elongation',
    # Intensité (first-order)
    'VoxelValue_Mean':                'original_firstorder_Mean',
    'VoxelValue_Maximum':             'original_firstorder_90Percentile',
    'VoxelValue_SD':                  'original_firstorder_Variance',
    # Texture
    'GLCM_Entropy':                   'original_glcm_JointEntropy',
    'GLCM_Contrast':                  'original_glcm_Contrast',
    'GLSZM_SizeZoneNonUniformity':    'original_glszm_SizeZoneNonUniformity',
    'GLDM_DependenceNonUniformity':   'original_gldm_DependenceNonUniformity',
}

# ── Ordre de tri des timepoints ────────────────────────────────────────────────
TP_ORDER = {tp: i for i, tp in enumerate(TARGET_TIMEPOINTS)}


def parse_feature_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Décompose la colonne RadiomicsFeature au format :
        mask__sequence__timepoint__original_family_FeatureName
    en colonnes séparées _mask, _seq, _timepoint, _feat.
    """
    parsed = df['RadiomicsFeature'].str.split('__', expand=True)
    df = df.copy()
    df['_mask']      = parsed[0].str.lower()
    df['_seq']       = parsed[1].str.lower()
    df['_timepoint'] = parsed[2].str.lower()
    # La feature peut elle-même contenir des __ (ex: original_shape_MeshVolume)
    df['_feat']      = parsed.iloc[:, 3:].apply(
        lambda r: '__'.join(r.dropna()), axis=1
    )
    return df


def extract_patient(sheet_name: str, df: pd.DataFrame) -> list[dict]:
    """
    Pour un patient, filtre et pivote les données en format large.
    Retourne une liste de dicts (un dict = une ligne du CSV final).
    """
    if 'RadiomicsFeature' not in df.columns:
        print(f"  [{sheet_name}] ignoré : colonne RadiomicsFeature absente")
        return []

    df = parse_feature_column(df)

    # Filtrage
    df = df[df['_mask'].isin(TARGET_MASKS)]
    df = df[df['_seq'].isin(TARGET_SEQS)]
    df = df[df['_timepoint'].isin(TARGET_TIMEPOINTS)]

    if df.empty:
        print(f"  [{sheet_name}] aucune donnée après filtrage")
        return []

    # Ne garder que les features cibles
    suffix_to_canonical = {v: k for k, v in FEATURE_MAP.items()}
    df = df[df['_feat'].isin(FEATURE_MAP.values())].copy()
    df['_canonical'] = df['_feat'].map(suffix_to_canonical)

    records = []
    for (mask, seq, tp), grp in df.groupby(['_mask', '_seq', '_timepoint']):
        row = {
            'Patient_ID':      sheet_name,
            'Mask':            mask,
            'Sequence':        seq,
            'Timepoint_Label': tp,
        }
        for _, r in grp.iterrows():
            row[r['_canonical']] = r['RadiomicsValue']
        records.append(row)

    return records


def main():
    print(f"Lecture de {XLSX_PATH} …")
    all_sheets = pd.read_excel(XLSX_PATH, sheet_name=None)
    print(f"  {len(all_sheets)} feuilles trouvées : {list(all_sheets.keys())[:5]} …\n")

    all_records = []
    for sheet_name, df in all_sheets.items():
        records = extract_patient(sheet_name, df)
        all_records.extend(records)

    master = pd.DataFrame(all_records)

    # Colonnes ordonnées
    meta_cols = ['Patient_ID', 'Mask', 'Sequence', 'Timepoint_Label']
    feat_cols = list(FEATURE_MAP.keys())
    ordered   = meta_cols + [c for c in feat_cols if c in master.columns]
    master    = master[ordered]

    # Tri
    master['_tp_sort'] = master['Timepoint_Label'].map(TP_ORDER)
    master = (
        master
        .sort_values(['Patient_ID', 'Mask', 'Sequence', '_tp_sort'])
        .drop(columns='_tp_sort')
        .reset_index(drop=True)
    )

    master.to_csv(OUTPUT_PATH, index=False)

    # Rapport
    print(f"\n✓ Export : {OUTPUT_PATH}")
    print(f"  {len(master)} lignes  ×  {len(master.columns)} colonnes")
    print(f"  {master['Patient_ID'].nunique()} patients")
    print(f"\nCouverture par paramètre :")
    for c in feat_cols:
        if c in master.columns:
            pct = master[c].notna().mean() * 100
            print(f"  {c:<35} {pct:.1f}%")

    print("\nAperçu (5 premières lignes) :")
    print(master.head().to_string())


if __name__ == '__main__':
    main()

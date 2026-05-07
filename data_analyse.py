import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. ANALYSE DES DONNÉES CLINIQUES
# Chargement du fichier "PROTEAS-Clinical_and_demographic_data.xlsx"
clinical_file = r'17253793\PROTEAS-Clinical_and_demographic_data.xlsx'
clinical_df = pd.read_excel(clinical_file)

# Nettoyage : Extraction de la dose numérique (ex: "20 Gy" -> 20)
clinical_df['Dose_Gy'] = clinical_df['Rx dose at tumor margins'].str.extract(r'(\d+)').astype(float)

# Sélection des colonnes numériques principales
num_cols_clinical = ['Age (yrs)', 'Fractions', 'Dose_Gy', 'WHO PS', 'Karnofsky PS (%)', 'Number of metastases']

# --- Données manquantes ---
print("=" * 80)
print("AVANT IMPUTATION : Données manquantes par colonne (Clinique)")
print("=" * 80)
missing_before = clinical_df.isnull().sum()
print(missing_before)

# ============================================================================
# 3. STRATÉGIE DE NETTOYAGE RIGOUREUSE - IMPUTATION DES VALEURS MANQUANTES
# ============================================================================
print("\n" + "=" * 80)
print("STRATÉGIE D'IMPUTATION - DONNÉES ONCOLOGIQUES")
print("=" * 80)

# Créer une copie pour le rapport avant/après
clinical_df_before = clinical_df.copy()

# --- 3.1 SYNCHRONISATION DES SCORES DE PERFORMANCE (WHO PS et Karnofsky PS) ---
print("\n[1/4] Synchronisation WHO PS ↔ Karnofsky PS...")

# Règle de conversion clinique standard
who_to_karnofsky = {
    0: 100,
    1: 85,    # Moyenne de 80-90%
    2: 65,    # Moyenne de 60-70%
    3: 45     # Moyenne de 40-50%
}

# Vérifier la corrélation avant imputation
valid_both = clinical_df[['WHO PS', 'Karnofsky PS (%)']].dropna()
if len(valid_both) > 1:
    corr_before = valid_both['WHO PS'].corr(valid_both['Karnofsky PS (%)'])
    print(f"  Corrélation WHO PS ↔ Karnofsky PS : {corr_before:.3f}")
else:
    corr_before = None
    print(f"  Corrélation : Données insuffisantes")

# Cas 1 : WHO PS manquant, Karnofsky PS présent → déduire WHO PS (inverse)
karnofsky_to_who = {v: k for k, v in who_to_karnofsky.items()}
mask_who_missing = clinical_df['WHO PS'].isnull() & clinical_df['Karnofsky PS (%)'].notna()
for idx in clinical_df[mask_who_missing].index:
    karnofsky_val = clinical_df.loc[idx, 'Karnofsky PS (%)']
    # Trouver la correspondance la plus proche
    closest_who = min(karnofsky_to_who.keys(), key=lambda x: abs(x - karnofsky_val))
    clinical_df.loc[idx, 'WHO PS'] = karnofsky_to_who[closest_who]
    print(f"  ✓ Ligne {idx}: WHO PS imputé via Karnofsky ({karnofsky_val}% → WHO {karnofsky_to_who[closest_who]})")

# Cas 2 : Karnofsky PS manquant, WHO PS présent → déduire Karnofsky PS
mask_karnofsky_missing = clinical_df['Karnofsky PS (%)'].isnull() & clinical_df['WHO PS'].notna()
for idx in clinical_df[mask_karnofsky_missing].index:
    who_val = int(clinical_df.loc[idx, 'WHO PS'])
    clinical_df.loc[idx, 'Karnofsky PS (%)'] = who_to_karnofsky[who_val]
    print(f"  ✓ Ligne {idx}: Karnofsky PS imputé via WHO (WHO {who_val} → {who_to_karnofsky[who_val]}%)")

# Cas 3 : Les deux manquants → médiane
median_who = clinical_df['WHO PS'].median()
median_karnofsky = clinical_df['Karnofsky PS (%)'].median()
mask_both_missing = clinical_df['WHO PS'].isnull() & clinical_df['Karnofsky PS (%)'].isnull()
clinical_df.loc[mask_both_missing, 'WHO PS'] = median_who
clinical_df.loc[mask_both_missing, 'Karnofsky PS (%)'] = median_karnofsky
print(f"  ✓ {mask_both_missing.sum()} lignes : Les deux scores imputés par médiane (WHO={median_who}, Karnofsky={median_karnofsky}%)")

# --- 3.2 TRAITEMENT DE L'INCERTITUDE NEUROLOGIQUE ---
print("\n[2/4] Traitement Neurocognitive status...")
neuro_missing = clinical_df['Neurocognitive status'].isnull().sum()
clinical_df['Neurocognitive status'] = clinical_df['Neurocognitive status'].fillna('Unknown/Not Assessed')
print(f"  ✓ {neuro_missing} valeurs manquantes remplacées par 'Unknown/Not Assessed'")

# --- 3.3 ANALYSE DE FRÉQUENCE - PRÉSENCE ET EXTENSION DE LA MALADIE ---
print("\n[3/4] Imputation 'Presence and extent of cranial disease'...")
mode_cranial = clinical_df['Presence and extent of cranial disease'].mode()
if len(mode_cranial) > 0:
    most_frequent = mode_cranial[0]
else:
    most_frequent = 'Not specified'
cranial_missing = clinical_df['Presence and extent of cranial disease'].isnull().sum()
clinical_df['Presence and extent of cranial disease'] = clinical_df['Presence and extent of cranial disease'].fillna(most_frequent)
print(f"  ✓ Valeur la plus fréquente : '{most_frequent}'")
print(f"  ✓ {cranial_missing} valeurs manquantes imputées")

# --- 3.4 VALIDATION POST-IMPUTATION ---
print("\n[4/4] Validation post-imputation...")

# Colonnes à valider
imputed_cols = ['WHO PS', 'Karnofsky PS (%)', 'Neurocognitive status', 'Presence and extent of cranial disease']
missing_after = clinical_df[imputed_cols].isnull().sum()

print("\n" + "=" * 80)
print("RÉSUMÉ COMPARATIF - STATISTIQUES PRE/POST IMPUTATION")
print("=" * 80)

# Statistiques numériques
numeric_imputed = ['WHO PS', 'Karnofsky PS (%)']
comparison_stats = pd.DataFrame({
    'Colonne': numeric_imputed,
    'Moyenne (Avant)': [clinical_df_before[col].mean() for col in numeric_imputed],
    'Médiane (Avant)': [clinical_df_before[col].median() for col in numeric_imputed],
    'Écart-type (Avant)': [clinical_df_before[col].std() for col in numeric_imputed],
    'Moyenne (Après)': [clinical_df[col].mean() for col in numeric_imputed],
    'Médiane (Après)': [clinical_df[col].median() for col in numeric_imputed],
    'Écart-type (Après)': [clinical_df[col].std() for col in numeric_imputed],
})
print("\nVariables Numériques :")
print(comparison_stats.to_string(index=False))

# Vérification finale
print("\n" + "=" * 80)
print("VÉRIFICATION FINALE - VALEURS MANQUANTES")
print("=" * 80)
print("\nAVANT imputation :")
print(missing_before[imputed_cols])
print("\nAPRÈS imputation :")
print(missing_after[imputed_cols])

if missing_after.sum() == 0:
    print("\n✅ SUCCÈS : Toutes les valeurs manquantes ont été imputées avec succès !")
else:
    print(f"\n⚠️  ATTENTION : {missing_after.sum()} valeurs manquantes restantes")

print("\n" + "=" * 80)

# ============================================================================
# SAUVEGARDE DES DONNÉES IMPUTÉES DANS LE FICHIER EXCEL
# ============================================================================
print("\nSAUVEGARDE DES DONNÉES IMPUTÉES...")
output_file = r'17253793\PROTEAS-Clinical_and_demographic_data_imputed.xlsx'
clinical_df.to_excel(output_file, index=False, engine='openpyxl')
print(f"✅ Fichier sauvegardé : {output_file}")
print("=" * 80)

# --- Visualisation : Boxplot ---
plt.figure(figsize=(12, 6))
clinical_df[num_cols_clinical].boxplot()
plt.title('Distribution des variables cliniques (Boxplot)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- Visualisation : Corrélation ---
plt.figure(figsize=(10, 8))
sns.heatmap(clinical_df[num_cols_clinical].corr(), annot=True, cmap='RdBu_r', center=0)
plt.title('Matrice de Corrélation - Données Cliniques')
plt.show()


# 2. ANALYSE DES DONNÉES RADIOMICS
# Chargement du fichier "PROTEAS-MRI_radiomics_data.xlsx"
radiomics_file = r'17253793\PROTEAS-MRI_radiomics_data.xlsx'
xls_rad = pd.ExcelFile(radiomics_file)

all_patients_data = []
# On parcourt chaque feuille (un patient par feuille)
for sheet in xls_rad.sheet_names:
    df_sheet = pd.read_excel(xls_rad, sheet_name=sheet)
    # Pivoter les données : Features en colonnes, Valeurs en lignes
    df_pivoted = df_sheet.set_index('RadiomicsFeature').T
    df_pivoted['Patient_ID'] = sheet
    all_patients_data.append(df_pivoted)

# Fusion de tous les patients dans un seul DataFrame
combined_rad_df = pd.concat(all_patients_data, ignore_index=True)
combined_rad_df = combined_rad_df.apply(pd.to_numeric, errors='coerce')

# --- Données manquantes ---
print("\nDonnées manquantes par colonne (Radiomics) :")
print(combined_rad_df.isnull().sum())

# Sélection d'un échantillon de caractéristiques (ex: caractéristiques de forme 'Shape')
shape_features = [c for c in combined_rad_df.columns if 'shape' in c.lower()][:10]

# --- Visualisation : Boxplot Radiomics ---
plt.figure(figsize=(14, 7))
combined_rad_df[shape_features].boxplot()
plt.xticks(rotation=45, ha='right')
plt.title('Boxplot des caractéristiques de forme (Radiomics)')
plt.tight_layout()
plt.show()

# --- Visualisation : Corrélation Radiomics ---
plt.figure(figsize=(12, 10))
sns.heatmap(combined_rad_df[shape_features].corr(), annot=True, cmap='viridis')
plt.title('Corrélation entre caractéristiques Radiomics (Shape)')
plt.tight_layout()
plt.show()
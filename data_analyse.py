import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. ANALYSE DES DONNÉES CLINIQUES
# Chargement du fichier "PROTEAS-Clinical_and_demographic_data.xlsx"
clinical_file = 'PROTEAS-Clinical_and_demographic_data.xlsx'
clinical_df = pd.read_excel(clinical_file)

# Nettoyage : Extraction de la dose numérique (ex: "20 Gy" -> 20)
clinical_df['Dose_Gy'] = clinical_df['Rx dose at tumor margins'].str.extract('(\d+)').astype(float)

# Sélection des colonnes numériques principales
num_cols_clinical = ['Age (yrs)', 'Fractions', 'Dose_Gy', 'WHO PS', 'Karnofsky PS (%)', 'Number of metastases']

# --- Données manquantes ---
print("Données manquantes par colonne (Clinique) :")
print(clinical_df.isnull().sum())

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
radiomics_file = 'PROTEAS-MRI_radiomics_data.xlsx'
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
combined_rad_df = combined_rad_df.apply(pd.to_numeric, errors='ignore')

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
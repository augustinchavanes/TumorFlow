import pandas as pd

# Nom du fichier source
input_file = "17253793/PROTEAS-Clinical_and_demographic_data_imputed.xlsx"
output_file = "clinical_features.csv"

# Colonnes avec les noms exacts détectés dans ton fichier
columns_to_extract = [
    "Patient ID (Zenodo)",
    "Karnofsky PS (%)",
    "Rx dose at tumor margins",
    "Fractions",
    "Prior Chemo ", # Espace inclus à la fin
    "Presence and extent of cranial disease",
    "Age (yrs)",
    "Tumour Histology ", # Espace inclus à la fin
    "Number of metastases",
    "WHO PS",
    "Type & characteristics of intervention"
]

def run_extraction():
    try:
        # Lecture d'un fichier Excel
        df = pd.read_excel(input_file)
        
        # On ne garde que les colonnes qui existent vraiment dans le fichier
        available_cols = [c for c in columns_to_extract if c in df.columns]
        
        # Création du nouveau fichier
        df_filtered = df[available_cols]
        df_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"✅ Extraction réussie !")
        print(f"📁 Fichier créé : {output_file}")
        print(f"📊 Colonnes extraites : {len(available_cols)} sur {len(columns_to_extract)}")

    except Exception as e:
        print(f"❌ Erreur lors de l'extraction : {e}")

if __name__ == "__main__":
    run_extraction()
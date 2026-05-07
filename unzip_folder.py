import os
import zipfile

def unzip_logic_cascaded(root_path):
    """
    Niveau 1 : 17253793/P03.zip -> extrait dans 17253793/ (crée P03/)
    Niveau 2 : 17253793/P03/toto.zip -> extrait dans 17253793/P03/toto/
    Nettoyage : Suppression des ZIP après extraction.
    """
    if not os.path.exists(root_path):
        print(f"[ERREUR] Le dossier {root_path} n'existe pas.")
        return

    print(f"--- Début de l'extraction en cascade sur : {root_path} ---")

    # --- ÉTAPE 1 : TRAITEMENT DE LA RACINE ---
    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)

        if zipfile.is_zipfile(item_path):
            # On extrait directement dans root_path.
            # Si le ZIP contient déjà un dossier P03/ à l'intérieur,
            # extractall(root_path) créera root_path/P03/ sans doublon.
            if extract_flat(item_path, root_path):
                os.remove(item_path)
                print(f"  [OK] Racine : {item} extrait et supprimé.")

                # Dossier patient résultant = root_path/P03 (sans .zip)
                patient_folder = os.path.join(root_path, os.path.splitext(item)[0])
                if os.path.isdir(patient_folder):
                    process_level_two(patient_folder)

        elif os.path.isdir(item_path):
            # Dossier patient déjà extrait : on traite ses ZIP internes
            process_level_two(item_path)


def process_level_two(patient_folder):
    """Niveau 2 : P03/toto.zip -> P03/toto/"""
    for item in os.listdir(patient_folder):
        item_path = os.path.join(patient_folder, item)

        if zipfile.is_zipfile(item_path):
            target_sub_dir = os.path.splitext(item_path)[0]  # P03/toto

            if extract_flat(item_path, target_sub_dir):
                os.remove(item_path)
                print(f"      [OK] Interne : {item} -> {os.path.basename(target_sub_dir)}/ (supprimé)")


def extract_flat(zip_path, target_dir):
    """
    Extrait le ZIP dans target_dir en aplatissant d'un niveau si le ZIP
    ne contient qu'un seul dossier racine portant le même nom que lui-même.

    Exemple :
      P03.zip contient P03/BraTS/, P03/DICOM/, ...
      -> on extrait dans root_path (pas dans root_path/P03/)
      -> résultat : root_path/P03/BraTS/, root_path/P03/DICOM/, ...  ✓

      toto.zip contient des fichiers à plat
      -> on extrait dans target_dir/
      -> résultat : target_dir/fichier1, target_dir/fichier2, ...     ✓
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Nom du ZIP sans extension (ex: "P03")
            zip_stem = os.path.splitext(os.path.basename(zip_path))[0]

            # Racines de premier niveau dans le ZIP
            top_level = set()
            for name in zf.namelist():
                top = name.split('/')[0]
                if top:
                    top_level.add(top)

            # Si le ZIP contient UN seul dossier racine qui s'appelle
            # exactement comme le ZIP -> on extrait dans le parent
            # pour éviter la duplication P03/P03/
            if len(top_level) == 1 and list(top_level)[0] == zip_stem:
                dest = os.path.dirname(target_dir)  # remonte d'un niveau
            else:
                dest = target_dir

            os.makedirs(dest, exist_ok=True)
            zf.extractall(dest)
            return True

    except Exception as e:
        print(f"    [ERREUR] Impossible d'extraire {zip_path}: {e}")
        return False


# --- LANCEMENT ---
chemin_final = "17253793/"   # ← adapte ce chemin
unzip_logic_cascaded(chemin_final)
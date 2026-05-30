# TumorFlow

Plateforme de traitement et de modélisation de tumeurs cérébrales à partir de données IRM longitudinales (BraTS-like), données cliniques et segmentation tumorale.

Le projet couvre :
- préparation des données,
- extraction de features tabulaires,
- entraînement/prédiction de paramètres d’un modèle PDE (Fisher-KPP avec terme de traitement),
- simulation 3D de l’évolution tumorale,
- visualisation 2D/3D des volumes.

![Statistiques du dataset](dataset.png)

---

## 1) Objectif du projet

L’idée est de prédire des paramètres biologiques/physiques de croissance tumorale :
- `D` : diffusion/invasion,
- `r` : prolifération,
- `alpha` : effet du traitement (chimio modélisée),

puis de simuler l’évolution volumique d’un patient dans le temps.

Le pipeline principal actuel est `random_forest_pipeline.py`.

---

## 2) Installation rapide

### Prérequis
- Python 3.10+ (testé en environnement Windows/PowerShell),
- accès au dataset Zenodo.

### Setup

```powershell
python -m venv EnvVirtuel
.\EnvVirtuel\Scripts\activate
pip install -r .\requirements.txt
```

---

## 3) Dataset

Source :
[Zenodo DOI 10.5281/zenodo.17253793](https://doi.org/10.5281/zenodo.17253793)

Après téléchargement, placer `17253793/` à la racine du repo puis dézipper récursivement si nécessaire :

```powershell
python .\unzip_folder.py
```

### Structure attendue (exemple patient)

```text
17253793/
  P03/
    BraTS/
      baseline/
        fla.nii.gz
        t1.nii.gz
        t1c.nii.gz
        t2.nii.gz
      fu1/ ... fu4/
    tumor_segmentation/
      P03_tumor_mask_baseline.nii.gz
      P03_tumor_mask_fu1.nii.gz
      ...
    P03_brain_mask.nii.gz
    P03_CT.nii.gz
    P03_RTP.nii.gz
    DICOM/
      ...
```

---

## 4) Arborescence du projet (dossiers utiles)

- `17253793/` : données patients
- `features/` : CSV de features + wrappers d’extraction
- `results_rfml/` : résultats d’entraînement RF (modèle, métriques, prédictions)
- `prediction/<Patient_ID>/` : sorties de simulation patient
- `visualisation/` : viewers IRM (2D/3D)
- `DataAnalyse/` : script d’analyse exploratoire

Scripts principaux :
- `random_forest_pipeline.py` (pipeline principal),
- `extract_radiomics.py`, `extract_nifti_features.py`, `exctract_clinical_variables.py` (génération des features),
- `visualisation/brain_scanner_viewer.py`, `visualisation/brain3d_viewer.py` (visualisation).

---

## 5) Workflow recommandé (nouvel arrivant)

### Étape A — Analyse exploratoire

```powershell
python .\DataAnalyse\data_analyse.py
```

### Étape B — Générer les features

Les sorties sont écrites dans `features/` :
- `features/clinical_features.csv`
- `features/radiomics_features.csv`
- `features/nifti_features.csv`

Commandes :

```powershell
python .\exctract_clinical_variables.py
python .\extract_radiomics.py
python .\extract_nifti_features.py
```

Tu peux aussi lancer les wrappers dans `features/` :

```powershell
python .\features\exctract_clinical_variables.py
python .\features\extract_radiomics.py
python .\features\extract_nifti_features.py
```

### Étape C — Entraîner le modèle RF

```powershell
python .\random_forest_pipeline.py --data_dir 17253793 --output results_rfml
```

### Étape D — Prédire/simuler un patient

```powershell
python .\random_forest_pipeline.py --predict --patient_id P03 --simulate_days 200 --output results_rfml
```

Sorties dans `prediction/P03/` (courbe de volume, NIfTI simulé, CSV de volume simulé).

---

## 6) Pipeline ML/PDE (résumé technique)

Le modèle PDE implémenté est de type Fisher-KPP avec traitement :

\[
\partial u / \partial t = D \nabla^2 u + r u(1-u) - \alpha c(t)u
\]

où :
- `u` = densité tumorale normalisée,
- `D`, `r`, `alpha` = paramètres prédits,
- `c(t)` = concentration médicamenteuse modélisée.

### Ce que fait `random_forest_pipeline.py`

1. Charge les tables de features (`features/` prioritaire, fallback racine).
2. Construit des volumes longitudinaux via segmentations NIfTI.
3. Estime une “vérité terrain” des paramètres (`D`, `r`, `alpha`, `t_treat`) par patient.
4. Construit la matrice de features.
5. Entraîne un Random Forest multi-sorties.
6. Sauvegarde modèle + métriques + figures.
7. En mode `--predict`, prédit `D/r/alpha` pour un patient et simule en 3D.

---

## 7) Visualisation

### Viewer 2D scanner
```powershell
python .\visualisation\brain_scanner_viewer.py 17253793\P03
```

### Viewer 3D
```powershell
python .\visualisation\brain3d_viewer.py 17253793\P03 --all
```

---

## 8) Fichiers de sortie importants

### Entraînement (`results_rfml/`)
- `pde_ml_model.pkl` : modèle sauvegardé,
- `metrics.csv` : MAE/RMSE/R² par paramètre,
- `predictions_test.csv` : vérité vs prédiction sur le test set,
- `ground_truth_parameters.csv` : paramètres estimés patient par patient,
- `feature_matrix.csv` : features finales utilisées.

### Prédiction (`prediction/<PID>/`)
- `<PID>_u_Txxxj.nii.gz` : densité 3D finale (continu 0–1),
- `<PID>_mask_Txxxj.nii.gz` : masque binaire 3D final (densité seuillée),
- `<PID>_u_4D.nii.gz` : série temporelle 4D de densité (X, Y, Z, T),
- `<PID>_mask_4D.nii.gz` : série temporelle 4D binaire (X, Y, Z, T),
- `<PID>_4D_timepoints.csv` : correspondance frame → jour,
- `<PID>_tissue_map.nii.gz` : carte de tissus (1=CSF, 2=GM, 3=WM), si T1 disponible,
- `<PID>_volumes_sim.csv` : évolution volumique simulée,
- `simulation_volume.png` : courbe simulation vs points cliniques disponibles.

Options utiles :
- `--bin_threshold 0.5` : seuil de binarisation du masque,
- `--n_snapshots 20` : nombre de frames 3D dans la série 4D (`0` désactive la 4D).

### Modélisation spatiale GM/WM/CSF (modèle de Swanson)

Si un T1 baseline est disponible, le cerveau est segmenté en CSF/GM/WM (K-means sur
l'intensité T1) et la diffusion devient **hétérogène** :

\[
\partial u / \partial t = \nabla\cdot(D(x)\,\nabla u) + r(x)\,u(1-u) - \alpha(x)\,c\,u
\]

La matière blanche diffuse plus vite que la matière grise (autoroutes axonales), le CSF
est fortement réduit. Options :
- `--d_wm_gm_ratio 5` : ratio diffusion WM/GM (défaut 5),
- `--r_wm_gm_ratio 1` : ratio prolifération WM/GM,
- `--alpha_wm_gm_ratio 1` : ratio réponse au traitement WM/GM,
- `--csf_factor 0.1` : réduction diffusion/prolifération dans le CSF,
- `--no_tissue` : désactive et revient à la simulation homogène classique.

Repli automatique sur le mode homogène si le T1 est absent ou de dimensions
incompatibles avec la grille de simulation.

---

## 9) Détails de configuration utiles

- Le pipeline RF lit d’abord dans `features/`, puis dans la racine.
- Le fallback “sphère synthétique” est utilisé si le masque de segmentation baseline est absent.
- `Dose_Gy` = radiothérapie (pas la dose de chimio).
- `Prior_Chemo` = indicateur binaire (oui/non), pas une dose continue.

---

## 10) Dépannage

### `ModuleNotFoundError: openpyxl`

Installer les dépendances dans le venv :

```powershell
.\EnvVirtuel\Scripts\activate
pip install -r .\requirements.txt
```

### Le patient n’est pas trouvé en prédiction

- vérifier l’ID (`P03`, `P04a`, etc.),
- vérifier la présence de `17253793/<PID>/`,
- vérifier les masques dans `tumor_segmentation/`.

### Courbe simulée “très mauvaise”

Causes fréquentes :
- patient hors distribution d’entraînement,
- timepoints réels irréguliers vs hypothèse `dt_days` fixe,
- qualité/consistance des segmentations longitudinales.

---

## 11) Limitations connues

- Pas d’intégration explicite matière grise/blanche dans les features RF actuelles.
- Pas de localisation anatomique explicite (lobe/atlas) dans la matrice finale.
- Certaines simplifications temporelles (espacement de suivi).
- Variabilité inter-patient élevée et petit effectif.

---

## 12) Roadmap suggérée

- intégrer des features de localisation anatomique,
- utiliser les dates réelles DICOM pour l’axe temporel,
- renforcer l’évaluation (split temporel/patient, calibration, incertitude),
- harmoniser l’organisation des scripts (suppression des doublons racine vs sous-dossiers).

---

## 13) Contribuer

1. Créer une branche de travail.
2. Lancer les scripts dans l’ordre du workflow.
3. Vérifier que les sorties sont générées dans les bons dossiers (`features/`, `results_rfml/`, `prediction/`).
4. Ouvrir une PR avec :
   - contexte du changement,
   - scripts testés,
   - exemples de sorties.

---

## 14) Références

- Dataset : [Zenodo DOI 10.5281/zenodo.17253793](https://doi.org/10.5281/zenodo.17253793)
- Modèle de croissance tumoral inspiré de Fisher-KPP.
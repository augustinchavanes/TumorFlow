# TumorFlow
* Création d'un environnement virtuel

Créer un environnement virtuel permet d'éviter les conflits de versions des bilbiothèques Python. 

Etape n1: 
``` 
python -m venv EnvVirtuel
```
Etape n2:
```
.\EnvVirtuel\Scripts\activate
```
Etape n3:
```
pip install -r .\requirements.txt
```

* Téléchargement du Dataset

Pour telecharger le dataset il suffit de se rendre sur le site web suivant:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17253793.svg)](https://doi.org/10.5281/zenodo.17253793)

Et ensuite, télécharger le dossier complet.
![Statistiques du dataset](dataset.png)

Une fois le dossier téléchargé, veuillez le dézipper dans le dossier de travail.

## Description du dataset

Ce dataset est une étude de cas longitudinale (suivi dans le temps) d'une quarantaine de patients atteints d'une tumeur cérébrale, combinant imagerie multimodale et données de planification thérapeutique.

### Organisation des fichiers pour un patient

#### 1.Imagerie de Référence (Dossier `BraTS`)
Ce dossier contient des images IRM traitées selon les standards du défi **BraTS** (Brain Tumor Segmentation). Le suivi est divisé en **5 étapes temporelles** :
* **`baseline`** : État initial au diagnostic (avant traitement majeur).
* **`fu1` à `fu4`** : Quatre examens de suivi (*Follow-up*) permettant d'observer l'évolution de la pathologie.

**Séquences disponibles pour chaque étape :**
* **T1 & T1c** : Visualisation de la structure anatomique et prise de contraste (rehaussement de la tumeur).
* **T2 & FLAIR** : Identification des zones d'œdème et d'infiltration tumorale.

#### 2.Données de Segmentation (`tumor segmentation`)
C'est la partie la plus précieuse du dataset, contenant les **masques de vérité terrain** (*ground truth*) :
* **Localisation** : Fichiers `.nii` délimitant précisément les contours de la tumeur sur les images IRM.
* **Utilité** : Indispensable pour entraîner ou valider des algorithmes d'IA (Deep Learning) afin de mesurer automatiquement le volume tumoral.
* **Évolution** : La présence de fichiers pour chaque `fu` permet de quantifier la croissance ou la régression de la tumeur sous l'effet du traitement.

#### 3.Planification Thérapeutique (`RTP` & `CT`)
Le dataset inclut des données liées à la stratégie de **radiothérapie** :
* **`P01_CT.nii`** : Scanner X (Computed Tomography) utilisé pour calculer la densité électronique des tissus.
* **`P01_RTP.nii`** : Plan de traitement (*Radiotherapy Plan*) définissant la zone cible à irradier, tracée par les oncologues.
* **`P01_brain_mask.nii`** : Segmentation du cerveau entier utilisée pour isoler l'organe des tissus environnants (os, muscles).

#### 4.Données Sources (`DICOM`)
Le dossier `DICOM` contient les fichiers bruts originaux non transformés, classés par dates de capture. C'est la source primaire à partir de laquelle les fichiers NIfTI (`.nii.gz`) de ce projet ont été générés.
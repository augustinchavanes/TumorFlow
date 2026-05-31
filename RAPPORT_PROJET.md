# TumorFlow — Rapport de projet

**Prédiction des paramètres d'une EDP de croissance tumorale par apprentissage automatique, puis simulation 3D de l'évolution de métastases cérébrales sous traitement.**

---

## 1. Le dataset de base

### 1.1 Source
Le projet s'appuie sur le jeu de données **PROTEAS** (archive Zenodo `17253793`), une cohorte de patients atteints de **métastases cérébrales** traitées par radiothérapie (et, pour certains, chimiothérapie préalable). Il rassemble trois familles de données :

| Famille | Fichier / dossier | Contenu |
|--------|-------------------|---------|
| **Cliniques / démographiques** | `PROTEAS-Clinical_and_demographic_data.xlsx` | Âge, scores de performance (WHO PS, Karnofsky), dose de radiothérapie, fractions, histologie, nombre de métastases, chimio antérieure, statut neurocognitif… |
| **Radiomiques** | `PROTEAS-MRI_radiomics_data.xlsx` | ~1 feuille Excel par patient ; centaines de caractéristiques de forme et d'intensité (PyRadiomics) extraites des IRM |
| **Imagerie volumétrique** | `17253793/P04a/`, `P04b/`, … (≈ 41 dossiers patients) | Séquences IRM (BraTS : `t1`, `t1c`, `t2`, `fla`), masques de segmentation tumorale, masque cérébral, CT, plan de radiothérapie (RTP) |

### 1.2 Structure par patient
```
17253793/<PatientID>/
├── BraTS/
│   ├── baseline/   fla.nii.gz  t1.nii.gz  t1c.nii.gz  t2.nii.gz
│   ├── fu1/ … fu4/                 ← suivis longitudinaux
├── tumor_segmentation/
│   └── <PID>_tumor_mask_baseline … fu4.nii.gz
├── <PID>_brain_mask.nii.gz
├── <PID>_CT.nii.gz
└── <PID>_RTP.nii.gz
```
Le caractère **longitudinal** (baseline + jusqu'à 4 suivis) est central : c'est lui qui permet de mesurer une **trajectoire de volume tumoral dans le temps**, donc d'estimer une dynamique de croissance.

### 1.3 Caractère « petit échantillon »
La cohorte est de l'ordre de **~40 patients** (certains avec deux lésions, suffixes `a`/`b`). C'est un point dimensionnant : tout le pipeline ML est conçu pour **éviter le sur-apprentissage sur très peu de données** (voir §5 et §6).

---

## 2. L'analyse de données réalisée

L'analyse exploratoire et le nettoyage sont concentrés dans `DataAnalyse/data_analyse.py`. Elle porte sur deux blocs.

### 2.1 Données cliniques — nettoyage rigoureux
- **Extraction numérique** de la dose : `"20 Gy"` → `20.0` (regex).
- **Audit des valeurs manquantes** colonne par colonne (avant/après).
- **Imputation raisonnée, médicalement justifiée** (et non un simple « remplir par la moyenne ») :
  - **Synchronisation WHO PS ↔ Karnofsky** via la table de conversion clinique standard
    (WHO 0→100 %, 1→85 %, 2→65 %, 3→45 %). Si l'un des deux scores manque, on le déduit de l'autre ; si les deux manquent, on impute par la médiane.
  - **Statut neurocognitif** manquant → catégorie explicite `"Unknown/Not Assessed"` (on ne fabrique pas une fausse valeur).
  - **Extension de la maladie crânienne** → imputation par le **mode** (valeur la plus fréquente).
- **Validation post-imputation** : comparaison moyenne/médiane/écart-type avant vs après pour vérifier que l'imputation **ne déforme pas les distributions**, puis sauvegarde d'un fichier `..._imputed.xlsx`.
- **Visualisations** : boxplots des variables cliniques + **matrice de corrélation** (`Matrice_Correlation_Cliniques.png`).

### 2.2 Données radiomiques
- Lecture **feuille par feuille** (un patient = une feuille), pivot `RadiomicsFeature` → colonnes, concaténation en un tableau unique.
- **Audit des valeurs manquantes** par patient et au global.
- Focus sur les **caractéristiques de forme** (`shape`) : boxplots + **matrice de corrélation** (`Matrice_Correlation_Caract_Formes.png`).

### 2.3 Conclusions tirées de l'analyse
Ces conclusions ont **directement orienté la modélisation** :
1. **Forte redondance / colinéarité** entre features radiomiques (surtout de forme) → inutile de toutes les garder, risque de bruit. ➜ on **réduit drastiquement** la matrice d'entrée.
2. **Données cliniques incomplètes mais récupérables** par règles métier → un petit nombre de variables cliniques fiables est exploitable (Âge, chimio antérieure…).
3. **Échantillon limité** → nécessité d'un modèle **simple et régularisé**, et d'une **vérité terrain dérivée de la physique** plutôt que purement statistique.
4. La **morphologie** de la tumeur (sphéricité, diamètre, élongation) et l'**intensité** portent une information sur l'agressivité/infiltration → candidates naturelles comme proxys des paramètres de l'EDP.

---

## 3. La démarche : des données vers les coefficients de l'EDP

Le cœur du problème : on ne dispose **pas** directement des coefficients physiques (D, r, α) ; il faut les **fabriquer** (vérité terrain) puis apprendre à les prédire depuis les features baseline. Pipeline en 4 temps.

### Étape A — Volumes longitudinaux
Pour chaque patient, on reconstruit la **trajectoire de volume tumoral** `V(t)` :
- **Priorité aux masques NIfTI réels** (`tumor_segmentation/`), volume = nb voxels × volume voxel.
- **Repli** sur les volumes du CSV radiomique si un masque manque.
- Le temps est exprimé en jours depuis le baseline : `t = tp_int × dt_days` (par défaut `dt_days = 56`, soit ~8 semaines entre deux IRM).

### Étape B — Vérité terrain : estimation des paramètres par problème inverse
Fonction `estimate_patient_parameters`. On **ajuste une ODE** (version 0D de l'EDP, sur le volume normalisé) à la trajectoire observée :
- Normalisation `u ∈ (0,1]` avec un plafond `V_cap = 2.5 × max(V)`.
- **`r`, `α`, `t_treat`** sont optimisés par **évolution différentielle** (`scipy.optimize.differential_evolution`) — un optimiseur global robuste aux minima locaux, adapté quand on a peu de points et un paysage non convexe.
- **`D`** n'est pas optimisé sur la courbe (un volume scalaire ne contraint pas bien la diffusion spatiale) : il est dérivé de la **morphologie** via une heuristique de type Swanson (`_estimate_D_from_shape`) à partir de la sphéricité, du diamètre et de l'élongation.
- On calcule un **R²** de l'ajustement et on marque le patient comme **valide (`success`)** si `R² > 0.3`. Seuls les patients valides alimentent le ML.
- Bonus physiques rapportés : vitesse de front `v = 2√(rD)` et largeur de front `λ = √(D/r)` (relations analytiques de Fisher-KPP).

### Étape C — Matrice de features (X)
Fonction `build_feature_matrix`. **Conclusion clé de l'analyse de données appliquée ici** : on ne garde que **6 variables** (le reste = bruit/colinéarité) :

| Feature | Rôle / paramètre visé |
|---------|----------------------|
| `t1c_Maximum3DDiameter` | étendue spatiale → **D** |
| `t1c_Sphericity` | régularité de forme → **r** |
| `t1c_VoxelValue_Mean` | proxy densité cellulaire |
| `ratio_T_O_Volume` | ratio tumeur/œdème (FLAIR) → infiltration |
| `Age` | covariable clinique |
| `Prior_Chemo` | chimio antérieure → **α** |

Les NaN restants sont imputés par la **médiane** de la colonne.

### Étape D — Apprentissage et prédiction
Voir §5 et §6 : un **Random Forest multi-sorties** apprend `X (6 features baseline) → (D, r, α)`, puis pour un nouveau patient on prédit les coefficients et on **simule l'EDP 3D**.

---

## 4. L'EDP choisie et pourquoi

### 4.1 Le modèle : Fisher-KPP avec terme de traitement
$$\frac{\partial u}{\partial t} = D\,\nabla^2 u \;+\; r\,u\,(1-u)\;-\;\alpha\,c(x,t)\,u$$

avec :
- `u(x,t)` ∈ [0,1] : densité **normalisée** de cellules tumorales,
- `D` [mm²/jour] : **diffusion/invasion** (migration cellulaire),
- `r` [1/jour] : **prolifération nette** (croissance logistique, saturation en `(1−u)`),
- `α` [1/jour] : **efficacité du traitement**,
- `c(x,t)` : concentration normalisée du médicament (décroissance exponentielle, demi-vie ≈ 14 j).

### 4.2 Justification du choix
- **Fisher-KPP = standard de référence** pour la croissance de gliomes/tumeurs cérébrales (modèle de Swanson). Il capture les **deux phénomènes dominants** : prolifération (terme réactionnel logistique) et **infiltration diffuse** dans le parenchyme (terme de diffusion) — ce qui correspond exactement au comportement infiltrant observé en imagerie.
- **Parcimonie / identifiabilité** : seulement 2–3 paramètres physiques interprétables, ce qui est **indispensable avec ~40 patients**. Un modèle plus riche serait non identifiable.
- **Liens analytiques exploitables** : `v = 2√(rD)`, `λ = √(D/r)` donnent des contrôles de cohérence physiques et des sorties cliniquement parlantes.
- **Terme de traitement séparable** : le facteur `−α·c(x,t)·u` permet de brancher/débrancher la chimio et d'en faire varier l'intensité (dose) sans changer le reste.

### 4.3 Deux niveaux de résolution
1. **0D (ODE sur le volume)** — utilisé pour la **vérité terrain** (étape B). À deux phases : prolifération seule avant `t_treat`, puis ajout du traitement. Intégré par `solve_ivp` (RK45).
2. **3D (EDP sur la grille voxel)** — utilisé pour la **simulation/prédiction** (`simulate_pde_3d`). La condition initiale est le **vrai masque tumoral baseline** (`u=1` dans la tumeur), pas une sphère synthétique (repli uniquement si le masque manque).

### 4.4 Extension spatiale (modèle de Swanson hétérogène)
Quand un **T1 baseline** est disponible, le cerveau est segmenté en **CSF / matière grise (GM) / matière blanche (WM)** par K-means (3 classes triées par intensité). On construit alors des **champs spatiaux** `D(x), r(x), α(x)` et l'EDP devient :
$$\frac{\partial u}{\partial t} = \nabla\!\cdot\!\big(D(x)\nabla u\big) + r(x)\,u(1-u) - \alpha(x)\,c\,u$$
La diffusion passe en **forme à flux** `∇·(D(x)∇u)` (schéma volumes finis, D moyenné aux faces) — indispensable dès que `D` varie dans l'espace. Convention : **GM = référence (×1)**, **WM = diffusion plus rapide** (ratio réglable, typiquement 5–10×), **CSF fortement réduit**.

---

## 5. La prédiction des coefficients (modèle ML)

### 5.1 Choix du modèle : Random Forest multi-sorties
- `RandomForestRegressor` encapsulé dans un `MultiOutputRegressor` → **un modèle par cible** (D, r, α), ce qui donne aussi des **importances de features par paramètre**.
- **Hyperparamètres délibérément régularisés** (anti sur-apprentissage sur petit échantillon) :
  - `n_estimators = 300` (compromis biais/variance),
  - `max_depth = 5` (arbres peu profonds),
  - `min_samples_leaf = 2`,
  - `max_features = 2` (splits simples, limite la mémorisation du bruit),
  - `oob_score = True` (score out-of-bag « gratuit »).
- Entrées **normalisées** (`StandardScaler`).

### 5.2 Évaluation honnête
La validation est conçue pour ne pas se mentir sur si peu de données :
- **Validation croisée K-fold** sur tout le jeu valide **et** sur le sous-ensemble train seul,
- **score OOB** par paramètre,
- **baseline « moyenne du train »** : on compare le R² du RF à celui d'un prédicteur constant ; si la baseline gagne, le modèle n'apporte rien,
- métriques **MAE / RMSE / R²** sur un set de test,
- **top features** par paramètre (Mean Decrease Impurity).

Artefacts sauvegardés : `pde_ml_model.pkl`, `metrics.csv`, `predictions_test.csv`.

### 5.3 De la prédiction à la simulation
Pour un nouveau patient (`--predict`) : extraction des features baseline → prédiction `(D, r, α[, t_treat])` → (option) champs tissulaires → `simulate_pde_3d` → sorties NIfTI 3D/4D (densité continue + masques binaires), courbe de volume, carte de tissus.

---

## 6. Les contraintes mises en place et comment on les a trouvées

C'est le point qui rend le modèle **physiquement plausible** malgré le peu de données. On distingue plusieurs familles de contraintes.

### 6.1 Bornes sur les paramètres (problème inverse)
Dans l'optimisation de la vérité terrain :
```
r       ∈ [0.005, 0.1]    /jour
alpha   ∈ [0.0, 0.2]      /jour
t_treat ∈ [0, 0.8·t_max]
```
- **Origine** : ordres de grandeur biologiques pour des temps de doublement réalistes (`r` borné évite des croissances explosives non physiologiques) ; `α ≥ 0` (un traitement ne fait pas grandir la tumeur) ; `t_treat ≤ 0.8·t_max` garantit qu'il reste **au moins un suivi après le début du traitement** pour pouvoir l'estimer.
- **Comment trouvées** : combinaison **littérature** (gammes de prolifération de gliomes/métastases) + **garde-fous numériques** calés sur l'horizon temporel propre à chaque patient.

### 6.2 Borne physique sur la diffusion D
`_estimate_D_from_shape` borne `D ∈ [0.001, 15] mm²/jour`, **étalonnage littérature (Swanson, 2000)**. La valeur dépend de l'irrégularité (`1−sphéricité`), de l'étendue (diamètre) et de l'anisotropie (`1−élongation`).

### 6.3 Contrainte de positivité à la prédiction
Les sorties du RF sont **bornées au plancher** avant simulation : `D ≥ 0.001`, `r ≥ 1e-5`, `α ≥ 0`, `t_treat ≥ 0`. Cela évite qu'un bruit de prédiction ne rende l'EDP instable ou non physique.

### 6.4 Contraintes du schéma numérique 3D
- **Stabilité (condition CFL)** : le pas de temps est choisi `dt = min(1, 0.9 · dx²/(6·D_max))`. **Origine** : critère de stabilité d'un schéma explicite pour la diffusion en 3D (le `6` = 6 voisins). C'est une **contrainte mathématique** imposée par le solveur explicite, pas un réglage libre.
- **Bornes de la densité** : `u` est **clampé dans [0,1]** à chaque pas (cohérent avec la normalisation et la saturation logistique).
- **Conditions aux limites de Neumann (flux nul)** aux bords du domaine : la tumeur ne « fuit » pas hors de la grille ; en forme à flux, le flux est annulé aux frontières (pas d'artefact périodique).

### 6.5 Contraintes anatomiques (champs tissulaires)
- **GM = référence (×1)**, **WM = ×`d_wm_gm_ratio`** (diffusion plus rapide en matière blanche, fait biologique bien établi : les cellules migrent le long des faisceaux), **CSF = ×`csf_factor`** (diffusion/prolifération quasi nulles dans le liquide).
- **Comment trouvées** : ce sont des **ratios issus de la littérature** (Swanson : WM ~5–10× GM), **réglables en ligne de commande** (`--d_wm_gm_ratio`, `--csf_factor`, …) car **non recalibrés** sur cette cohorte. Ils sont donc des **hypothèses explicites et ajustables**, pas des constantes cachées.
- Garde-fou de cohérence : la segmentation tissulaire n'est utilisée que si le T1 a **exactement la même grille** que la simulation (sinon repli sur simulation homogène).

### 6.6 Contraintes sur la donnée d'entraînement
- **Filtre qualité** : seuls les patients avec ajustement `R² > 0.3` entrent dans le ML (`success`).
- **Garde-fou d'effectif** : entraînement refusé si `< 6` patients valides (message explicite).
- **Réduction à 6 features** : contrainte anti-bruit/anti-colinéarité directement issue de l'analyse de données (§2.3).

### 6.7 Contraintes liées au traitement (protocole)
Dans la simulation 3D, le traitement est modélisé comme un **protocole multi-cycles** (plusieurs injections espacées, chacune décroissant exponentiellement avec la demi-vie médicament). Cela contraint la forme temporelle de `c(x,t)` à rester **cliniquement réaliste** plutôt qu'une dose unique permanente.

> Remarque de transparence : l'espacement des cycles dans le code de simulation (`i*70.0` jours) et le nombre de cycles (6) sont des paramètres de protocole codés en dur ; ils gagneraient à être exposés en CLI pour coller à un protocole patient précis.

---

## 7. Synthèse de la chaîne complète

```
IRM + masques + cliniques (PROTEAS)
        │
        ▼  (analyse + nettoyage : DataAnalyse/)
features fiables, peu redondantes
        │
        ▼  (volumes longitudinaux V(t))
problème inverse ODE Fisher-KPP  →  vérité terrain (D, r, α)  [bornes physiques]
        │
        ▼  (6 features baseline → 3 cibles)
Random Forest multi-sorties régularisé  →  (D, r, α) prédits  [validation CV/OOB/baseline]
        │
        ▼  (condition initiale = vrai masque ; champs GM/WM/CSF)
EDP Fisher-KPP 3D (flux form, CFL, Neumann, u∈[0,1])
        │
        ▼
densité 3D/4D, masques binaires, courbes de volume, visualisation 3D
```

---

## 8. Limites connues et pistes
- **Échantillon réduit (~40 patients)** : R² par paramètre à interpréter avec prudence ; la robustesse vient surtout de la régularisation et des contraintes physiques.
- **`dt_days` fixe (56 j)** : approxime l'espacement réel des IRM ; les vrais intervalles cliniques amélioreraient la vérité terrain.
- **Ratios tissulaires non calibrés** sur la cohorte (valeurs littérature) → à recaler si données disponibles.
- **D non optimisé sur la dynamique** (dérivé de la forme) : une calibration spatiale (fronts FLAIR/T1c) serait plus fidèle.
- **Protocole de traitement** partiellement codé en dur dans la simulation.
- **Le patient prédit doit idéalement être hors du train** ; sur si peu de patients, une prédiction médiocre (ex. P03) s'explique surtout par le faible effectif et les écarts de timing.

---

*Document généré à partir du code réel du dépôt : `random_forest_pipeline.py`, `DataAnalyse/data_analyse.py`, et les scripts d'extraction de `features/`.*

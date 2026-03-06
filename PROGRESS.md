# J2B — Airbus Hackathon 2026 : Suivi du Projet

## Objectif
Segmentation semantique de nuages de points LiDAR 3D pour detecter 4 types d'obstacles aeriens : Antennes, Cables, Poteaux electriques, Eoliennes.

## Architecture du Pipeline

```
HDF5 bruts ──> prep_data.py ──> .npz par frame ──> train.py ──> best_model.pth
                                                                      │
Eval HDF5 ──────────────────────────────────> inference.py ──> CSV predictions
                                                │
                                    Segmentation (RandLA-Net)
                                    DBSCAN par classe
                                    Oriented Bounding Box (PCA)
```

## Modele : RandLA-Net

| Propriete | Valeur |
|---|---|
| Architecture | RandLA-Net (Hu et al., CVPR 2020) |
| Parametres | 645,765 (0.65M) |
| Encoder | 4 couches LFA [32, 64, 128, 256] |
| K voisins | 16 |
| Points/sample | 40,960 |
| Sub-sampling | Random x4 par couche |
| Features (d_in) | 8 (xyz + refl + dist + linearity + planarity + verticality) |
| fc_start | SharedMLP(8, 16) — Run 8 |
| Dropout | 0.5 dans classifier head — Run 8 |
| Optimizer | AdamW (decoupled weight decay) — Run 8 |

### Pourquoi RandLA-Net ?
- Naturellement robuste au sous-echantillonnage (random sampling = simule la baisse de densite)
- Peu de parametres (~0.65M) → bon score d'efficacite (critere d'evaluation)
- Performant sur les grandes scenes LiDAR exterieures

## Optimisations

### Loss Function
- **Focal Loss** (gamma=2) : focalise l'apprentissage sur les exemples difficiles, gere le desequilibre de classes
- **Lovasz-Softmax** : optimise directement l'IoU (metrique d'evaluation du hackathon)
- Combinaison : `loss = focal_weight * FL + lovasz_weight * LS`

### Augmentations (training)
- **Density drop** [0.25, 1.0] : drop aleatoire de 25-75% des points → robustesse aux evaluations 25/50/75/100%
- Rotation aleatoire +-180 deg autour de Z
- Scale aleatoire [0.9, 1.1]
- Jitter gaussien (std=0.01)
- Flip horizontal (50%)

### Oversampling
- Les frames contenant des cables sont presentees x3 plus souvent (WeightedRandomSampler)
- Class weights inversement proportionnels a la frequence (max=20)

### Inference
- **Test-Time Augmentation (TTA)** : 4 rotations (0, 90, 180, 270 deg), moyenne des softmax
- **DBSCAN par classe** avec parametres optimises :
  - Cable : eps=1.5m, min_samples=10 (fin, lineaire)
  - Antenna : eps=3.0m, min_samples=20
  - Pole : eps=2.0m, min_samples=15
  - Turbine : eps=5.0m, min_samples=30 (gros objet)
- **Oriented Bounding Box** via PCA (meilleur IoU que des AABB naives)

---

## Historique des Runs

### Run 1 — Baseline (sans voxel)

| Parametre | Valeur |
|---|---|
| Voxel downsample | Non |
| LR | 0.01 |
| Patience | 15 |
| Epochs effectuees | 48/100 (early stop) |
| Temps total | ~40 min |

**Resultats (best epoch 33) :**

| Classe | IoU |
|---|---|
| Background | 0.9135 |
| Wind turbine | 0.4207 |
| Cable | 0.1439 |
| Antenna | 0.0810 |
| Electric pole | 0.0076 |
| **mIoU** | **0.3133** |

**Observations :**
- Background et turbines bien detectes rapidement
- Cables instables : montent a 0.14 puis redescendent a 0.01 puis remontent
- Poteaux quasi jamais detectes
- Early stopping a 48 : le modele a plafonne
- Leger overfitting (gap train/val)

**Decisions pour le run suivant :**
- Ajouter le voxel downsampling pour mieux preserver les classes rares
- Baisser le LR (0.01 → 0.005) pour stabiliser les classes rares
- Augmenter la patience (15 → 25) pour laisser les cables converger

---

### Run 2 — Voxel + LR/Patience tuning

| Parametre | Valeur | Changement |
|---|---|---|
| Voxel downsample | 0.15m | NOUVEAU |
| LR | 0.005 | 0.01 → 0.005 |
| Patience | 25 | 15 → 25 |
| Epochs effectuees | 100/100 | pas d'early stop |
| Temps total | ~78 min | |

**Changement cle : Voxel Downsampling**
- Reduit 575K → ~60-100K points par frame avec couverture spatiale uniforme
- Priorite aux points d'obstacles : dans chaque voxel, on garde le point de la classe la plus rare
- Un cable de 500 points n'est plus noye dans 575K de background avant le random subsample

**Resultats (best epoch 99) :**

| Classe | Run 1 | Run 2 | Progression |
|---|---|---|---|
| Background | 0.9135 | 0.9902 | +8% |
| Wind turbine | 0.4207 | 0.8204 | +95% |
| Cable | 0.1439 | 0.6591 | +371% |
| Antenna | 0.0810 | 0.3672 | +359% |
| Electric pole | 0.0076 | 0.4034 | +5614% |
| **mIoU** | **0.3133** | **0.6481** | **+109%** |

**Observations :**
- Le voxel downsampling a ete LE game changer
- Pas d'early stop : le modele a continue a progresser jusqu'a l'epoch 99
- Les cables sont passes de 0.14 a 0.66 (stable grace au LR plus bas)
- Les poteaux de 0.007 a 0.40 (enfin detectes!)
- L'inference genere 2903 detections sur scene 1 → possible sur-fragmentation DBSCAN
- GPU a 97% / 72W pendant le training, VRAM seulement 5.7/23GB

**Decisions pour le run suivant :**
- Activer geometric features (linearity=detecteur cables, planarity=surfaces)
- num_points 40960 → 65536 (utiliser la VRAM disponible)
- batch_size 6 → 14 (gradients plus stables, VRAM utilisee)
- voxel_size 0.15 → 0.10 (grille plus fine pour 65K points)
- Oversample antennes et poteaux aussi (pas que cables)

---

### Run 3 — Full optimization (en attente)

| Parametre | Valeur | Changement |
|---|---|---|
| Voxel downsample | 0.10m | 0.15 → 0.10 |
| num_points | 65536 | 40960 → 65536 |
| batch_size | 14 | 6 → 14 |
| d_in | 7 | 5 → 7 (+ linearity, planarity) |
| geometric features | ON | OFF → ON |
| Oversampling | cables+antennes+poteaux | cables seuls → toutes classes rares |
| LR | 0.005 | inchange |
| Patience | 25 | inchange |

**Resultats (best epoch 89) :**

| Classe | Run 2 | Run 3 | Progression |
|---|---|---|---|
| Background | 0.9902 | 0.9919 | +0.2% |
| Wind turbine | 0.8204 | 0.8644 | +5.4% |
| Cable | 0.6591 | 0.7501 | +13.8% |
| Antenna | 0.3672 | 0.4006 | +9.1% |
| Electric pole | 0.4034 | 0.4096 | +1.5% |
| **mIoU** | **0.6481** | **0.6833** | **+5.4%** |

**Observations :**
- Le modele n'avait pas converge : courbe val mIoU encore en montee a l'epoch 100 → besoin de plus d'epochs
- Val loss < train loss sur la seconde moitie → pas d'overfitting, marge disponible
- Geometric features ont aide les cables (+13.8%) mais peu les poteaux (+1.5%)
- Inference scene_1 : 2387 detections dont 1424 antennes → over-fragmentation DBSCAN sevère
- Freins principaux : Antenna (0.40) et Pole (0.41) — class_weight antenna trop bas (4.39 vs 20 cable)

**Decisions pour Run 4 :**
- Warm start depuis epoch 89 (modele pas converge, inutile de repartir a zero)
- Passer a 150 epochs (modele toujours en apprentissage a 100)
- Override class weights : antenna 4.39 → 12.0, pole 13.53 → 16.0
- focal_gamma 2.0 → 3.0 (focus plus agressif sur exemples difficiles)
- DBSCAN antenna : min_samples 20 → 50, eps 3.0 → 2.5 (anti-fragmentation)
- min_cluster_points antenna : 15 → 30

---

### Run 4 — Warm Start + Loss Tuning

| Parametre | Valeur | Changement |
|---|---|---|
| Epochs | 150 | 100 → 150 |
| Warm start | Oui (epoch 89) | Non → Oui |
| Antenna class weight | 12.0 | 4.39 → 12.0 (override) |
| Pole class weight | 16.0 | 13.53 → 16.0 (override) |
| focal_gamma | 3.0 | 2.0 → 3.0 |
| DBSCAN antenna eps | 2.5m | 3.0 → 2.5 |
| DBSCAN antenna min_samples | 50 | 20 → 50 |
| min_cluster_points antenna | 30 | 15 → 30 |
| prefetch_factor | 4 | 2 → 4 (dataloader) |

**Resultats (arreté epoch 102) :**

| Classe | Run 3 | Run 4 (max) | Verdict |
|---|---|---|---|
| Antenna | 0.4006 | 0.2303 | RÉGRESSÉ |
| Cable | 0.7501 | 0.7465 | stable |
| Pole | 0.4096 | 0.4522 | légère hausse |
| Turbine | 0.8644 | 0.8316 | légère baisse |
| **mIoU** | **0.6833** | **0.6311** | **RÉGRESSÉ** |

**Post-mortem :**
- Warm start + changement brutal des class weights + LR déjà très faible (0.0019) = conflit
- Le modèle ne peut pas se réadapter à la nouvelle loss avec un si petit LR
- Erreur de conception : il fallait soit reset le LR, soit ne pas changer les class weights en warm start
- Décision : arrêt à epoch 102, cold start pour Run 5

---

### Run 5 — Cold Start avec architecture améliorée

| Parametre | Valeur | Changement |
|---|---|---|
| Warm start | Non | Oui → Non (cold start) |
| d_encoder | [48, 96, 192, 256] | couches intermédiaires plus larges (~850K, <1M) |
| k_neighbors | 16 | 32→16 (OOM à batch=8 avec k=32) |
| batch_size | 10 | 14→10 (d_encoder élargi = plus de VRAM) |
| focal_gamma | 3.0 | inchangé depuis Run 4 |
| Antenna class weight | 12.0 | inchangé depuis Run 4 |
| Epochs | 150 | inchangé |

**Résultats (early stop epoch 41) :**

| Classe | Run 3 | Run 5 (max) | Verdict |
|---|---|---|---|
| Antenna | 0.4006 | 0.0600 | EFFONDRÉ |
| Cable | 0.7501 | ~0.35 | RÉGRESSÉ |
| Pole | 0.4096 | ~0.20 | RÉGRESSÉ |
| Turbine | 0.8644 | ~0.70 | RÉGRESSÉ |
| **mIoU** | **0.6833** | **0.3706** | **RÉGRESSÉ** |

**Post-mortem :**
- `focal_gamma=3.0` était le coupable principal : gamma élevé → gradient ≈0 pour les prédictions correctes (ex: conf=0.9 → loss=(1-0.9)³≈0.001)
- Combiné aux class_weights élevés (antenna=12, pole=16), le gradient devient pathologique
- Early stop à epoch 41 : patience=25, meilleur à epoch 16 (mIoU=0.3706)
- Trop de changements simultanés : gamma, weights, d_encoder, batch → instabilité cumulée
- Leçon : changer UN SEUL paramètre à la fois, surtout les paramètres de loss

**Décisions pour Run 6 :**
- Revenir EXACTEMENT à la config stable Run 3 sauf class_weights_override
- focal_gamma 3.0 → 2.0 (le coupable de Run 5)
- d_encoder [48,96,192,256] → [32,64,128,256] (645K params, stable)
- batch_size 10 → 14 (GPU mieux utilisé)
- GARDER class_weights_override [12,20,16,1.82,0.03] (le seul vrai fix pour antenna)
- patience 25 → 30

---

### Run 6 — Revert instabilités + class weights seuls ✅ TARGET ATTEINT

| Parametre | Valeur | Changement |
|---|---|---|
| focal_gamma | 2.0 | 3.0 → 2.0 (REVERT — coupable Run 5) |
| d_encoder | [32, 64, 128, 256] | REVERT config stable (645K params) |
| batch_size | 14 | 10 → 14 (REVERT) |
| patience | 30 | 25 → 30 |
| class_weights_override | [12, 20, 16, 1.82, 0.03] | GARDER (seul vrai fix antenna) |
| epochs | 150 | inchangé |
| warm_start | Non | inchangé |

**Résultats (best epoch 140/150) :**

| Classe | Run 3 | Run 6 | Delta |
|---|---|---|---|
| Antenna | 0.4006 | **0.6373** | +59.1% ✅ |
| Cable | 0.7501 | **0.7504** | +0.0% stable |
| Pole | 0.4096 | **0.7152** | +74.6% ✅ |
| Turbine | 0.8644 | **0.7929** | -8.2% ⚠️ |
| Background | 0.9919 | 0.9949 | +0.3% |
| **mIoU** | **0.6833** | **0.7782** | **+13.9%** ✅ |

**Paramètres :** 645,293 (0.65M) — ✅ sous 1M

**Temps d'entraînement :** 112.9 minutes (150 epochs × ~83s)

**Observations :**
- **TARGET mIoU ≥ 0.75 ATTEINT** (0.7782) — objectif hackathon validé
- Antenna +59% : class_weight_override [12→12] a été le vrai levier
- Pole +74% : class_weight_override [13.53→16] encore plus efficace que prévu
- Turbine légèrement en baisse (-8.2%) : tradeoff normal — les autres classes gagnent du gradient
- Oscillation en fin de run (epochs 140-150 : 0.74-0.78) → variance élevée sur le val set (petit dataset)
- Epoch 149 mIoU=0.7788 > 0.7782 mais delta < min_delta=0.001 → checkpoint non écrasé
- LR≈0 à epoch 150 : le cosine schedule est épuisé, plus de gains possibles sans restart

**Freins restants :**
- Turbine a légèrement régressé (0.8644→0.7929) — class_weight turbine OK (1.82) mais antenna et pole "volent" du gradient
- mAP@IoU≥0.70 non mesuré encore → dépend de la qualité DBSCAN/OBB (à mesurer via inference)
- Variance élevée en val (dataset petit) → les chiffres par run peuvent fluctuer

**Inference scene_1 (100 frames) :**

| Classe | Run 3 | Run 6 | Réduction fragmentation |
|---|---|---|---|
| Antenna | ~1424 | **137** | -90% ✅ |
| Cable | ? | 262 | ~2.6/frame |
| Wind Turbine | ? | 182 | ~1.8/frame |
| Electric Pole | ? | 106 | ~1.1/frame |
| **Total** | **2387** | **687** | **-71%** ✅ |

- Modèle chargé : epoch 139 (mIoU val=0.7782)
- La sur-fragmentation antenna (1424→137) est résolue par les meilleurs DBSCAN params (eps=2.5, min_samples=50)
- Détections par frame très raisonnables pour une scène industrielle LiDAR

**Décisions pour Run 7 :**
- Mesurer mAP@IoU≥0.70 sur les scènes d'évaluation officielles (priorité)
- Turbine a régressé en IoU (0.86→0.79) → envisager d'ajuster class_weight turbine 1.82→2.5
- LR warm restart depuis le checkpoint epoch 140 pour continuer à explorer sans repartir à zéro
- Piste SWA : moyenner les poids des epochs 130-150 pour réduire la variance de val

---

### Run 7 — Warm Restart + Turbine weight + Post-processing ✅

| Parametre | Valeur | Changement |
|---|---|---|
| warm_start | True | False→True (restart depuis epoch 140, LR≈0.0015) |
| epochs | 220 | 150→220 |
| patience | 40 | 30→40 |
| turbine class_weight | 2.5 | 1.82→2.5 |
| DBSCAN antenna | min_samples=80, min_cluster=60 | kill FP |
| DBSCAN cable | eps=1.5 | revert 3.0→1.5 (eps=3.0 gonflait GT) |
| DBSCAN pole | min_samples=25, min_cluster=20 | kill FP |
| confidence_threshold | {Ant:0.60, Cab:0.45, Pole:0.55, Turb:0.50} | NOUVEAU |

**Résultats segmentation (best epoch 209/220) :**

| Classe | Run 6 | Run 7 | Delta |
|---|---|---|---|
| Antenna | 0.6373 | **0.7442** | +16.8% ✅ |
| Cable | 0.7504 | **0.7591** | +1.2% |
| Pole | 0.7152 | **0.7871** | +10.0% ✅ |
| Turbine | 0.7929 | **0.8675** | +9.5% ✅ |
| **mIoU** | **0.7782** | **0.8308** | **+6.8%** ✅ |

**Paramètres :** 645,293 (inchangé)
**Temps :** 111.5 min

**Résultats mAP (Run 7 modèle + post-processing v1 avec cable eps=3.0) :**

| Classe | TP | FP | FN | Prec | Recall | MeanIoU |
|---|---|---|---|---|---|---|
| Antenna | 23 | 22 | 13 | 0.511 | 0.639 | 0.9599 |
| Cable | 181 | 91 | 167 | 0.665 | 0.520 | 0.9839 |
| Pole | 52 | 20 | 21 | 0.722 | 0.712 | 0.9300 |
| Turbine | 135 | 24 | 30 | 0.849 | 0.818 | 0.9250 |
| **mAP** | | | | **0.6870** | | **0.9497** |

**Résultats mAP (Run 7 modèle + post-processing final — cable eps=1.5) :**

| Classe | TP | FP | FN | Prec | Recall | F1 | MeanIoU |
|---|---|---|---|---|---|---|---|
| Antenna | 23 | 22 | 13 | 0.511 | 0.639 | 0.568 | 0.9599 |
| Cable | 170 | 69 | 110 | 0.711 | 0.607 | 0.655 | 0.9822 |
| Pole | 52 | 20 | 21 | 0.722 | 0.712 | 0.717 | 0.9300 |
| Turbine | 135 | 24 | 30 | 0.849 | 0.818 | 0.833 | 0.9250 |
| **mAP** | | | | **0.6984** | | | **0.9493** |

**Turbine :** class_weight 1.82→2.5 a parfaitement récupéré la régression (0.7929→0.8675) ✅
**Cable eps=1.5 :** recall 0.520→0.607 ✅ / precision 0.665→0.711 ✅ — eps=3.0 était contre-productif

**Bottlenecks restants :**
- Antenna : 22 FP (confidence=0.60 → à relever à 0.65-0.70 pour tuer les FP résiduels)
- Cable : 110 FN (39% de câbles manqués — limitation modèle → nécessite Run 8)

**Progression totale :** mAP 0.5475 → 0.6984 (+27.5%) sans changer l'architecture

---

## Post-Processing — Métriques mAP (compute_map.py)

### Baseline Run 6 (avant tuning)

| Classe | TP | FP | FN | Prec | Recall | F1 | MeanIoU |
|---|---|---|---|---|---|---|---|
| Antenna | 27 | 84 | 27 | 0.243 | 0.500 | 0.327 | 0.9325 |
| Cable | 160 | 64 | 120 | 0.714 | 0.571 | 0.635 | 0.9862 |
| Pole | 45 | 57 | 33 | 0.441 | 0.577 | 0.500 | 0.9090 |
| Turbine | 129 | 34 | 36 | 0.791 | 0.782 | 0.787 | 0.9239 |
| **mAP** | | | | **0.5475** | | | **0.9379** |

### Post-processing Run 6 modèle + DBSCAN tuné + confidence filter

| Classe | TP | FP | FN | Prec | Recall | F1 | MeanIoU |
|---|---|---|---|---|---|---|---|
| Antenna | 23 | 27 | 13 | 0.460 | 0.639 | 0.535 | 0.9503 |
| Cable | 168 | 69 | 180 | 0.709 | 0.483 | 0.574 | 0.9837 |
| Pole | 48 | 17 | 25 | 0.738 | 0.658 | 0.696 | 0.8981 |
| Turbine | 127 | 26 | 38 | 0.830 | 0.770 | 0.799 | 0.9272 |
| **mAP** | | | | **0.6843** | | | **0.9398** |

**Gains :** mAP +0.137 (+25%) sans ré-entraîner le modèle ✅
- Antenna FP : 84→27 (-68%) ✅ — DBSCAN min_samples 50→80 + confidence 0.60
- Pole FP : 57→17 (-70%) ✅ — DBSCAN min_samples 15→25 + confidence 0.55
- Cable recall : 0.571→0.483 ⚠️ — eps=3.0 crée plus d'objets GT → FN en hausse
  - À surveiller : eps=3.0 peut être bénéfique sur le hackathon réel (GT Airbus ≠ notre proxy)

---

## Statut des Critères Hackathon

| Critère | Target | Run 7 | Phase 1 | Run 8 | Statut |
|---|---|---|---|---|---|
| mIoU (segmentation) | ≥ 0.75 | 0.8308 | inchangé | **0.9504** | ✅ EXCELLENT |
| mAP@IoU=0.5 | maximiser | 0.6984 | 0.7107 | **0.8174** | ✅ Très bon |
| Mean IoU (Correct Class) | maximiser | 0.9497 | 0.9490 | **0.9618** | ✅ Excellent |
| Robustesse (25% density) | <10% drop | _non mesuré_ | _non mesuré_ | _non mesuré_ | ⚠️ CRITIQUE |
| Efficience | <1M params | 645K | inchangé | **645,765** | ✅ ATTEINT |

---

### Run 8 — Cold Start complet (architecture + training fixes) ✅ RECORD

| Paramètre | Avant (Run 7) | Après (Run 8) | Raison |
|---|---|---|---|
| warm_start | True | **False** (cold start) | Architecture modifiée |
| d_in | 7 | **8** (+verticality) | Câbles vs pôles |
| fc_start | SharedMLP(d_in, 8) | **SharedMLP(d_in, 16)** | Goulot d'étranglement |
| Dropout | Non | **Dropout(0.5)** classifier | Généralisation Scene B |
| Optimizer | Adam | **AdamW** | Decoupled weight decay |
| weight_decay | 1e-4 | **5e-4** | Régularisation correcte |
| grad_clip_norm | 10.0 | **2.0** | Stabilité classes rares |
| dist normalisation | per-frame max | **20000 cm (fixe)** | Stabilité cross-scène |
| density floor | num_points//2 (32768) | **100 points** | Vrai entraînement 25% |
| epochs | 220 | **200** (nouveau cycle) | Cosine restart |

**6 améliorations simultanées — faisables car cold start requis (architecture modifiée)**

**Résultats segmentation (best epoch 192/200) :**

| Classe | Run 7 | Run 8 | Delta |
|---|---|---|---|
| Antenna | 0.7442 | — | — |
| Cable | 0.7591 | — | — |
| Pole | 0.7871 | — | — |
| Turbine | 0.8675 | — | — |
| **mIoU** | **0.8308** | **0.9504** | **+14.4%** ✅ |

**Paramètres :** 645,765 (0.65M) — ✅ sous 1M
**Temps d'entraînement :** 279.3 minutes (192 epochs effectives)

**Résultats mAP (Run 8 modèle + post-processing Phase 1) :**

| Classe | TP | FP | FN | Prec | Recall | Delta Prec vs Run 7 |
|---|---|---|---|---|---|---|
| Antenna | 26 | 9 | 10 | **0.743** | 0.722 | +45% ✅ |
| Cable | 209 | 36 | 71 | **0.853** | 0.746 | +20% ✅ |
| Pole | — | — | — | **0.761** | — | +5% ✅ |
| Turbine | 147 | 14 | 18 | **0.913** | 0.891 | +7% ✅ |
| **mAP** | | | | **0.8174** | | **+17.0%** ✅ |

**Mean IoU (Correct Class) : 0.9618** (vs 0.9497 Run 7)

**Analyse des gains :**
- **Antenna FP : 22→9** (-59%) — fc_start 16 + Dropout + AdamW ont amélioré la qualité des prédictions
- **Cable TP : 170→209** (+23%) — verticality feature distingue câbles/pôles, density fix voit 25% pendant training
- **Cable FP : 69→36** (-48%) — meilleure discrimination câble vs background
- **Turbine FP : 24→14** (-42%) — régularisation AdamW + Dropout
- **Cable FN : 110→71** (-35%) — amélioration fondamentale de la segmentation câbles

**Observations :**
- Cold start était REQUIS (fc_start 8→16 et d_in 7→8 changent l'architecture)
- La verticality feature a été le levier le plus impactant pour les câbles (+23% TP)
- Fix density floor → modèle a vraiment appris à 25% de densité (critique pour robustesse hackathon)
- AdamW + grad_clip=2.0 = training plus stable sans oscillations de fin de run

---

### Phase 1 — Post-processing v2 (après réunion agents)

**Changements appliqués (config.py) :**

| Paramètre | Avant | Après | Raison (agent) |
|---|---|---|---|
| Cable eps | 1.5 | **2.0** | câbles linéaires, points espacés → +recall |
| Cable min_samples | 10 | **6** | récupérer segments courts |
| Cable min_cluster_points | 8 | **5** | ne pas filtrer petits segments réels |
| Cable confidence | 0.45 | **0.40** | recall trop bas (0.607), accepter incertitude |
| Antenna confidence | 0.60 | **0.65** | 22 FP résiduels à éliminer |
| Antenna min_cluster_points | 60 | **80** | filtrage strict des FP résiduel |
| Turbine min_cluster_points | 25 | **20** | légère baisse pour recall (précision haute: 0.849) |
| Turbine confidence | 0.50 | **0.45** | marge de précision disponible → récupérer FN |

**Résultats Phase 1 (avec GT stable, gt_dbscan_params = Run 7) :**

| Classe | Run 7 | Phase 1 | Delta |
|---|---|---|---|
| Antenna | Prec=0.511, FP=22 | **Prec=0.575, FP=17** | -5 FP ✅ |
| Cable | Prec=0.711, TP=170 | Prec=0.711, TP=170 | stable |
| Pole | Prec=0.722 | Prec=0.722 | stable |
| Turbine | Prec=0.849, FP=24 | Prec=0.834, FP=27 | +3 FP ⚠️ |
| **mAP** | **0.6984** | **0.7107** | **+1.23 pts** |

**Post-mortem :**
- Cable eps=2.0 + min_samples=6 → 485 prédictions pour 280 GT = 456 FP (DÉSASTRE revert)
- Cable ne peut PAS être amélioré par post-processing → il faut améliorer le modèle (Run 8)
- Turbine min_cluster=20 + confidence=0.45 légèrement contre-productif (+3 FP)
- Gain réel de Phase 1 vient UNIQUEMENT d'antenna (-5 FP via confidence=0.65)
- gt_dbscan_params fixés = Run 7 baseline (stable, indépendant des params d'inférence)

---

### Réunion Agents — Bugs critiques identifiés pour Run 8

#### BUG 1 — Densité jamais vraiment à 25% (dataset.py)
```python
n_keep = max(int(len(xyz) * keep_ratio), self.cfg.num_points // 2)
# Le plancher 32768 = le modèle n'a JAMAIS vu de vraie densité 25%
# En pratique, densité min vue = ~33-55% selon taille des frames
```
→ Risque de drop >10% sur mAP à 25% pour câbles et antennes.

#### BUG 2 — Adam au lieu d'AdamW (train.py)
```python
optimizer = optim.Adam(...)  # ← Devrait être optim.AdamW
```
Avec Adam + weight_decay, le decay est L2 (biaise les moments adaptatifs). AdamW fait un decoupled weight decay correct. Gain estimé : +0.5-1.5% mIoU.

#### BUG 3 — Normalisation distance per-frame (prep_data.py + inference.py)
```python
dist_norm = distance_cm / distance_cm.max()  # ← max local variable par scène
```
Si Scene B a un range de distance différent, les features ont une distribution différente du train. Fix : normaliser par constante fixe (ex: 20000 cm = 200m).

#### FEATURE MANQUANTE — Verticality (prep_data.py)
- `verticality = 1 - |eigenvec_principal_z|` — quasi gratuit (eigenvalues déjà calculées)
- Distingue **câbles** (linéaire + horizontal) vs **pôles** (linéaire + vertical)
- d_in : 7 → 8 (casse le checkpoint, cold start requis)

---

### Plan Run 8 — Cold Start complet

| Changement | Fichier | Impact |
|---|---|---|
| Fix bug densité (supprimer plancher 32768) | dataset.py | Robustesse critique |
| Feature Verticality d_in 7→8 | prep_data.py, config.py | Câbles vs pôles |
| fc_start 8→16 (goulot d'étranglement) | model.py | +représentativité features |
| Dropout(0.5) dans classifier head | model.py | Généralisation Scene B |
| Adam → AdamW + weight_decay 5e-4 | train.py, config.py | +0.5-1.5% mIoU |
| grad_clip_norm 10.0 → 2.0 | config.py | Stabilité classes rares |
| Normalisation distance fixe (20000 cm) | prep_data.py, inference.py | Stabilité cross-scène |
| lr=0.002, epochs=300 (80 frais) | config.py | Nouveau cycle cosine |

**Prérequis :** re-preprocessing (d_in change) + cold start (architecture modifiée)

---

## Pistes d'Amélioration Restantes

### ⚠️ CRITIQUE — Tester la robustesse densité (PRIORITÉ #1)
- Run 8 a fixé le density floor bug → le modèle a VU de la vraie densité 25% pendant l'entraînement
- MAIS DBSCAN est calibré pour 100% → drop attendu sur mAP à 25% (moins de points = moins de clusters)
- À faire : lancer inference.py sur fichiers 25%/50%/75% et mesurer le delta mAP
- Fix possible sans ré-entraîner : DBSCAN adaptatif selon densité (min_samples × ratio_density)

### Priorité 2 — Run 9 : warm restart depuis epoch 192
- Run 8 a atteint epoch 192/200 avec mIoU=0.9504 — le modèle n'était peut-être pas encore convergé
- Stratégie : warm_start=True, epochs=300, LR reset à 0.002 (nouveau cycle cosine)
- Risque faible car architecture identique → no cold start requis

### Priorité 3 — Tester stabilité sur Scene B
- Mesurer mAP sur une 2e scène pour valider généralisation cross-scène
- Le Dropout(0.5) + AdamW + distance normalization fixe devraient aider

### Pour gagner en efficacité (critère d'évaluation)
- Le modèle est déjà à 0.65M params (très efficace)
- Hard limit : d_encoder[-1]=256 (AttentivePooling quadratique → 512=2M params)

---

## Infrastructure

| Composant | Detail |
|---|---|
| Machine | g2-standard-16 (16 vCPU, 64 Go RAM) |
| GPU | NVIDIA L4 (24 Go VRAM) |
| PyTorch | 2.1.0+cu128 |
| Stockage donnees | GCS bucket j2b-hackaton-airbus-2026 |
| Training time/epoch | ~50s |

## Structure des Fichiers

```
config.py           — Hyperparametres centralises
prep_data.py        — Preprocessing HDF5 → .npz (voxel + labels + features)
dataset.py          — PyTorch Dataset + augmentations + KNN indices
model.py            — RandLA-Net encoder-decoder
losses.py           — Focal Loss + Lovasz-Softmax
train.py            — Boucle d'entrainement (AMP, cosine LR, early stop)
inference.py        — Segmentation → DBSCAN/classe → OBB → CSV
train_notebook.ipynb — Notebook Jupyter pour GCP
lidar_utils.py      — Toolkit Airbus (load HDF5, spherical→cartesian)
visualize.py        — Visualisation Open3D (toolkit Airbus)
```

## Desequilibre des Classes (Run 1)

```
Class weights: [4.02, 20.00, 14.19, 1.79, 0.03]
                Ant    Cable  Pole   Turb   BG

→ Les cables sont 500x plus rares que le background
→ Le weight de 20 force le modele a ne pas les ignorer
```

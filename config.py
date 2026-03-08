"""
Configuration centralisée pour le pipeline RandLA-Net.
Tous les hyperparamètres sont ici — rien n'est hardcodé ailleurs.
"""
import os

class Config:
    # ═══════════════════════════════════════════
    # PATHS
    # ═══════════════════════════════════════════
    raw_data_dir = "airbus_hackathon_trainingdata"
    processed_dir = "processed_data"
    checkpoint_dir = "checkpoints"
    predictions_dir = "predictions"
    log_dir = "logs"

    # ═══════════════════════════════════════════
    # DATA
    # ═══════════════════════════════════════════
    num_scenes = 10
    num_points = 65536          # Points par sample (fixe pour batching)
    num_classes = 5             # 0-3 = obstacles, 4 = background
    sub_sampling_ratio = 4      # Ratio de sous-échantillonnage par couche
    num_layers = 4              # Nombre de couches encoder/decoder

    # Features d'entrée: xyz(3) + reflectivity(1) + dist_norm(1) = 5
    # Avec geometric features: + linearity(1) + planarity(1) = 7
    use_geometric_features = True
    use_verticality_feature = True  # Run 8: +1 feature (d_in 7→8), distingue câble horizontal vs poteau vertical
    d_in = (7 if use_geometric_features else 5) + (1 if use_geometric_features and use_verticality_feature else 0)
    k_geometric = 16           # K voisins pour geo features (training on-the-fly ET inférence)

    # Voxel downsampling (preprocessing)
    voxel_size = 0.10          # 10cm — plus de points préservés pour 65K target

    # Validation split
    val_ratio = 0.15           # 15% des frames pour validation

    # ═══════════════════════════════════════════
    # MODEL — RandLA-Net
    # ═══════════════════════════════════════════
    k_neighbors = 16           # K=32 OOM même à batch=8 (AttPool tensor B×N×K×d trop grand)
    d_encoder = [32, 64, 128, 256]  # Run 6: retour config stable Run 3 (645K params)
    # NE PAS toucher d_encoder[-1]: 256 est le max (512 = 2M params à cause de AttentivePooling²)

    # ═══════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════
    batch_size = 14
    lr = 0.005                 # Run 11: retour lr d'origine (cold start)
    weight_decay = 5e-4
    epochs = 300               # Run 11: 300 epochs cold start
    warmup_epochs = 15         # Run 11: 5→15 — cold start avec geo features on-the-fly instables
                               # 5 epochs insuffisant : optimizer accumule des moments corrompus
                               # avant que les geo features se stabilisent sur nuage sparse

    # Run 11: cold start obligatoire — features on-the-fly incompatibles avec checkpoint Run 8
    # Le modèle doit apprendre de zéro avec des geo features calculées à la densité réelle
    warm_start = False
    num_workers = 12         # 32 vCPU tuning: plus de debit dataloader sans changer la qualite
    pin_memory = True
    grad_clip_norm = 2.0       # Run 8: 10.0→2.0 (meilleure stabilité classes rares)
    use_amp = True             # Mixed precision (2x speed sur L4)

    # Early stopping
    patience = 70              # Run 11: 50→70 — curriculum long : chaque transition de palier
                               # dégrade temporairement val_miou (15-20 epochs de régression)
                               # patience=50 risque early stop prématuré autour epoch 200-230
    min_delta = 0.001          # Amélioration minimale

    # ═══════════════════════════════════════════
    # LOSS
    # ═══════════════════════════════════════════
    focal_gamma = 2.0
    focal_weight = 1.0
    lovasz_weight = 1.0
    # Class weights (calculés automatiquement par prep_data, fallback ici)
    # Ordre: Antenna, Cable, Pole, Turbine, Background
    default_class_weights = [12.0, 20.0, 16.0, 2.5, 0.03]

    # Override manuel des class weights
    # Run 7: turbine 1.82→2.5 pour récupérer la régression IoU (0.86→0.79 en Run 6)
    # Ordre: Antenna, Cable, Pole, Turbine, Background
    class_weights_override = [12.0, 20.0, 16.0, 2.5, 0.03]

    # ═══════════════════════════════════════════
    # AUGMENTATION (training only)
    # ═══════════════════════════════════════════
    # Density drop — LE trick pour la robustesse 25/50/75/100%
    density_drop_min = 0.25    # Plancher final (atteint à epoch 180 via curriculum)
    density_drop_max = 1.0

    # Run 11 : Curriculum learning densité (density_min décroît progressivement)
    # 30 epochs à densité 1.0 : cold start a besoin du signal propre avant dégradation
    # (Run 10 montrait que features à pleine densité = meilleur score 100% = 0.866)
    # Le plancher 25% est atteint à epoch 180 → 120 epochs restants pour consolider la robustesse
    # Clé : {epoch_start: density_min} — density_max reste 1.0
    density_curriculum_schedule = {0: 1.0, 30: 0.70, 80: 0.50, 130: 0.35, 180: 0.25}

    # Run 13 : Validation multi-densité — checkpoint sur mIoU moyen (25%+50%+75%+100%) / 4
    # Alignement exact avec le critère d'évaluation hackathon (4 densités séparées).
    # val_density = densité utilisée pour la validation rapide (logging epoch-par-epoch).
    # La validation complète 4 densités est effectuée tous les val_multi_density_freq epochs.
    val_density = 0.50         # Densité de fallback (validation rapide inter-cycles)
    val_multi_density_freq = 10  # Run 13: 5→10 — réduit overhead val multi-densité de 60%→30% # Validation complète toutes les N epochs (4x plus lent que val simple)
                               # N=5 : overhead ~25% du temps total si val dure 20% du train time

    # Run 9 : Biased density sampling (quand density_min < 0.5)
    # 50% des samples entre [min, 0.5], 50% entre [0.5, 1.0]
    biased_density_sampling = True

    # Run 12 : Biased sampling agressif en phase tardive du curriculum
    # Après epoch 180 (density_min=0.25) : 70% des samples dans [0.25, 0.40]
    # Run 11 : 50/50 split autour de 0.5 → trop peu de samples à exactement 25%
    biased_density_late_epoch = 180   # Epoch d'activation du mode agressif
    biased_density_late_prob  = 0.70  # 70% dans la zone critique [density_min, 0.40]

    # Run 11 : Bruit gaussien différentiel sur features géométriques (training seulement)
    # verticality plus instable à faible densité (estimation normale locale dégradée)
    # → sigma plus élevé sur verticality force le modèle à s'appuyer sur linearity/planarity
    # pour distinguer cable (horizontal) de pole (vertical)
    # Note: appliqué de manière différentielle dans dataset.py (voir geo_feature_noise_verticality_std)
    geo_feature_noise_std = 0.04              # linearity + planarity
    geo_feature_noise_verticality_std = 0.08  # verticality (ratio 2x validé par agent-robustness)

    # Géométrique
    rotation_range = 180.0     # ±180° autour de Z
    scale_min = 0.9
    scale_max = 1.1
    jitter_std = 0.01          # Bruit gaussien sur XYZ
    random_flip_prob = 0.5     # Flip horizontal

    # Oversampling des frames avec classes rares
    oversample_rare_classes = True
    oversample_factor = 3      # x3 pour frames avec câbles

    # ═══════════════════════════════════════════
    # INFERENCE
    # ═══════════════════════════════════════════
    # Test-Time Augmentation
    use_tta = True
    tta_rotations = [0, 90, 180, 270]  # Degrés autour de Z

    # DBSCAN par classe
    # Cable : REVERT — eps=2.0 + min_samples=6 créait 485 prédictions pour 280 GT = 456 FP
    # Antenna : min_cluster 60→80 + confidence 0.60→0.65 → FP 22→17 ✅ (validé)
    # Turbine : min_cluster 25→20 + confidence 0.50→0.45 → recall légèrement amélioré
    dbscan_params = {
        0: {"eps": 2.5, "min_samples": 80},    # Antenna — inchangé
        1: {"eps": 1.5, "min_samples": 10},    # Cable — params 100% densité
        2: {"eps": 2.0, "min_samples": 25},    # Pole — inchangé
        3: {"eps": 5.0, "min_samples": 30},    # Wind turbine — inchangé
    }

    # Filtrage des petits clusters (bruit)
    min_cluster_points = {
        0: 80,   # Antenna — 60→80 ✅ validé (réduit FP 22→17)
        1: 8,    # Cable — params 100% densité
        2: 20,   # Pole — inchangé
        3: 20,   # Turbine — 25→20 ✅ validé
    }

    # Adaptive DBSCAN antenna selon densité.
    # À 25%, le seuil fixe 80/80 devient mécaniquement trop strict sur une partie
    # des frames, même avant les erreurs de segmentation. On baisse donc seulement
    # les seuils structurellement impossibles, en gardant une confidence élevée.
    antenna_density_params = {
        0.30: {"min_samples": 24, "min_cluster": 20, "confidence": 0.60},  # ≤30% (~25%)
        0.55: {"min_samples": 40, "min_cluster": 32, "confidence": 0.62},  # ≤55% (~50%)
        0.80: {"min_samples": 60, "min_cluster": 48, "confidence": 0.64},  # ≤80% (~75%)
        # >80% → params normaux (min_samples=80, min_cluster=80, confidence=0.65)
    }

    # Adaptive DBSCAN câble selon densité (lookup table par seuil)
    # Les premiers tests sur l'eval set montrent une inflation des détections câble
    # quand la densité baisse, surtout sur la scène B. On reste donc adaptatif à
    # basse densité, mais avec des seuils plus conservateurs qu'avant pour limiter
    # la fragmentation et les FP.
    cable_density_params = {
        0.30: {"min_samples": 5, "min_cluster": 6, "confidence": 0.40},  # ≤30% (~25%) plus conservateur
        0.55: {"min_samples": 7, "min_cluster": 6, "confidence": 0.42},  # ≤55% (~50%)
        0.80: {"min_samples": 8, "min_cluster": 6, "confidence": 0.44},  # ≤80% (~75%)
        # >80% → params normaux (min_samples=10, min_cluster=8, confidence=0.45)
    }

    # Score de confiance minimum par cluster (moyenne des probas softmax)
    # Filtre les détections peu sûres sans toucher aux poids du modèle
    # Cable REVERT 0.40→0.45 : confidence=0.40 trop permissif avec eps=2.0 → avalanche FP
    confidence_threshold = {
        0: 0.65,   # Antenna — 0.60→0.65 ✅ validé (réduit FP)
        1: 0.45,   # Cable — REVERT 0.40→0.45 (combinaison eps=2.0+conf=0.40 = désastre)
        2: 0.55,   # Pole — inchangé
        3: 0.45,   # Turbine — 0.50→0.45 ✅ validé
    }

    # ═══════════════════════════════════════════
    # ÉVALUATION — GT extraction (compute_map.py)
    # ═══════════════════════════════════════════
    # Params FIXES pour extraire la GT depuis les labels HDF5 dans compute_map.py.
    # = exactement les params Run 7 (qui donnaient mAP=0.6984, notre référence)
    # ⚠️ NE PAS MODIFIER — GT doit rester stable pour comparer les runs entre eux
    # GT scene_1 avec ces params : ~36 antenna, ~280 cable, ~73 pole, ~165 turbine
    gt_dbscan_params = {
        0: {"eps": 2.5, "min_samples": 80},   # Antenna — Run 7 (36 GT objects)
        1: {"eps": 1.5, "min_samples": 10},   # Cable — Run 7 (280 GT objects)
        2: {"eps": 2.0, "min_samples": 25},   # Pole — Run 7 (73 GT objects)
        3: {"eps": 5.0, "min_samples": 30},   # Turbine — Run 7 (165 GT objects)
    }
    gt_min_cluster_points = {
        0: 60,   # Antenna — Run 7
        1: 8,    # Cable — Run 7
        2: 20,   # Pole — Run 7
        3: 25,   # Turbine — Run 7
    }

    # Class labels pour le CSV
    class_labels = {
        0: "Antenna",
        1: "Cable",
        2: "Electric Pole",
        3: "Wind Turbine",
    }

    @classmethod
    def ensure_dirs(cls):
        """Crée les dossiers nécessaires."""
        for d in [cls.processed_dir, cls.checkpoint_dir,
                  cls.predictions_dir, cls.log_dir]:
            os.makedirs(d, exist_ok=True)

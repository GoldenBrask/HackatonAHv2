"""
PyTorch Dataset pour nuages de points LiDAR.
- Charge les frames preprocessées (.npz)
- Applique les augmentations (densité, rotation, scale, jitter, flip)
- Calcule les indices KNN et subsampling à la volée
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.spatial import cKDTree
from config import Config
from prep_data import compute_geometric_features


class LidarDataset(Dataset):
    """
    Dataset pour entraînement RandLA-Net.
    Chaque sample est un frame preprocessé (.npz).
    """

    def __init__(self, file_list, training=True, cfg=None):
        self.file_list = file_list
        self.training = training
        self.cfg = cfg or Config()
        self.current_epoch = 0  # Mis à jour par train.py pour le curriculum

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        xyz = data["xyz"].astype(np.float32)               # (N_raw, 3)
        # Run 11: charger seulement les 5 features de base (xyz+refl+dist)
        # Les geo features sont recalculées on-the-fly APRÈS density drop (k=16 cohérent)
        # Cold start : pas de mismatch possible avec un checkpoint existant
        features = data["features"][:, :5].astype(np.float32)  # (N_raw, 5)
        labels = data["labels"].astype(np.int64)           # (N_raw,)

        # ── Density drop ──
        if self.training:
            # Curriculum + biased sampling (Run 9/11)
            density_min = self._get_density_min()
            biased = getattr(self.cfg, 'biased_density_sampling', False)
            if biased and density_min < 0.5:
                # Run 12 : phase tardive → 70% des samples dans la zone critique [min, 0.40]
                late_epoch = getattr(self.cfg, 'biased_density_late_epoch', 999)
                late_prob  = getattr(self.cfg, 'biased_density_late_prob', 0.5)
                low_prob   = late_prob if self.current_epoch >= late_epoch else 0.5
                if np.random.random() < low_prob:
                    keep_ratio = np.random.uniform(density_min, 0.40)
                else:
                    keep_ratio = np.random.uniform(0.40, self.cfg.density_drop_max)
            else:
                keep_ratio = np.random.uniform(density_min, self.cfg.density_drop_max)

            n_keep = max(int(len(xyz) * keep_ratio), 100)
            if n_keep < len(xyz):
                keep_idx = np.random.choice(len(xyz), n_keep, replace=False)
                xyz = xyz[keep_idx]
                features = features[keep_idx]
                labels = labels[keep_idx]
        else:
            # Run 12 : validation à densité réduite → checkpoint = meilleur modèle robuste
            # Run 11 : val à 100% → checkpointing optimisait la perf pleine densité (mauvais objectif)
            val_density = getattr(self.cfg, 'val_density', 1.0)
            if val_density < 1.0:
                np.random.seed(idx)  # Reproductible par sample (stable entre epochs)
                n_keep = max(int(len(xyz) * val_density), 100)
                keep_idx = np.random.choice(len(xyz), n_keep, replace=False)
                xyz      = xyz[keep_idx]
                features = features[keep_idx]
                labels   = labels[keep_idx]

        # ── Features géométriques on-the-fly (après density drop, avant _fix_size) ──
        # Run 11: k=16 identique training ET inférence → zéro mismatch de distribution
        # Cold start : le modèle apprend directement avec les features sparse
        if self.cfg.use_geometric_features:
            k_geo = self.cfg.k_geometric  # k=16 partout (cohérence training/inference)
            use_vert = getattr(self.cfg, 'use_verticality_feature', False)
            if use_vert:
                lin, plan, vert = compute_geometric_features(
                    xyz, k=k_geo, compute_verticality=True
                )
                geo = np.column_stack([lin, plan, vert]).astype(np.float32)
            else:
                lin, plan = compute_geometric_features(xyz, k=k_geo)
                geo = np.column_stack([lin, plan]).astype(np.float32)

            # Feature noise différentiel (training seulement)
            # Run 11: verticality sigma=0.08 vs linearity/planarity sigma=0.04
            # verticality plus instable à faible densité (estimation normale locale dégradée)
            # → sigma 2x sur verticality force le modèle à s'appuyer sur lin/plan
            #   pour distinguer cable (horizontal) de pole (vertical)
            if self.training:
                sigma_linplan = getattr(self.cfg, 'geo_feature_noise_std', 0.04)
                sigma_vert = getattr(self.cfg, 'geo_feature_noise_verticality_std', sigma_linplan)
                if use_vert and geo.shape[1] == 3:
                    # Colonnes : [linearity, planarity, verticality]
                    geo[:, :2] += np.random.normal(0, sigma_linplan, (geo.shape[0], 2)).astype(np.float32)
                    geo[:, 2]  += np.random.normal(0, sigma_vert,    geo.shape[0]).astype(np.float32)
                else:
                    geo += np.random.normal(0, sigma_linplan, geo.shape).astype(np.float32)

            features = np.hstack([features, geo])  # (N, 8)

        # ── Sous-échantillonnage / padding à num_points fixe ──
        xyz, features, labels = self._fix_size(xyz, features, labels)

        # ── Augmentations géométriques ──
        if self.training:
            xyz = self._random_rotate_z(xyz)
            xyz = self._random_scale(xyz)
            xyz = self._random_jitter(xyz)
            xyz = self._random_flip(xyz)

        # ── Mettre à jour les features xyz (les 3 premières colonnes) ──
        features[:, :3] = xyz

        # ── Calculer les indices hiérarchiques pour RandLA-Net ──
        result = self._compute_hierarchical_indices(xyz)
        result["xyz"] = torch.from_numpy(xyz)
        result["features"] = torch.from_numpy(features)
        result["labels"] = torch.from_numpy(labels)

        return result

    def _get_density_min(self):
        """Retourne le density_min courant selon le curriculum schedule."""
        schedule = getattr(self.cfg, 'density_curriculum_schedule', None)
        if not schedule:
            return self.cfg.density_drop_min
        density_min = self.cfg.density_drop_min
        for epoch_thresh in sorted(schedule.keys()):
            if self.current_epoch >= epoch_thresh:
                density_min = schedule[epoch_thresh]
        return density_min

    def _fix_size(self, xyz, features, labels):
        """Sous-échantillonne ou pad pour obtenir exactement num_points.

        Run 9 FIX: les points paddés reçoivent un bruit gaussien (sigma=5mm) sur XYZ
        pour briser les distances KNN nulles (distance=0 → LocalSpatialEncoding corrompue).
        """
        N = len(xyz)
        target = self.cfg.num_points

        if N >= target:
            # Random subsample
            idx = np.random.choice(N, target, replace=False)
            return xyz[idx], features[idx], labels[idx]
        else:
            # Pad par duplication avec bruit — évite KNN distance=0
            pad_n = target - N
            pad_idx = np.random.choice(N, pad_n, replace=True)
            pad_xyz = xyz[pad_idx] + np.random.normal(0, 0.005, (pad_n, 3)).astype(np.float32)
            pad_feat = features[pad_idx].copy()
            pad_feat[:, :3] = pad_xyz  # Cohérence xyz dans les features paddées
            xyz = np.vstack([xyz, pad_xyz])
            features = np.vstack([features, pad_feat])
            labels = np.concatenate([labels, labels[pad_idx]])
            return xyz, features, labels

    def _random_rotate_z(self, xyz):
        """Rotation aléatoire autour de Z."""
        angle = np.random.uniform(
            -self.cfg.rotation_range, self.cfg.rotation_range
        )
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a, 0],
                       [sin_a, cos_a, 0],
                       [0, 0, 1]], dtype=np.float32)
        return xyz @ R.T

    def _random_scale(self, xyz):
        """Scale aléatoire."""
        s = np.random.uniform(self.cfg.scale_min, self.cfg.scale_max)
        return xyz * s

    def _random_jitter(self, xyz):
        """Bruit gaussien."""
        noise = np.random.normal(0, self.cfg.jitter_std,
                                  size=xyz.shape).astype(np.float32)
        return xyz + noise

    def _random_flip(self, xyz):
        """Flip horizontal aléatoire (axe Y)."""
        if np.random.random() < self.cfg.random_flip_prob:
            xyz[:, 1] = -xyz[:, 1]
        return xyz

    def _compute_hierarchical_indices(self, xyz):
        """
        Calcule les indices KNN, subsampling et upsampling pour chaque couche.
        C'est le cœur de la préparation pour RandLA-Net.
        """
        result = {}
        current_points = xyz.copy()
        K = self.cfg.k_neighbors
        ratio = self.cfg.sub_sampling_ratio

        for i in range(self.cfg.num_layers):
            N = len(current_points)
            N_sub = N // ratio

            # KNN via cKDTree (rapide, ~30ms pour 40K points)
            tree = cKDTree(current_points)
            _, knn_idx = tree.query(current_points, k=K)
            result[f"neigh_{i}"] = torch.from_numpy(knn_idx.astype(np.int64))

            # Random subsampling
            sub_idx = np.sort(np.random.choice(N, N_sub, replace=False))
            result[f"sub_{i}"] = torch.from_numpy(sub_idx.astype(np.int64))

            # Upsampling: pour chaque point courant, son voisin le plus
            # proche dans le sous-ensemble
            sub_points = current_points[sub_idx]
            tree_sub = cKDTree(sub_points)
            _, up_idx = tree_sub.query(current_points, k=1)
            result[f"up_{i}"] = torch.from_numpy(up_idx.astype(np.int64))

            current_points = sub_points

        return result


def build_dataloaders(cfg=None):
    """
    Construit les DataLoaders train/val avec oversampling.
    """
    cfg = cfg or Config()

    # Lister tous les fichiers preprocessés
    all_files = sorted(glob.glob(os.path.join(cfg.processed_dir, "*.npz")))
    all_files = [f for f in all_files if "stats" not in os.path.basename(f)]
    if not all_files:
        raise FileNotFoundError(
            f"Aucun fichier .npz trouvé dans {cfg.processed_dir}. "
            "Lancez d'abord prep_data.py"
        )

    # Split train/val
    np.random.seed(42)
    n_val = max(1, int(len(all_files) * cfg.val_ratio))
    indices = np.random.permutation(len(all_files))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]

    print(f"Train: {len(train_files)} frames, Val: {len(val_files)} frames")

    # ── Oversampling des frames avec classes rares ──
    sampler = None
    if cfg.oversample_rare_classes:
        weights = _compute_sample_weights(train_files, cfg)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

    train_dataset = LidarDataset(train_files, training=True, cfg=cfg)
    val_dataset = LidarDataset(val_files, training=False, cfg=cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=3,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=3,
    )

    return train_loader, val_loader


def _compute_sample_weights(file_list, cfg):
    """
    Calcule les poids de sampling pour oversampling.
    Frames avec classes rares (câbles, antennes, poteaux) pondérées x3.
    """
    weights = []
    for f in file_list:
        data = np.load(f)
        labels = data["labels"]
        # Poids par classe rare présente dans le frame
        w = 1.0
        if np.sum(labels == 1) > 0:  # Cable
            w = max(w, cfg.oversample_factor)
        if np.sum(labels == 0) > 0:  # Antenna
            w = max(w, cfg.oversample_factor * 0.8)
        if np.sum(labels == 2) > 0:  # Pole
            w = max(w, cfg.oversample_factor * 0.8)
        weights.append(w)
    return weights

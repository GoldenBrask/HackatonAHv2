"""
Preprocessing complet des données Airbus HDF5.

Pour chaque scène .h5 :
  1. Charge les données brutes
  2. Filtre les tirs invalides (distance_cm == 0)
  3. Sépare en frames (par pose ego)
  4. Convertit sphérique → cartésien (XYZ en mètres)
  5. Assigne les class_ID depuis les couleurs RGB
  6. Construit les features: xyz + reflectivity + distance_norm
  7. (Optionnel) Calcule features géométriques: linearity, planarity
  8. Sauvegarde chaque frame en .npz

Exécuter sur GCP avant l'entraînement.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial import cKDTree
import lidar_utils
from config import Config

# Mapping RGB → Class ID (from Airbus README)
COLOR_MAP = {
    (38, 23, 180): 0,    # Antenna
    (177, 132, 47): 1,   # Cable
    (129, 81, 97): 2,    # Electric pole
    (66, 132, 9): 3,     # Wind turbine
}
BACKGROUND_ID = 4


def voxel_downsample(xyz, features, labels, voxel_size=0.15):
    """
    Voxel downsampling qui préserve la couverture spatiale.
    Pour chaque cellule 3D, garde le point avec la classe la plus rare
    (privilégie les obstacles vs background).

    575K points → ~60-100K points selon la scène.
    """
    # Calculer les indices de voxel pour chaque point
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)

    # Encoder les 3 indices en un seul hash
    # Offset pour éviter les négatifs
    voxel_indices -= voxel_indices.min(axis=0)
    dims = voxel_indices.max(axis=0) + 1
    flat_idx = (voxel_indices[:, 0] * dims[1] * dims[2]
                + voxel_indices[:, 1] * dims[2]
                + voxel_indices[:, 2])

    # Pour chaque voxel, garder le point avec la classe la plus rare
    # Priorité : Cable(1) > Antenna(0) > Pole(2) > Turbine(3) > Background(4)
    # On utilise un score inversé : classe rare = score élevé
    priority = np.array([3, 4, 2, 1, 0])  # index = class_id, valeur = priorité
    point_priority = priority[labels]

    # Pour chaque voxel unique, garder le point de plus haute priorité
    unique_voxels = np.unique(flat_idx)
    keep_indices = np.empty(len(unique_voxels), dtype=np.int64)

    # Méthode vectorisée : trier par (voxel, -priorité), garder le premier par voxel
    sort_key = flat_idx.astype(np.int64) * 10 - point_priority
    sorted_order = np.argsort(sort_key, kind="stable")
    sorted_voxels = flat_idx[sorted_order]

    # Premier index de chaque voxel unique
    mask = np.concatenate([[True], sorted_voxels[1:] != sorted_voxels[:-1]])
    keep_indices = sorted_order[mask]

    n_before = len(xyz)
    xyz = xyz[keep_indices]
    features = features[keep_indices]
    labels = labels[keep_indices]

    return xyz, features, labels, n_before


def assign_class_ids(df):
    """Vectorized RGB → class_ID assignment."""
    labels = np.full(len(df), BACKGROUND_ID, dtype=np.int64)

    r = df["r"].values
    g = df["g"].values
    b = df["b"].values

    for (cr, cg, cb), class_id in COLOR_MAP.items():
        mask = (r == cr) & (g == cg) & (b == cb)
        labels[mask] = class_id

    return labels


def compute_geometric_features(xyz, k=16, compute_verticality=False):
    """
    Calcule les features géométriques pour chaque point.
    Basé sur les eigenvalues/eigenvectors de la covariance locale (k voisins).

    - linearity   = (λ1 - λ2) / λ1  → élevée pour les câbles et poteaux
    - planarity   = (λ2 - λ3) / λ1  → élevée pour les surfaces planes
    - verticality = 1 - |eigenvec_principal_z|  (Run 8, si compute_verticality=True)
      → élevée pour les câbles (horizontaux), basse pour les poteaux (verticaux)
      → distingue câble (linéaire + horizontal) vs poteau (linéaire + vertical)

    Returns: (linearity, planarity) ou (linearity, planarity, verticality)
    """
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k)

    N = len(xyz)
    linearity   = np.zeros(N, dtype=np.float32)
    planarity   = np.zeros(N, dtype=np.float32)
    verticality = np.zeros(N, dtype=np.float32) if compute_verticality else None

    # Process en batches pour la mémoire
    batch_size = 10000
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_idx = idx[start:end]  # (batch, k)
        neighbors = xyz[batch_idx]   # (batch, k, 3)

        # Centrer
        centroid = neighbors.mean(axis=1, keepdims=True)  # (batch, 1, 3)
        centered = neighbors - centroid                    # (batch, k, 3)

        # Covariance: (batch, 3, 3)
        cov = np.einsum("bki,bkj->bij", centered, centered) / k

        if compute_verticality:
            # eigh retourne eigenvalues (croissant) ET eigenvectors en colonnes
            eigenvalues, eigenvectors = np.linalg.eigh(cov)  # (batch, 3), (batch, 3, 3)
            l1 = eigenvalues[:, 2]  # Plus grande
            l2 = eigenvalues[:, 1]
            l3 = eigenvalues[:, 0]  # Plus petite (pas utilisée ici)

            # Eigenvector principal = direction de la plus grande variance = direction du câble
            # eigenvectors[:, :, 2] = colonne 2 = eigenvec pour λ1
            principal_z = eigenvectors[:, 2, 2]  # Composante Z de l'eigenvec principal
            verticality[start:end] = 1.0 - np.abs(principal_z).astype(np.float32)
        else:
            eigenvalues = np.linalg.eigvalsh(cov)  # (batch, 3), λ3 ≤ λ2 ≤ λ1
            l1 = eigenvalues[:, 2]
            l2 = eigenvalues[:, 1]

        # Éviter division par zéro
        safe_l1 = np.maximum(l1, 1e-8)
        linearity[start:end] = (l1 - l2) / safe_l1
        planarity[start:end] = (l2 - (eigenvalues[:, 0] if not compute_verticality else eigenvalues[:, 0])) / safe_l1

    if compute_verticality:
        return linearity, planarity, verticality
    return linearity, planarity


def process_scene(scene_path, scene_id, cfg, global_stats):
    """Traite une scène HDF5 complète."""
    print(f"\n{'='*60}")
    print(f"Scène {scene_id}: {scene_path}")
    print(f"{'='*60}")

    # 1. Charger
    df = lidar_utils.load_h5_data(scene_path)
    print(f"  Points bruts: {len(df):,}")

    # 2. Filtrer tirs invalides
    df = df[df["distance_cm"] > 0].copy()
    print(f"  Points valides: {len(df):,}")

    # 3. Extraire les frames uniques
    poses = lidar_utils.get_unique_poses(df)
    print(f"  Frames: {len(poses)}")

    frame_count = 0
    for _, pose_row in poses.iterrows():
        pose_idx = int(pose_row["pose_index"])
        frame_df = lidar_utils.filter_by_pose(df, pose_row)

        if len(frame_df) < 100:
            continue

        # 4. Convertir en XYZ local (mètres)
        xyz = lidar_utils.spherical_to_local_cartesian(frame_df)
        xyz = xyz.astype(np.float32)

        # 5. Class labels
        labels = assign_class_ids(frame_df)

        # 6. Features de base
        reflectivity = frame_df["reflectivity"].values.astype(np.float32) / 255.0
        # Run 8 FIX: normalisation distance par constante fixe (20000 cm = 200m)
        # Avant: / dist_max (per-frame) → distribution différente selon la scène → instable cross-scène
        # Après: / 20000.0 → invariant à la scène, cohérent train/inférence
        distance_norm = frame_df["distance_cm"].values.astype(np.float32) / 20000.0

        # 7. (Optionnel) Features géométriques
        if cfg.use_geometric_features:
            use_vert = getattr(cfg, "use_verticality_feature", False)
            if use_vert:
                linearity, planarity, verticality = compute_geometric_features(
                    xyz, k=cfg.k_geometric, compute_verticality=True
                )
                features = np.column_stack([
                    xyz, reflectivity, distance_norm, linearity, planarity, verticality
                ])
            else:
                linearity, planarity = compute_geometric_features(
                    xyz, k=cfg.k_geometric
                )
                features = np.column_stack([
                    xyz, reflectivity, distance_norm, linearity, planarity
                ])
        else:
            features = np.column_stack([xyz, reflectivity, distance_norm])

        # 7b. Voxel downsampling (préserve couverture spatiale)
        if cfg.voxel_size > 0:
            xyz, features, labels, n_before = voxel_downsample(
                xyz, features, labels, voxel_size=cfg.voxel_size
            )
            if frame_count == 0:  # Log seulement pour la première frame
                print(f"  Voxel downsample: {n_before:,} → {len(xyz):,} points "
                      f"(voxel={cfg.voxel_size}m)")

        # 8. Sauvegarder
        ego_pose = np.array([
            pose_row["ego_x"], pose_row["ego_y"],
            pose_row["ego_z"], pose_row["ego_yaw"]
        ], dtype=np.float64)

        out_name = f"scene{scene_id:02d}_frame{pose_idx:03d}.npz"
        out_path = os.path.join(cfg.processed_dir, out_name)
        np.savez_compressed(
            out_path,
            xyz=xyz,
            features=features,
            labels=labels,
            ego_pose=ego_pose,
        )
        frame_count += 1

        # Stats globales
        for c in range(5):
            global_stats[c] += int(np.sum(labels == c))

    print(f"  → {frame_count} frames sauvegardées")
    return frame_count


def main():
    cfg = Config()
    cfg.ensure_dirs()

    # Trouver toutes les scènes
    scene_files = sorted([
        os.path.join(cfg.raw_data_dir, f)
        for f in os.listdir(cfg.raw_data_dir)
        if f.endswith(".h5")
    ])

    if not scene_files:
        print(f"ERREUR: Aucun fichier .h5 dans {cfg.raw_data_dir}/")
        sys.exit(1)

    print(f"Trouvé {len(scene_files)} scènes")
    print(f"Geometric features: {'ON' if cfg.use_geometric_features else 'OFF'}")
    print(f"Voxel downsample: {'OFF' if cfg.voxel_size <= 0 else f'{cfg.voxel_size}m'}")
    print(f"Output: {cfg.processed_dir}/")

    # Traiter chaque scène
    global_stats = Counter()
    total_frames = 0

    for i, sf in enumerate(scene_files, 1):
        n = process_scene(sf, i, cfg, global_stats)
        total_frames += n

    # ── Résumé et calcul des class weights ──
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ DU PREPROCESSING")
    print(f"{'='*60}")
    print(f"Total frames: {total_frames}")

    total_points = sum(global_stats.values())
    class_names = {
        0: "Antenna", 1: "Cable", 2: "Electric pole",
        3: "Wind turbine", 4: "Background"
    }

    print(f"\nRépartition des classes:")
    for c in range(5):
        count = global_stats[c]
        pct = 100.0 * count / total_points if total_points > 0 else 0
        print(f"  {class_names[c]:20s}: {count:>12,} points ({pct:6.2f}%)")

    # Calcul des poids inversement proportionnels à la fréquence
    class_weights = []
    for c in range(5):
        freq = global_stats[c] / total_points if total_points > 0 else 1
        w = 1.0 / (freq + 1e-6)
        class_weights.append(w)

    # Normaliser pour que le max = 20 (éviter explosion)
    max_w = max(class_weights)
    class_weights = [w / max_w * 20.0 for w in class_weights]

    print(f"\nClass weights calculés:")
    for c in range(5):
        print(f"  {class_names[c]:20s}: {class_weights[c]:.2f}")

    # Sauvegarder les stats
    stats = {
        "total_frames": total_frames,
        "total_points": total_points,
        "class_counts": dict(global_stats),
        "class_weights": class_weights,
    }
    stats_path = os.path.join(cfg.processed_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats sauvegardées: {stats_path}")


if __name__ == "__main__":
    main()

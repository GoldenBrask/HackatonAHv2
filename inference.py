"""
Pipeline d'inférence complet:
  1. Charger le modèle entraîné
  2. Segmenter chaque frame (avec TTA optionnel)
  3. DBSCAN par classe → clusters
  4. Oriented Bounding Box (OBB) via Minimum Area Rectangle → bbox 3D
  5. Générer le CSV au format Airbus

Usage:
  python inference.py --input eval_data.h5 --output predictions.csv
  python inference.py --input eval_data.h5 --output predictions.csv --no-tta
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from scipy.spatial import cKDTree, ConvexHull
from sklearn.cluster import DBSCAN

import lidar_utils
from config import Config
from model import RandLANet


def normalize_state_dict_keys(state_dict):
    """Strip torch.compile wrapper prefixes when present."""
    if not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    if all(k.startswith("_orig_mod.") for k in keys):
        return {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
    return state_dict


def load_model(checkpoint_path, device, cfg=None):
    """Charge le modèle depuis un checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("config", {})

    model = RandLANet(
        d_in=model_cfg.get("d_in", cfg.d_in if cfg else 5),
        num_classes=model_cfg.get("num_classes", cfg.num_classes if cfg else 5),
        d_encoder=model_cfg.get("d_encoder", cfg.d_encoder if cfg else [32, 64, 128, 256]),
        num_layers=model_cfg.get("num_layers", cfg.num_layers if cfg else 4),
    ).to(device)

    state_dict = normalize_state_dict_keys(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Modèle chargé: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}, "
          f"Val mIoU: {checkpoint.get('val_miou', '?')}")
    return model


def compute_hierarchical_indices(xyz, cfg):
    """Calcule les indices KNN/sub/up pour un sample."""
    result = {}
    current_points = xyz.copy()

    for i in range(cfg.num_layers):
        N = len(current_points)
        N_sub = N // cfg.sub_sampling_ratio

        tree = cKDTree(current_points)
        _, knn_idx = tree.query(current_points, k=cfg.k_neighbors)
        result[f"neigh_{i}"] = torch.from_numpy(knn_idx.astype(np.int64)).unsqueeze(0)

        sub_idx = np.sort(np.random.choice(N, N_sub, replace=False))
        result[f"sub_{i}"] = torch.from_numpy(sub_idx.astype(np.int64)).unsqueeze(0)

        sub_points = current_points[sub_idx]
        tree_sub = cKDTree(sub_points)
        _, up_idx = tree_sub.query(current_points, k=1)
        result[f"up_{i}"] = torch.from_numpy(up_idx.astype(np.int64)).unsqueeze(0)

        current_points = sub_points

    return result


def voxel_downsample_inference(xyz, refl, dist, voxel_size=0.10):
    """Voxel downsampling pour l'inférence (sans labels).

    Aligne l'inference sur le preprocessing training (prep_data.py:voxel_downsample).
    575K points bruts → ~80-100K points voxelisés, même distribution que training.
    Garde le premier point (par ordre de flat_idx) dans chaque cellule voxel.
    """
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)
    voxel_indices -= voxel_indices.min(axis=0)
    dims = voxel_indices.max(axis=0) + 1
    flat_idx = (voxel_indices[:, 0] * dims[1] * dims[2]
                + voxel_indices[:, 1] * dims[2]
                + voxel_indices[:, 2])

    sorted_order = np.argsort(flat_idx, kind="stable")
    sorted_voxels = flat_idx[sorted_order]
    mask = np.concatenate([[True], sorted_voxels[1:] != sorted_voxels[:-1]])
    keep_idx = sorted_order[mask]

    return xyz[keep_idx], refl[keep_idx], dist[keep_idx]


def prepare_features(xyz, reflectivity, distance_cm, cfg):
    """Construit les features de base (xyz + refl + dist).

    Run 9: les geo features (linearity/planarity/verticality) sont calculées
    on-the-fly APRÈS le density drop dans process_file, pour rester cohérent
    avec l'entraînement qui les calcule aussi après density drop.
    """
    refl_norm = reflectivity / 255.0
    # Run 8 FIX: normalisation distance par constante fixe (cohérente avec prep_data.py)
    dist_norm = distance_cm / 20000.0
    features = np.column_stack([xyz, refl_norm, dist_norm])
    return features.astype(np.float32)


def get_adaptive_params(class_id, density, cfg):
    """Retourne les paramètres DBSCAN adaptés à la densité (Run 9).

    Scaling sqrt(density) pour min_samples et min_cluster (conservateur).
    eps INCHANGÉ — distance physique, les objets ne changent pas de taille.
    confidence légèrement réduite à faible densité (max -8% à 25%).
    """
    params = cfg.dbscan_params[class_id]
    min_pts = cfg.min_cluster_points.get(class_id, 10)
    conf = cfg.confidence_threshold.get(class_id, 0.0)

    if density >= 1.0:
        return params["eps"], params["min_samples"], min_pts, conf

    # sqrt scaling : 75%→0.87, 50%→0.71, 25%→0.50 (plancher 0.50)
    scale = max(density ** 0.5, 0.5)
    adj_min_samples = max(int(params["min_samples"] * scale), 3)
    adj_min_pts = max(int(min_pts * scale), 5)
    adj_conf = conf * (0.9 + 0.1 * density)  # max -8% à 25%

    return params["eps"], adj_min_samples, adj_min_pts, adj_conf


def subsample_and_pad(xyz, features, num_points):
    """Ajuste à num_points exact."""
    N = len(xyz)
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
        return xyz[idx], features[idx], idx
    else:
        # Alignement training/inference:
        # dataset.py ajoute un bruit 5 mm aux points paddés pour éviter des
        # distances KNN nulles qui dégradent LocalSpatialEncoding à faible densité.
        pad_n = num_points - N
        pad_idx = np.random.choice(N, pad_n, replace=True)
        pad_xyz = xyz[pad_idx] + np.random.normal(0, 0.005, (pad_n, 3)).astype(np.float32)
        pad_features = features[pad_idx].copy()
        pad_features[:, :3] = pad_xyz
        idx_all = np.concatenate([np.arange(N), pad_idx])
        return (
            np.vstack([xyz, pad_xyz]),
            np.vstack([features, pad_features]),
            idx_all,
        )


def rotate_z(xyz, angle_deg):
    """Rotation autour de Z."""
    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return xyz @ R.T


@torch.no_grad()
def segment_frame(model, xyz, features, device, cfg, use_tta=True):
    """
    Segmente un nuage de points.
    Retourne les probabilités par classe pour TOUS les points originaux.
    """
    N_orig = len(xyz)
    all_probs = np.zeros((N_orig, cfg.num_classes), dtype=np.float32)

    rotations = cfg.tta_rotations if (use_tta and cfg.use_tta) else [0]

    for rot in rotations:
        # Rotation
        xyz_rot = rotate_z(xyz, rot) if rot != 0 else xyz.copy()

        # Subsample à num_points
        xyz_sub, feat_sub, sub_idx = subsample_and_pad(
            xyz_rot, features.copy(), cfg.num_points
        )

        # Mettre à jour les features XYZ
        feat_sub[:, :3] = xyz_sub

        # Compute indices
        np.random.seed(0)  # Reproductible pour les indices
        batch_data = compute_hierarchical_indices(xyz_sub, cfg)
        for key in batch_data:
            batch_data[key] = batch_data[key].to(device)

        xyz_t = torch.from_numpy(xyz_sub).unsqueeze(0).to(device)
        feat_t = torch.from_numpy(feat_sub).unsqueeze(0).to(device)

        with autocast("cuda", enabled=cfg.use_amp and device.type == "cuda"):
            logits = model(xyz_t, feat_t, batch_data)  # (1, N, C)

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (N, C)

        # Propager les probas aux points originaux
        # sub_idx mappe points subsamplés → points originaux
        if N_orig <= cfg.num_points:
            # Cas padding: les N_orig premiers sont les vrais points
            valid_probs = probs[:N_orig]
            all_probs += valid_probs
        else:
            # Cas subsample: propager via KNN
            tree_sub = cKDTree(xyz_rot[sub_idx[:cfg.num_points]])
            _, nn_idx = tree_sub.query(xyz_rot, k=1)
            all_probs += probs[nn_idx.flatten()]

    # Moyenne des rotations
    all_probs /= len(rotations)

    return all_probs


def compute_obb(points):
    """
    Calcule l'Oriented Bounding Box via Minimum Area Rectangle (MAR) en BEV.

    Supérieur à PCA pour les clusters asymétriques (poteaux, antennes en L/T) :
    - PCA → axe de variance max (optimal pour ellipsoïdes, pas pour rectangles)
    - MAR → rectangle de superficie minimale (toujours la boîte la plus serrée)

    Algorithme : ConvexHull XY + rotating calipers (teste chaque arête du hull)
    Returns: center(3), width, length, height, yaw
    """
    pts_2d = points[:, :2]

    # Minimum Area Rectangle via rotating calipers sur le convex hull
    best_angle = 0.0
    best_center_2d = pts_2d.mean(axis=0)
    best_w, best_l = 0.1, 0.1

    try:
        hull = ConvexHull(pts_2d)
        hull_pts = pts_2d[hull.vertices]
        n = len(hull_pts)

        min_area = np.inf
        for i in range(n):
            edge = hull_pts[(i + 1) % n] - hull_pts[i]
            angle = np.arctan2(edge[1], edge[0])
            c, s = np.cos(-angle), np.sin(-angle)
            R = np.array([[c, -s], [s, c]])
            rotated = hull_pts @ R.T

            mins_r = rotated.min(axis=0)
            maxs_r = rotated.max(axis=0)
            w, l = maxs_r[0] - mins_r[0], maxs_r[1] - mins_r[1]
            area = w * l

            if area < min_area:
                min_area = area
                best_angle = angle
                best_w, best_l = w, l
                # Centre du rectangle dans le frame tourné → frame monde
                center_rot = (mins_r + maxs_r) / 2
                c_inv, s_inv = np.cos(angle), np.sin(angle)
                R_inv = np.array([[c_inv, -s_inv], [s_inv, c_inv]])
                best_center_2d = center_rot @ R_inv.T

    except Exception:
        # Fallback AABB si ConvexHull échoue (cluster quasi-colinéaire)
        mins_2d = pts_2d.min(axis=0)
        maxs_2d = pts_2d.max(axis=0)
        best_center_2d = (mins_2d + maxs_2d) / 2
        best_w = maxs_2d[0] - mins_2d[0]
        best_l = maxs_2d[1] - mins_2d[1]
        best_angle = 0.0

    # Hauteur et centre vertical depuis Z brut
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    height = max(z_max - z_min, 0.01)
    center_z = (z_min + z_max) / 2

    center = np.array([best_center_2d[0], best_center_2d[1], center_z])
    return center, best_w, best_l, height, best_angle


def cable_cluster_geometry(points):
    """Estimate a 2D line proxy for one cable cluster."""
    pts_xy = points[:, :2]
    center_xy = pts_xy.mean(axis=0)
    centered_xy = pts_xy - center_xy

    if len(points) >= 2:
        _, _, vh = np.linalg.svd(centered_xy, full_matrices=False)
        direction = vh[0]
    else:
        direction = np.array([1.0, 0.0], dtype=np.float32)

    direction = direction.astype(np.float32)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction /= norm

    proj = centered_xy @ direction
    idx_min = int(np.argmin(proj))
    idx_max = int(np.argmax(proj))

    end_a_xy = center_xy + proj[idx_min] * direction
    end_b_xy = center_xy + proj[idx_max] * direction

    end_a = np.array([end_a_xy[0], end_a_xy[1], points[idx_min, 2]], dtype=np.float32)
    end_b = np.array([end_b_xy[0], end_b_xy[1], points[idx_max, 2]], dtype=np.float32)

    return {
        "center_xy": center_xy.astype(np.float32),
        "direction": direction,
        "endpoints": (end_a, end_b),
    }


def cable_merge_metrics(cluster_a, cluster_b):
    """Return geometric compatibility metrics for two cable clusters."""
    dot = float(np.clip(np.dot(cluster_a["direction"], cluster_b["direction"]), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(abs(dot))))

    aligned_dir = cluster_b["direction"] if dot >= 0 else -cluster_b["direction"]
    axis = cluster_a["direction"] + aligned_dir
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        axis = cluster_a["direction"]
    else:
        axis = axis / axis_norm

    perp = np.array([-axis[1], axis[0]], dtype=np.float32)
    lateral_offset = abs(np.dot(cluster_b["center_xy"] - cluster_a["center_xy"], perp))

    best_gap_xy = np.inf
    best_gap_z = np.inf
    for end_a in cluster_a["endpoints"]:
        for end_b in cluster_b["endpoints"]:
            gap_xy = float(np.linalg.norm(end_a[:2] - end_b[:2]))
            if gap_xy < best_gap_xy:
                best_gap_xy = gap_xy
                best_gap_z = abs(float(end_a[2] - end_b[2]))

    return angle_deg, lateral_offset, best_gap_xy, best_gap_z


def merge_cable_clusters(clusters, density, cfg):
    """Merge fragmented cable clusters when geometry strongly suggests continuity."""
    if len(clusters) <= 1:
        return clusters
    if not getattr(cfg, "cable_merge_enabled", False):
        return clusters
    if density > getattr(cfg, "cable_merge_max_density", 1.0):
        return clusters

    density_scale = 1.0 + 0.75 * max(0.0, 1.0 - density)
    max_angle = float(getattr(cfg, "cable_merge_max_angle_deg", 12.0))
    max_gap_xy = float(getattr(cfg, "cable_merge_max_endpoint_gap", 3.0)) * density_scale
    max_lateral = float(getattr(cfg, "cable_merge_max_lateral_offset", 1.25)) * (1.0 + 0.35 * (1.0 - density))
    max_gap_z = float(getattr(cfg, "cable_merge_max_z_gap", 1.0)) * (1.0 + 0.50 * (1.0 - density))

    parent = list(range(len(clusters)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            angle_deg, lateral_offset, gap_xy, gap_z = cable_merge_metrics(
                clusters[i], clusters[j]
            )
            if angle_deg > max_angle:
                continue
            if lateral_offset > max_lateral:
                continue
            if gap_xy > max_gap_xy:
                continue
            if gap_z > max_gap_z:
                continue
            union(i, j)

    merged = {}
    for idx, cluster in enumerate(clusters):
        root = find(idx)
        merged.setdefault(root, []).append(cluster)

    merged_clusters = []
    for group in merged.values():
        if len(group) == 1:
            merged_clusters.append(group[0])
            continue

        merged_points = np.vstack([g["points"] for g in group]).astype(np.float32)
        merged_conf = max(g["confidence"] for g in group)
        geometry = cable_cluster_geometry(merged_points)
        merged_clusters.append({
            "points": merged_points,
            "confidence": merged_conf,
            **geometry,
        })

    return merged_clusters


def detection_from_points(points, class_id, cfg):
    """Convert one cluster of points into the Airbus CSV bbox format."""
    center, w, l, h, yaw = compute_obb(points)
    return {
        "class_ID":      class_id,
        "class_label":   cfg.class_labels[class_id],
        "bbox_center_x": float(center[0]),
        "bbox_center_y": float(center[1]),
        "bbox_center_z": float(center[2]),
        "bbox_width":    float(w),
        "bbox_length":   float(l),
        "bbox_height":   float(h),
        "bbox_yaw":      float(yaw),
    }


def get_cluster_settings(class_id, density, cfg):
    """Prepare class-specific clustering settings for the current density."""
    conf_thresholds = getattr(cfg, "confidence_threshold", {})
    params = cfg.dbscan_params[class_id]

    min_pts = cfg.min_cluster_points.get(class_id, 10)
    min_conf = conf_thresholds.get(class_id, 0.0)
    min_samples = params["min_samples"]

    density_overrides = {}
    if class_id == 0:
        density_overrides = getattr(cfg, "antenna_density_params", {})
    elif class_id == 1:
        density_overrides = getattr(cfg, "cable_density_params", {})

    if density_overrides and density < 0.99:
        for threshold in sorted(density_overrides.keys()):
            if density <= threshold:
                override = density_overrides[threshold]
                min_samples = override["min_samples"]
                min_pts = override["min_cluster"]
                min_conf = override["confidence"]
                break

    cluster_space = getattr(cfg, "cluster_space", {}).get(
        class_id, "xy" if class_id in (0, 2) else "xyz"
    )
    core_threshold = getattr(cfg, "cluster_core_threshold", {}).get(class_id, min_conf)
    support_threshold = getattr(
        cfg, "cluster_support_threshold", {}
    ).get(class_id, max(min_conf * 0.6, 0.20))
    min_core_points = getattr(
        cfg, "cluster_min_core_points", {}
    ).get(class_id, max(2, min_pts // 4))
    support_radius = getattr(
        cfg, "cluster_support_radius", {}
    ).get(class_id, params["eps"])

    if density < 1.0:
        conf_scale = 0.90 + 0.10 * density
        core_threshold *= conf_scale
        support_threshold *= conf_scale
        min_core_points = max(
            int(np.ceil(min_core_points * max(density ** 0.5, 0.60))),
            2,
        )

    core_threshold = float(np.clip(core_threshold, 0.05, 0.95))
    support_threshold = float(
        np.clip(min(support_threshold, core_threshold), 0.05, core_threshold)
    )

    return {
        "eps": params["eps"],
        "min_samples": min_samples,
        "min_pts": min_pts,
        "min_conf": min_conf,
        "cluster_space": cluster_space,
        "core_threshold": core_threshold,
        "support_threshold": support_threshold,
        "min_core_points": min_core_points,
        "support_radius": max(float(support_radius), float(params["eps"])),
    }


def build_candidate_masks(xyz, pred_classes, probs, class_id, settings):
    """Select core points plus nearby support points for one class."""
    if probs is None:
        mask = pred_classes == class_id
        return mask, mask

    class_scores = probs[:, class_id]
    core_mask = (pred_classes == class_id) | (class_scores >= settings["core_threshold"])
    if not core_mask.any():
        return core_mask, core_mask

    support_mask = class_scores >= settings["support_threshold"]
    candidate_mask = core_mask.copy()

    extra_support_idx = np.flatnonzero(support_mask & ~candidate_mask)
    if extra_support_idx.size > 0:
        core_tree = cKDTree(xyz[core_mask])
        distances, _ = core_tree.query(xyz[extra_support_idx], k=1)
        candidate_mask[extra_support_idx[distances <= settings["support_radius"]]] = True

    return candidate_mask, core_mask


def cluster_and_bbox(xyz, pred_classes, cfg, probs=None, density=1.0):
    """Extract obstacle instances and convert them to oriented boxes."""
    detections = []

    for class_id in range(4):
        settings = get_cluster_settings(class_id, density, cfg)
        candidate_mask, core_mask = build_candidate_masks(
            xyz, pred_classes, probs, class_id, settings
        )

        if candidate_mask.sum() < settings["min_pts"]:
            continue

        points = xyz[candidate_mask]
        cls_probs = probs[candidate_mask, class_id] if probs is not None else None
        local_core_mask = core_mask[candidate_mask]
        cluster_input = points[:, :2] if settings["cluster_space"] == "xy" else points

        clustering = DBSCAN(
            eps=settings["eps"],
            min_samples=settings["min_samples"],
            n_jobs=-1,
        ).fit(cluster_input)

        labels = clustering.labels_
        unique_labels = set(labels) - {-1}
        cable_clusters = []

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = points[cluster_mask]

            if len(cluster_points) < settings["min_pts"]:
                continue

            if int(local_core_mask[cluster_mask].sum()) < settings["min_core_points"]:
                continue

            if cls_probs is not None and settings["min_conf"] > 0.0:
                confidence = float(cls_probs[cluster_mask].mean())
                if confidence < settings["min_conf"]:
                    continue
            else:
                confidence = 0.0

            if class_id == 1:
                geometry = cable_cluster_geometry(cluster_points)
                cable_clusters.append({
                    "points": cluster_points.astype(np.float32),
                    "confidence": confidence,
                    **geometry,
                })
            else:
                detections.append(detection_from_points(cluster_points, class_id, cfg))

        if class_id == 1 and cable_clusters:
            cable_clusters = merge_cable_clusters(cable_clusters, density, cfg)
            for cluster in cable_clusters:
                detections.append(detection_from_points(cluster["points"], class_id, cfg))

    return detections


def process_file(
    model,
    input_path,
    output_path,
    device,
    cfg,
    use_tta=True,
    density=1.0,
    simulate_density=None,
):
    """Traite un fichier HDF5 complet et génère le CSV.

    density : densité effective [0.25-1.0] utilisée pour adapter le clustering.
              1.0 = 100% des points, 0.25 = 25% des points.
    simulate_density : si non-None, applique en plus un sous-échantillonnage aléatoire
                       pour simuler une densité plus faible à partir du fichier d'entrée.
    """
    print(f"\nTraitement: {input_path}")
    print(f"  Densité prise en compte: {density*100:.0f}%")
    if simulate_density is not None and simulate_density < 1.0:
        print(f"  Sous-échantillonnage simulé: {simulate_density*100:.0f}% des points")

    df = lidar_utils.load_h5_data(input_path)
    print(f"  Points totaux: {len(df):,}")

    # Filtrer tirs invalides
    df = df[df["distance_cm"] > 0].copy()
    print(f"  Points valides: {len(df):,}")

    poses = lidar_utils.get_unique_poses(df)
    print(f"  Frames: {len(poses)}")

    all_rows = []

    for _, pose_row in poses.iterrows():
        pose_idx = int(pose_row["pose_index"])
        frame_df = lidar_utils.filter_by_pose(df, pose_row)

        if len(frame_df) < 100:
            continue

        # Convertir en XYZ
        xyz = lidar_utils.spherical_to_local_cartesian(frame_df).astype(np.float32)
        refl = frame_df["reflectivity"].values.astype(np.float32)
        dist = frame_df["distance_cm"].values.astype(np.float32)

        # Voxel downsampling (aligne inference sur training : 575K → ~80K comme prep_data.py)
        # Sans ça : inference 25% = 143K points → 65K tous réels
        #           training 25% = 20K points → 65K (45K padded copies) → distributions ≠
        if getattr(cfg, 'voxel_inference', True):
            n_before = len(xyz)
            xyz, refl, dist = voxel_downsample_inference(xyz, refl, dist, cfg.voxel_size)
            if pose_idx == 0:
                print(f"  Voxel downsample: {n_before:,} → {len(xyz):,} points")

        # Features de base (xyz + refl + dist) sur frame voxelisé
        features = prepare_features(xyz, refl, dist, cfg)  # (N, 5)

        # Réduction de densité synthétique (uniquement si explicitement demandée).
        # Les fichiers officiels Airbus _25/_50/_75 sont déjà downsamplés :
        # on ne doit PAS les sous-échantillonner une seconde fois.
        if simulate_density is not None and simulate_density < 1.0:
            np.random.seed(pose_idx)  # Reproductible par frame
            n_keep = max(int(len(xyz) * simulate_density), 100)
            keep_idx = np.random.choice(len(xyz), n_keep, replace=False)
            xyz      = xyz[keep_idx]
            features = features[keep_idx]
            features[:, :3] = xyz  # Mettre à jour les coordonnées XYZ dans les features

        # Features géométriques calculées sur les points courants (après density drop)
        # Run 10: precomputed features dans le pipeline normal, on-the-fly ici pour
        # rester cohérent avec ce que le modèle a vu (features sur données voxelisées)
        if cfg.use_geometric_features:
            from prep_data import compute_geometric_features
            use_vert = getattr(cfg, "use_verticality_feature", False)
            if use_vert:
                lin, plan, vert = compute_geometric_features(
                    xyz, k=cfg.k_geometric, compute_verticality=True
                )
                geo = np.column_stack([lin, plan, vert]).astype(np.float32)
            else:
                lin, plan = compute_geometric_features(xyz, k=cfg.k_geometric)
                geo = np.column_stack([lin, plan]).astype(np.float32)
            features = np.hstack([features, geo])  # (N, 8)

        # Segmentation
        probs = segment_frame(model, xyz, features, device, cfg, use_tta)
        pred_classes = probs.argmax(axis=1)

        # Clustering + BBox (DBSCAN adaptatif selon densité)
        detections = cluster_and_bbox(xyz, pred_classes, cfg, probs=probs, density=density)

        # Ajouter les infos de pose
        for det in detections:
            det["ego_x"] = float(pose_row["ego_x"])
            det["ego_y"] = float(pose_row["ego_y"])
            det["ego_z"] = float(pose_row["ego_z"])
            det["ego_yaw"] = float(pose_row["ego_yaw"])
            all_rows.append(det)

        n_det = len(detections)
        if n_det > 0:
            print(f"  Frame {pose_idx:3d}: {n_det} détections")

    # Générer CSV
    if all_rows:
        result_df = pd.DataFrame(all_rows)
        # Ordonner les colonnes selon le format Airbus
        columns = [
            "ego_x", "ego_y", "ego_z", "ego_yaw",
            "bbox_center_x", "bbox_center_y", "bbox_center_z",
            "bbox_width", "bbox_length", "bbox_height",
            "bbox_yaw", "class_ID", "class_label",
        ]
        result_df = result_df[columns]
        result_df.to_csv(output_path, index=False)
        print(f"\n✓ CSV sauvegardé: {output_path}")
        print(f"  {len(result_df)} détections totales")
    else:
        pd.DataFrame(columns=[
            "ego_x", "ego_y", "ego_z", "ego_yaw",
            "bbox_center_x", "bbox_center_y", "bbox_center_z",
            "bbox_width", "bbox_length", "bbox_height",
            "bbox_yaw", "class_ID", "class_label",
        ]).to_csv(output_path, index=False)
        print(f"\n⚠ Aucune détection. CSV vide: {output_path}")

    return len(all_rows)


def resolve_input_density(input_path):
    """Détermine la densité native du fichier d'entrée.

    Détection via suffixes officiels Airbus (_25/_50/_75/_100.h5).
    Cette valeur sert à adapter le clustering, mais ne déclenche pas de
    sous-échantillonnage supplémentaire.
    """
    filename = os.path.basename(input_path).lower()
    suffix_map = {
        "_25.h5": 0.25,
        "_50.h5": 0.50,
        "_75.h5": 0.75,
        "_100.h5": 1.00,
    }
    for suffix, value in suffix_map.items():
        if filename.endswith(suffix):
            print(f"Densité d'entrée auto-détectée depuis le nom de fichier: {value:.0%}")
            return value

    return 1.0


def main():
    parser = argparse.ArgumentParser(description="Inference RandLA-Net")
    parser.add_argument("--input", required=True, help="Fichier HDF5 d'entrée")
    parser.add_argument("--output", required=True, help="Fichier CSV de sortie")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth",
                        help="Chemin du checkpoint")
    parser.add_argument("--no-tta", action="store_true",
                        help="Désactiver Test-Time Augmentation")
    parser.add_argument("--density", type=float, default=None,
                        help="Simule une densité cible [0.25|0.50|0.75|1.0] "
                             "par sous-échantillonnage supplémentaire. "
                             "À utiliser pour tester la robustesse depuis un fichier 100%.")
    args = parser.parse_args()

    cfg = Config()
    cfg.ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Clustering: probabilistic-v3")

    model = load_model(args.checkpoint, device, cfg)
    input_density = resolve_input_density(args.input)
    effective_density = args.density if args.density is not None else input_density
    process_file(model, args.input, args.output, device, cfg,
                 use_tta=not args.no_tta,
                 density=effective_density,
                 simulate_density=args.density)


if __name__ == "__main__":
    main()

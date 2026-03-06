"""
compute_map.py — Évaluation locale des métriques Airbus Hackathon 2026

Métriques calculées (identiques aux critères officiels) :
  1. mAP @ IoU=0.5  — Mean Average Precision, seuil TP/FP = 0.5
  2. Mean IoU (Correct Class) — IoU moyen des boîtes correctement classées (TP)
  3. Par classe : Precision, Recall, F1, Mean IoU

IoU 3D exact (pour boîtes avec rotation yaw uniquement) :
  - Intersection 2D en BEV via Sutherland-Hodgman (polygone exact)
  - × overlap en Z
  → Valide pour des obstacles sans pitch/roll significatif

Usage :
  # Évaluer sur une scène (toutes les frames)
  python compute_map.py --scene airbus_hackathon_trainingdata/scene_1.h5 \\
                        --pred predictions/test_scene1.csv

  # Évaluer sur plusieurs scènes
  python compute_map.py --scene scene_1.h5 --pred pred_1.csv \\
                        --scene scene_2.h5 --pred pred_2.csv
"""

import argparse
import numpy as np
import pandas as pd

import lidar_utils
from config import Config
from inference import compute_obb

# Importé ici pour ne pas dupliquer le COLOR_MAP
COLOR_MAP = {
    (38, 23, 180): 0,    # Antenna
    (177, 132, 47): 1,   # Cable
    (129, 81, 97): 2,    # Electric pole
    (66, 132, 9): 3,     # Wind turbine
}
CLASS_NAMES = {0: "Antenna", 1: "Cable", 2: "Electric Pole", 3: "Wind Turbine"}


# ═══════════════════════════════════════════════════════════════════
# GÉOMÉTRIE — IoU 3D exact (Sutherland-Hodgman + Z overlap)
# ═══════════════════════════════════════════════════════════════════

def _is_inside(point, a, b):
    """Retourne True si point est dans le demi-plan gauche de l'arête a→b."""
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= 0


def _line_intersect(p1, p2, a, b):
    """Intersection de la droite p1-p2 et de la droite a-b."""
    dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
    dx2, dy2 = b[0] - a[0], b[1] - a[1]
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-10:
        return p1
    t = ((a[0] - p1[0]) * dy2 - (a[1] - p1[1]) * dx2) / denom
    return (p1[0] + t * dx1, p1[1] + t * dy1)


def _clip_by_halfplane(polygon, a, b):
    """Clip un polygone convexe contre le demi-plan gauche de l'arête a→b."""
    result = []
    n = len(polygon)
    for i in range(n):
        curr = polygon[i]
        prev = polygon[i - 1]
        curr_in = _is_inside(curr, a, b)
        prev_in = _is_inside(prev, a, b)
        if curr_in:
            if not prev_in:
                result.append(_line_intersect(prev, curr, a, b))
            result.append(curr)
        elif prev_in:
            result.append(_line_intersect(prev, curr, a, b))
    return result


def _polygon_area(polygon):
    """Aire d'un polygone via la formule du lacet (shoelace)."""
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def _obb_corners_2d(cx, cy, width, length, yaw):
    """Retourne les 4 coins d'un OBB 2D en ordre anti-horaire."""
    hw, hl = width / 2.0, length / 2.0
    c, s = np.cos(yaw), np.sin(yaw)
    corners = [
        (cx + c * (-hw) - s * (-hl), cy + s * (-hw) + c * (-hl)),
        (cx + c * hw    - s * (-hl), cy + s * hw    + c * (-hl)),
        (cx + c * hw    - s * hl,    cy + s * hw    + c * hl),
        (cx + c * (-hw) - s * hl,    cy + s * (-hw) + c * hl),
    ]
    return corners


def iou_3d(pred, gt):
    """
    IoU 3D exact entre deux boîtes OBB (rotation yaw uniquement).

    pred, gt : dicts avec clés bbox_center_x/y/z, bbox_width, bbox_length,
                          bbox_height, bbox_yaw
    """
    cx1, cy1, cz1 = pred["bbox_center_x"], pred["bbox_center_y"], pred["bbox_center_z"]
    w1, l1, h1    = pred["bbox_width"],    pred["bbox_length"],   pred["bbox_height"]
    yaw1          = pred["bbox_yaw"]

    cx2, cy2, cz2 = gt["bbox_center_x"], gt["bbox_center_y"], gt["bbox_center_z"]
    w2, l2, h2    = gt["bbox_width"],    gt["bbox_length"],   gt["bbox_height"]
    yaw2          = gt["bbox_yaw"]

    # Overlap en Z
    z1_lo, z1_hi = cz1 - h1 / 2, cz1 + h1 / 2
    z2_lo, z2_hi = cz2 - h2 / 2, cz2 + h2 / 2
    z_inter = max(0.0, min(z1_hi, z2_hi) - max(z1_lo, z2_lo))
    if z_inter == 0.0:
        return 0.0

    # Intersection 2D (Sutherland-Hodgman)
    corners1 = _obb_corners_2d(cx1, cy1, w1, l1, yaw1)
    corners2 = _obb_corners_2d(cx2, cy2, w2, l2, yaw2)

    clipped = list(corners1)
    n = len(corners2)
    for i in range(n):
        if not clipped:
            return 0.0
        clipped = _clip_by_halfplane(clipped, corners2[i], corners2[(i + 1) % n])

    if len(clipped) < 3:
        return 0.0

    area_inter = _polygon_area(clipped)
    inter_vol  = area_inter * z_inter

    vol1 = w1 * l1 * h1
    vol2 = w2 * l2 * h2
    union_vol = vol1 + vol2 - inter_vol

    return inter_vol / max(union_vol, 1e-10)


# ═══════════════════════════════════════════════════════════════════
# EXTRACTION GROUND TRUTH depuis HDF5
# ═══════════════════════════════════════════════════════════════════

def _assign_labels(df):
    """RGB → class_id (vectorisé)."""
    labels = np.full(len(df), 4, dtype=np.int64)  # 4 = background
    r, g, b = df["r"].values, df["g"].values, df["b"].values
    for (cr, cg, cb), cls in COLOR_MAP.items():
        mask = (r == cr) & (g == cg) & (b == cb)
        labels[mask] = cls
    return labels


def extract_gt_boxes(frame_df, cfg):
    """
    Extrait les boîtes GT d'une frame via DBSCAN + MAR OBB.

    Utilise cfg.gt_dbscan_params et cfg.gt_min_cluster_points (params FIXES)
    pour que la GT reste stable quand on tune les params d'inférence.

    ⚠️ Ne JAMAIS utiliser cfg.dbscan_params ici — ça introduirait un couplage
    entre GT et inférence qui rend les runs incomparables.

    Returns: liste de dicts (même format que le CSV prédit)
    """
    from sklearn.cluster import DBSCAN as _DBSCAN

    # Params GT fixes (indépendants des params d'inférence)
    gt_params   = getattr(cfg, "gt_dbscan_params",      cfg.dbscan_params)
    gt_min_pts  = getattr(cfg, "gt_min_cluster_points", cfg.min_cluster_points)

    xyz    = lidar_utils.spherical_to_local_cartesian(frame_df).astype(np.float32)
    labels = _assign_labels(frame_df)

    gt_boxes = []
    for class_id in range(4):
        mask = labels == class_id
        if mask.sum() < gt_min_pts.get(class_id, 10):
            continue

        points = xyz[mask]
        params = gt_params[class_id]
        clustering = _DBSCAN(
            eps=params["eps"], min_samples=params["min_samples"], n_jobs=-1
        ).fit(points)

        for label in set(clustering.labels_) - {-1}:
            cluster_pts = points[clustering.labels_ == label]
            if len(cluster_pts) < gt_min_pts.get(class_id, 10):
                continue
            center, w, l, h, yaw = compute_obb(cluster_pts)
            gt_boxes.append({
                "class_ID":      class_id,
                "class_label":   CLASS_NAMES[class_id],
                "bbox_center_x": float(center[0]),
                "bbox_center_y": float(center[1]),
                "bbox_center_z": float(center[2]),
                "bbox_width":    float(w),
                "bbox_length":   float(l),
                "bbox_height":   float(h),
                "bbox_yaw":      float(yaw),
            })
    return gt_boxes


# ═══════════════════════════════════════════════════════════════════
# MATCHING & MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════

def match_frame(preds, gts, iou_threshold=0.5):
    """
    Matching greedy entre prédictions et GT d'une frame.
    Matching par classe : une pred ne peut matcher qu'un GT de même classe.

    Returns:
      tp_ious : liste de IoU pour chaque TP (pour Mean IoU Correct Class)
      n_fp    : nombre de faux positifs
      n_fn    : nombre de faux négatifs
    """
    tp_ious = []
    n_fp    = 0
    n_fn    = 0

    for class_id in range(4):
        pred_c = [p for p in preds if p["class_ID"] == class_id]
        gt_c   = [g for g in gts   if g["class_ID"] == class_id]

        if not gt_c:
            n_fp += len(pred_c)
            continue
        if not pred_c:
            n_fn += len(gt_c)
            continue

        # Matrice IoU (n_pred × n_gt)
        iou_matrix = np.zeros((len(pred_c), len(gt_c)))
        for i, p in enumerate(pred_c):
            for j, g in enumerate(gt_c):
                iou_matrix[i, j] = iou_3d(p, g)

        matched_pred = set()
        matched_gt   = set()

        # Greedy : paire avec IoU max en premier
        flat_order = np.argsort(iou_matrix.ravel())[::-1]
        for idx in flat_order:
            i, j = divmod(int(idx), len(gt_c))
            if i in matched_pred or j in matched_gt:
                continue
            if iou_matrix[i, j] >= iou_threshold:
                tp_ious.append(iou_matrix[i, j])
                matched_pred.add(i)
                matched_gt.add(j)
            else:
                break  # Matrice triée : aucun meilleur candidat possible

        n_fp += len(pred_c) - len(matched_pred)
        n_fn += len(gt_c)   - len(matched_gt)

    return tp_ious, n_fp, n_fn


# ═══════════════════════════════════════════════════════════════════
# ÉVALUATION D'UNE SCÈNE
# ═══════════════════════════════════════════════════════════════════

def evaluate_scene(scene_path, pred_csv_path, cfg, iou_threshold=0.5):
    """
    Évalue une scène complète.
    Returns: dict avec les métriques globales et par classe.
    """
    print(f"\n{'─'*60}")
    print(f"Scène : {scene_path}")
    print(f"Preds : {pred_csv_path}")
    print(f"{'─'*60}")

    # Charger données
    df_scene  = lidar_utils.load_h5_data(scene_path)
    df_scene  = df_scene[df_scene["distance_cm"] > 0].copy()
    pred_df   = pd.read_csv(pred_csv_path)
    poses     = lidar_utils.get_unique_poses(df_scene)

    # Accumulateurs par classe
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0} for c in range(4)}
    global_tp_ious = []

    n_frames_with_gt   = 0
    n_frames_with_pred = 0

    for _, pose_row in poses.iterrows():
        frame_df = lidar_utils.filter_by_pose(df_scene, pose_row)
        if len(frame_df) < 100:
            continue

        # GT boxes de cette frame
        gt_boxes = extract_gt_boxes(frame_df, cfg)

        # Prédictions de cette frame (match par ego pose)
        frame_preds_df = pred_df[
            (pred_df["ego_x"]   == pose_row["ego_x"]) &
            (pred_df["ego_y"]   == pose_row["ego_y"]) &
            (pred_df["ego_z"]   == pose_row["ego_z"]) &
            (pred_df["ego_yaw"] == pose_row["ego_yaw"])
        ]
        pred_boxes = frame_preds_df.to_dict("records")

        if gt_boxes:
            n_frames_with_gt += 1
        if pred_boxes:
            n_frames_with_pred += 1

        # Matching frame par frame, classe par classe
        for class_id in range(4):
            pred_c = [p for p in pred_boxes if p["class_ID"] == class_id]
            gt_c   = [g for g in gt_boxes   if g["class_ID"] == class_id]

            if not gt_c:
                per_class[class_id]["fp"] += len(pred_c)
                continue
            if not pred_c:
                per_class[class_id]["fn"] += len(gt_c)
                continue

            iou_matrix = np.zeros((len(pred_c), len(gt_c)))
            for i, p in enumerate(pred_c):
                for j, g in enumerate(gt_c):
                    iou_matrix[i, j] = iou_3d(p, g)

            matched_pred = set()
            matched_gt   = set()
            flat_order   = np.argsort(iou_matrix.ravel())[::-1]

            for idx in flat_order:
                i, j = divmod(int(idx), len(gt_c))
                if i in matched_pred or j in matched_gt:
                    continue
                if iou_matrix[i, j] >= iou_threshold:
                    per_class[class_id]["tp"]      += 1
                    per_class[class_id]["iou_sum"] += iou_matrix[i, j]
                    global_tp_ious.append(iou_matrix[i, j])
                    matched_pred.add(i)
                    matched_gt.add(j)
                else:
                    break

            per_class[class_id]["fp"] += len(pred_c) - len(matched_pred)
            per_class[class_id]["fn"] += len(gt_c)   - len(matched_gt)

    # ─── Rapport ──────────────────────────────────────────────────
    print(f"\n  Frames traitées : {len(poses)} | avec GT: {n_frames_with_gt} | avec preds: {n_frames_with_pred}")
    print(f"\n  {'Classe':<18} {'TP':>5} {'FP':>5} {'FN':>5} "
          f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'MeanIoU':>9}")
    print(f"  {'─'*65}")

    ap_values    = []
    mean_ious    = []

    for class_id in range(4):
        d  = per_class[class_id]
        tp, fp, fn = d["tp"], d["fp"], d["fn"]

        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        m_iou  = d["iou_sum"] / tp if tp > 0 else 0.0

        ap_values.append(prec)   # AP à point fixe (sans confidence scores)
        if tp > 0:
            mean_ious.append(m_iou)

        print(f"  {CLASS_NAMES[class_id]:<18} {tp:>5} {fp:>5} {fn:>5} "
              f"{prec:>7.3f} {recall:>7.3f} {f1:>7.3f} {m_iou:>9.4f}")

    mAP          = np.mean(ap_values)
    mean_iou_cc  = np.mean(mean_ious) if mean_ious else 0.0
    global_tp    = sum(d["tp"] for d in per_class.values())
    global_fp    = sum(d["fp"] for d in per_class.values())
    global_fn    = sum(d["fn"] for d in per_class.values())

    print(f"\n  {'─'*65}")
    print(f"  {'GLOBAL':<18} {global_tp:>5} {global_fp:>5} {global_fn:>5}")
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  mAP @ IoU=0.5          : {mAP:>6.4f}        │")
    print(f"  │  Mean IoU (Correct Cls) : {mean_iou_cc:>6.4f}        │")
    print(f"  └─────────────────────────────────────────┘")

    if global_tp_ious:
        iou_arr = np.array(global_tp_ious)
        print(f"\n  Distribution IoU des TP :")
        print(f"    min={iou_arr.min():.3f}  p25={np.percentile(iou_arr,25):.3f}  "
              f"median={np.median(iou_arr):.3f}  p75={np.percentile(iou_arr,75):.3f}  "
              f"max={iou_arr.max():.3f}")

    return {
        "mAP":           mAP,
        "mean_iou_cc":   mean_iou_cc,
        "per_class":     per_class,
        "tp_ious":       global_tp_ious,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Évaluation mAP + Mean IoU — Airbus Hackathon")
    parser.add_argument("--scene", action="append", required=True,
                        help="Fichier HDF5 de la scène (répétable)")
    parser.add_argument("--pred",  action="append", required=True,
                        help="CSV de prédictions correspondant (répétable)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="Seuil IoU pour TP/FP (défaut: 0.5)")
    args = parser.parse_args()

    if len(args.scene) != len(args.pred):
        parser.error("--scene et --pred doivent être fournis en nombre égal")

    cfg = Config()

    all_mAP        = []
    all_mean_iou   = []

    for scene_path, pred_path in zip(args.scene, args.pred):
        result = evaluate_scene(scene_path, pred_path, cfg, args.iou_threshold)
        all_mAP.append(result["mAP"])
        all_mean_iou.append(result["mean_iou_cc"])

    if len(args.scene) > 1:
        print(f"\n{'═'*60}")
        print(f"RÉSUMÉ GLOBAL ({len(args.scene)} scènes)")
        print(f"  mAP @ IoU={args.iou_threshold}        : {np.mean(all_mAP):.4f}")
        print(f"  Mean IoU (Correct Class) : {np.mean(all_mean_iou):.4f}")
        print(f"{'═'*60}")


if __name__ == "__main__":
    main()

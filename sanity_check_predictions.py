"""
Sanity check rapide des CSV de prédictions sans ground truth.

Le script:
  - associe chaque CSV de prédiction à son HDF5 source
  - calcule des stats globales par fichier
  - calcule des stats par frame (via pose_index)
  - repère les frames suspectes (sur-densité, trop de câbles, câbles anormalement longs)
  - génère des commandes `visualize.py` prêtes à lancer

Usage:
  python sanity_check_predictions.py
  python sanity_check_predictions.py --pred-dir predictions/eval --scene-dir airbus_hackathon_evalset
  python sanity_check_predictions.py --pred predictions/eval/eval_sceneB_25.csv --top-k 15
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import lidar_utils


POSE_FIELDS = ["ego_x", "ego_y", "ego_z", "ego_yaw"]
CLASS_COLUMNS = {
    "Antenna": "antenna_count",
    "Cable": "cable_count",
    "Electric Pole": "pole_count",
    "Wind Turbine": "turbine_count",
}
POSE_POSITION_TOL = 1e-2
POSE_YAW_TOL = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check des prédictions Airbus")
    parser.add_argument(
        "--scene-dir",
        default="airbus_hackathon_evalset",
        help="Dossier contenant les HDF5 source",
    )
    parser.add_argument(
        "--pred-dir",
        default="predictions/eval",
        help="Dossier contenant les CSV de prédictions",
    )
    parser.add_argument(
        "--pred",
        action="append",
        help="CSV précis à analyser (répétable). Si omis, scanne pred-dir/eval_scene*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="sanity_reports",
        help="Dossier de sortie des rapports",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Nombre de frames suspectes à remonter par fichier",
    )
    return parser.parse_args()


def find_prediction_files(args: argparse.Namespace) -> list[Path]:
    if args.pred:
        return [Path(p) for p in args.pred]

    pred_dir = Path(args.pred_dir)
    return sorted(
        p for p in pred_dir.glob("eval_scene*.csv")
        if p.name != "summary.csv"
    )


def safe_zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def flag_string(row: pd.Series) -> str:
    flags = []
    if row["flag_dense_total"]:
        flags.append("dense-total")
    if row["flag_cable_dense"]:
        flags.append("dense-cable")
    if row["flag_long_cable"]:
        flags.append("long-cable")
    if row["flag_cable_dominant"]:
        flags.append("cable-dominant")
    if row["flag_antenna_swarm"]:
        flags.append("antenna-swarm")
    if row["flag_no_detection"]:
        flags.append("no-detection")
    return ",".join(flags)


def attach_pose_index(pred_df: pd.DataFrame, poses: pd.DataFrame) -> pd.DataFrame:
    """Attach pose_index to predictions with tolerance for CSV float roundtrips."""
    pred_with_pose = pred_df.merge(
        poses[["pose_index", *POSE_FIELDS]],
        on=POSE_FIELDS,
        how="left",
        validate="many_to_one",
    )
    if not pred_with_pose["pose_index"].isna().any():
        pred_with_pose["pose_index"] = pred_with_pose["pose_index"].astype(int)
        return pred_with_pose

    pose_values = poses[POSE_FIELDS].to_numpy(dtype=np.float64)
    unique_pred_poses = pred_df[POSE_FIELDS].drop_duplicates().reset_index(drop=True)
    pose_mapping = []
    unresolved = []

    for _, pose_row in unique_pred_poses.iterrows():
        pred_pose = pose_row[POSE_FIELDS].to_numpy(dtype=np.float64)
        deltas = np.abs(pose_values - pred_pose)

        mask = (
            (deltas[:, 0] <= POSE_POSITION_TOL)
            & (deltas[:, 1] <= POSE_POSITION_TOL)
            & (deltas[:, 2] <= POSE_POSITION_TOL)
            & (deltas[:, 3] <= POSE_YAW_TOL)
        )

        if not mask.any():
            best_idx = int(np.argmin(deltas[:, :3].max(axis=1) + 100.0 * deltas[:, 3]))
            unresolved.append({
                **{field: float(pose_row[field]) for field in POSE_FIELDS},
                "best_dx": float(deltas[best_idx, 0]),
                "best_dy": float(deltas[best_idx, 1]),
                "best_dz": float(deltas[best_idx, 2]),
                "best_dyaw": float(deltas[best_idx, 3]),
            })
            continue

        candidate_idx = np.where(mask)[0]
        if len(candidate_idx) > 1:
            score = deltas[candidate_idx, :3].sum(axis=1) + 100.0 * deltas[candidate_idx, 3]
            best_local = candidate_idx[int(np.argmin(score))]
        else:
            best_local = int(candidate_idx[0])

        pose_mapping.append({
            **{field: float(pose_row[field]) for field in POSE_FIELDS},
            "pose_index": int(poses.iloc[best_local]["pose_index"]),
        })

    if unresolved:
        example = unresolved[0]
        raise ValueError(
            "Impossible d'associer certaines prédictions à une pose. "
            f"Exemple: ego=({example['ego_x']}, {example['ego_y']}, "
            f"{example['ego_z']}, {example['ego_yaw']}) | "
            f"best deltas=({example['best_dx']:.6f}, {example['best_dy']:.6f}, "
            f"{example['best_dz']:.6f}, {example['best_dyaw']:.6f})"
        )

    pose_mapping_df = pd.DataFrame(pose_mapping)
    pred_with_pose = pred_df.merge(
        pose_mapping_df,
        on=POSE_FIELDS,
        how="left",
        validate="many_to_one",
    )
    if pred_with_pose["pose_index"].isna().any():
        missing = int(pred_with_pose["pose_index"].isna().sum())
        raise ValueError(f"Appariement tolérant incomplet: {missing} prédictions sans pose_index")

    pred_with_pose["pose_index"] = pred_with_pose["pose_index"].astype(int)
    return pred_with_pose


def build_frame_report(scene_path: Path, pred_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    scene_df = lidar_utils.load_h5_data(scene_path)
    scene_df = scene_df[scene_df["distance_cm"] > 0].copy()
    poses = lidar_utils.get_unique_poses(scene_df)
    if poses is None:
        raise ValueError(f"Impossible d'extraire les poses depuis {scene_path}")

    pred_df = pd.read_csv(pred_path)
    if pred_df.empty:
        report = poses[["pose_index", *POSE_FIELDS]].copy()
        for col in CLASS_COLUMNS.values():
            report[col] = 0
        report["total_count"] = 0
        report["cable_max_length"] = 0.0
        report["cable_mean_length"] = 0.0
        report["cable_width_mean"] = 0.0
        report["cable_ratio"] = 0.0
        report["flags"] = "no-detection"
        report["sanity_score"] = 0.0
        summary = {
            "file": pred_path.name,
            "frames": len(report),
            "frames_with_detections": 0,
            "total_detections": 0,
            "antenna_count": 0,
            "cable_count": 0,
            "pole_count": 0,
            "turbine_count": 0,
            "max_total_per_frame": 0,
            "max_cable_per_frame": 0,
            "max_cable_length": 0.0,
        }
        return report, summary

    pred_with_pose = attach_pose_index(pred_df, poses)

    counts = (
        pred_with_pose.groupby(["pose_index", "class_label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns=CLASS_COLUMNS)
    )

    cable_df = pred_with_pose[pred_with_pose["class_label"] == "Cable"]
    cable_stats = (
        cable_df.groupby("pose_index")
        .agg(
            cable_max_length=("bbox_length", "max"),
            cable_mean_length=("bbox_length", "mean"),
            cable_width_mean=("bbox_width", "mean"),
        )
    )

    report = poses[["pose_index", *POSE_FIELDS]].copy()
    report = report.merge(counts, on="pose_index", how="left")
    report = report.merge(cable_stats, on="pose_index", how="left")
    report = report.fillna(0.0)

    for col in CLASS_COLUMNS.values():
        if col not in report.columns:
            report[col] = 0
        report[col] = report[col].astype(int)

    report["total_count"] = report[list(CLASS_COLUMNS.values())].sum(axis=1)
    report["cable_ratio"] = np.where(
        report["total_count"] > 0,
        report["cable_count"] / report["total_count"],
        0.0,
    )

    total_q95 = float(report["total_count"].quantile(0.95))
    cable_q95 = float(report["cable_count"].quantile(0.95))
    cable_len_q95 = float(report["cable_max_length"].quantile(0.95))
    ant_q95 = float(report["antenna_count"].quantile(0.95))

    report["flag_dense_total"] = report["total_count"] >= max(total_q95, 6.0)
    report["flag_cable_dense"] = report["cable_count"] >= max(cable_q95, 4.0)
    report["flag_long_cable"] = report["cable_max_length"] >= max(cable_len_q95, 20.0)
    report["flag_cable_dominant"] = (report["cable_count"] >= 4) & (report["cable_ratio"] >= 0.70)
    report["flag_antenna_swarm"] = report["antenna_count"] >= max(ant_q95, 6.0)
    report["flag_no_detection"] = report["total_count"] == 0

    report["sanity_score"] = (
        1.00 * safe_zscore(report["total_count"]).clip(lower=0.0)
        + 1.25 * safe_zscore(report["cable_count"]).clip(lower=0.0)
        + 0.75 * safe_zscore(report["cable_max_length"]).clip(lower=0.0)
        + 1.50 * report["flag_dense_total"].astype(float)
        + 1.50 * report["flag_cable_dense"].astype(float)
        + 1.00 * report["flag_long_cable"].astype(float)
        + 0.75 * report["flag_cable_dominant"].astype(float)
        + 0.75 * report["flag_antenna_swarm"].astype(float)
        + 0.25 * report["flag_no_detection"].astype(float)
    )
    report["flags"] = report.apply(flag_string, axis=1)
    report = report.sort_values(["sanity_score", "total_count"], ascending=False)

    summary = {
        "file": pred_path.name,
        "frames": len(report),
        "frames_with_detections": int((report["total_count"] > 0).sum()),
        "total_detections": int(report["total_count"].sum()),
        "antenna_count": int(report["antenna_count"].sum()),
        "cable_count": int(report["cable_count"].sum()),
        "pole_count": int(report["pole_count"].sum()),
        "turbine_count": int(report["turbine_count"].sum()),
        "max_total_per_frame": int(report["total_count"].max()),
        "max_cable_per_frame": int(report["cable_count"].max()),
        "max_cable_length": float(report["cable_max_length"].max()),
    }
    return report, summary


def inspect_commands(scene_path: Path, top_rows: Iterable[pd.Series]) -> list[str]:
    commands = []
    for row in top_rows:
        pose_index = int(row["pose_index"])
        commands.append(f"python visualize.py --file {scene_path} --pose-index {pose_index}")
    return commands


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = find_prediction_files(args)
    if not pred_files:
        raise FileNotFoundError("Aucun CSV de prédiction trouvé à analyser.")

    summary_rows = []

    for pred_path in pred_files:
        scene_path = scene_dir / f"{pred_path.stem}.h5"
        if not scene_path.exists():
            raise FileNotFoundError(f"HDF5 source introuvable pour {pred_path}: {scene_path}")

        print(f"\n{'=' * 70}")
        print(f"{pred_path.name}")
        print(f"{'=' * 70}")

        frame_report, summary = build_frame_report(scene_path, pred_path)
        summary_rows.append(summary)

        print(
            f"Detections={summary['total_detections']} | "
            f"Frames actives={summary['frames_with_detections']}/{summary['frames']} | "
            f"Max/frame={summary['max_total_per_frame']} | "
            f"Max cable/frame={summary['max_cable_per_frame']} | "
            f"Max cable length={summary['max_cable_length']:.2f}m"
        )

        top = frame_report.head(args.top_k).copy()
        display_cols = [
            "pose_index",
            "total_count",
            "cable_count",
            "antenna_count",
            "pole_count",
            "turbine_count",
            "cable_max_length",
            "sanity_score",
            "flags",
        ]
        print(top[display_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

        report_path = output_dir / f"{pred_path.stem}_frames.csv"
        top_path = output_dir / f"{pred_path.stem}_top.csv"
        commands_path = output_dir / f"{pred_path.stem}_inspect.txt"

        frame_report.to_csv(report_path, index=False)
        top.to_csv(top_path, index=False)

        cmds = inspect_commands(scene_path, (row for _, row in top.iterrows()))
        commands_path.write_text("\n".join(cmds) + ("\n" if cmds else ""), encoding="utf-8")

        print(f"Rapport frame: {report_path}")
        print(f"Top suspicious: {top_path}")
        print(f"Inspect cmds : {commands_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSummary sauvegarde: {summary_path}")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)))


if __name__ == "__main__":
    main()

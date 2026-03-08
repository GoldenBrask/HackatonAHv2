"""
Lance une évaluation proxy locale sur des scènes labellisées du training set.

Pipeline:
  1. inference.py sur une ou plusieurs scènes training
  2. compute_map.py pour chaque densité
  3. sauvegarde des rapports texte + metrics_summary.csv

Usage:
  python run_proxy_map_batch.py --checkpoint checkpoints/best_model.pth
  python run_proxy_map_batch.py --checkpoint checkpoints/best_model.pth --scene scene_1.h5 --scene scene_2.h5
  python run_proxy_map_batch.py --checkpoint checkpoints/best_model.pth --dry-run
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_SCENES = ("scene_1.h5", "scene_2.h5")
DEFAULT_DENSITIES = (100, 75, 50, 25)
MAP_RE = re.compile(r"mAP @ IoU=.*?:\s*([0-9.]+)")
IOU_RE = re.compile(r"Mean IoU \(Correct Class\)\s*:\s*([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch proxy evaluation with compute_map")
    parser.add_argument(
        "--scene-dir",
        default="airbus_hackathon_trainingdata",
        help="Dossier contenant les scènes training labellisées",
    )
    parser.add_argument(
        "--scene",
        action="append",
        help="Nom de scène ou chemin HDF5 à utiliser (répétable). Défaut: scene_1.h5 + scene_2.h5",
    )
    parser.add_argument(
        "--densities",
        default="100,75,50,25",
        help="Liste de densités séparées par des virgules",
    )
    parser.add_argument(
        "--output-dir",
        default="predictions/proxy_eval",
        help="Dossier racine des CSV proxy",
    )
    parser.add_argument(
        "--report-dir",
        default="proxy_reports",
        help="Dossier de sortie des rapports compute_map",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pth",
        help="Checkpoint à utiliser pour inference.py",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Exécutable Python à utiliser",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Passe --no-tta à inference.py",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Relance même si les CSV / rapports existent déjà",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Utilise les CSV déjà générés",
    )
    parser.add_argument(
        "--skip-map",
        action="store_true",
        help="Ne lance pas compute_map.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les commandes sans les exécuter",
    )
    return parser.parse_args()


def selected_scenes(args: argparse.Namespace) -> list[Path]:
    raw = args.scene or list(DEFAULT_SCENES)
    scene_dir = Path(args.scene_dir)
    result = []
    for item in raw:
        p = Path(item)
        result.append(p if p.exists() else scene_dir / item)
    return result


def selected_densities(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    for value in values:
        if value not in {100, 75, 50, 25}:
            raise ValueError(f"Densité non supportée: {value}")
    return values


def infer_output_path(scene_path: Path, density: int, output_dir: Path) -> Path:
    return output_dir / f"{scene_path.stem}_{density}.csv"


def build_inference_command(
    python_exe: str,
    scene_path: Path,
    out_path: Path,
    checkpoint: Path,
    density: int,
    no_tta: bool,
) -> list[str]:
    cmd = [
        python_exe,
        "inference.py",
        "--input",
        str(scene_path),
        "--output",
        str(out_path),
        "--checkpoint",
        str(checkpoint),
    ]
    if density != 100:
        cmd.extend(["--density", f"{density / 100:.2f}"])
    if no_tta:
        cmd.append("--no-tta")
    return cmd


def build_compute_map_command(
    python_exe: str,
    scene_paths: list[Path],
    pred_paths: list[Path],
) -> list[str]:
    cmd = [python_exe, "compute_map.py"]
    for scene_path, pred_path in zip(scene_paths, pred_paths):
        cmd.extend(["--scene", str(scene_path), "--pred", str(pred_path)])
    return cmd


def parse_metrics(output: str) -> tuple[float | None, float | None]:
    maps = MAP_RE.findall(output)
    ious = IOU_RE.findall(output)
    map_value = float(maps[-1]) if maps else None
    iou_value = float(ious[-1]) if ious else None
    return map_value, iou_value


def run_command(cmd: list[str], dry_run: bool) -> subprocess.CompletedProcess[str] | None:
    print("Commande:", " ".join(cmd))
    if dry_run:
        return None
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def write_summary(rows: list[dict[str, object]], report_dir: Path) -> None:
    summary_path = report_dir / "metrics_summary.csv"
    fieldnames = [
        "density",
        "scenes",
        "mAP",
        "mean_iou_correct_class",
        "report_file",
        "status",
        "elapsed_sec",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary sauvegardé: {summary_path}")


def main() -> None:
    args = parse_args()
    scene_paths = selected_scenes(args)
    densities = selected_densities(args.densities)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)

    missing_scenes = [p for p in scene_paths if not p.exists()]
    if missing_scenes and not args.dry_run:
        missing_str = "\n".join(f"  - {p}" for p in missing_scenes)
        raise FileNotFoundError(f"Scènes introuvables:\n{missing_str}")
    if missing_scenes and args.dry_run:
        missing_str = "\n".join(f"  - {p}" for p in missing_scenes)
        print(f"WARNING: scènes introuvables en dry-run:\n{missing_str}")

    if not checkpoint.exists() and not args.dry_run:
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint}")
    if not checkpoint.exists() and args.dry_run:
        print(f"WARNING: checkpoint introuvable en dry-run: {checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_inference:
        for scene_path in scene_paths:
            for density in densities:
                out_path = infer_output_path(scene_path, density, output_dir)
                print(f"\n[{scene_path.name}] densité {density}% -> {out_path}")
                if out_path.exists() and not args.overwrite:
                    print("CSV déjà présent, skip (utilise --overwrite pour relancer).")
                    continue
                cmd = build_inference_command(
                    python_exe=args.python,
                    scene_path=scene_path,
                    out_path=out_path,
                    checkpoint=checkpoint,
                    density=density,
                    no_tta=args.no_tta,
                )
                result = run_command(cmd, args.dry_run)
                if result is not None and result.stdout:
                    print(result.stdout)

    summary_rows = []

    if not args.skip_map:
        for density in densities:
            pred_paths = [infer_output_path(scene_path, density, output_dir) for scene_path in scene_paths]
            missing_preds = [p for p in pred_paths if not p.exists()]
            if missing_preds and not args.dry_run:
                missing_str = "\n".join(f"  - {p}" for p in missing_preds)
                raise FileNotFoundError(f"CSV proxy introuvables pour densité {density}%:\n{missing_str}")

            report_path = report_dir / f"compute_map_{density}.txt"
            print(f"\n[compute_map] densité {density}% -> {report_path}")

            if report_path.exists() and not args.overwrite and not args.dry_run:
                output = report_path.read_text(encoding="utf-8")
                elapsed = 0.0
                status = "skipped"
                print("Rapport déjà présent, skip (utilise --overwrite pour relancer).")
            else:
                cmd = build_compute_map_command(args.python, scene_paths, pred_paths)
                t0 = time.time()
                result = run_command(cmd, args.dry_run)
                elapsed = time.time() - t0 if not args.dry_run else 0.0
                status = "done" if not args.dry_run else "dry-run"
                output = "" if result is None else result.stdout
                if not args.dry_run:
                    report_path.write_text(output, encoding="utf-8")
                    print(output)

            map_value, iou_value = parse_metrics(output)
            summary_rows.append({
                "density": density,
                "scenes": ",".join(p.name for p in scene_paths),
                "mAP": "" if map_value is None else f"{map_value:.4f}",
                "mean_iou_correct_class": "" if iou_value is None else f"{iou_value:.4f}",
                "report_file": report_path.name,
                "status": status,
                "elapsed_sec": f"{elapsed:.2f}",
            })

    if summary_rows:
        write_summary(summary_rows, report_dir)
        print("\nRésumé proxy:")
        for row in summary_rows:
            print(
                f"  {row['density']}% | mAP={row['mAP'] or 'n/a'} | "
                f"MeanIoU={row['mean_iou_correct_class'] or 'n/a'} | "
                f"status={row['status']}"
            )


if __name__ == "__main__":
    main()

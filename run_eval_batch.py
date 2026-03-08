"""
Lance l'inference probabilistic sur tout l'eval set Airbus en une commande.

Fonctionnalites:
  - traite sceneA/sceneB et densites 100/75/50/25
  - ecrit les CSV dans un dossier unique
  - genere un summary.csv avec le nombre de detections par classe

Usage:
  python run_eval_batch.py --checkpoint checkpoints/best_model.pth
  python run_eval_batch.py --checkpoint checkpoints/best_model.pth --dry-run
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


SCENES = ("A", "B")
DENSITIES = (100, 75, 50, 25)
CLASS_LABELS = ("Antenna", "Cable", "Electric Pole", "Wind Turbine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference Airbus eval set")
    parser.add_argument(
        "--eval-dir",
        default="airbus_hackathon_evalset",
        help="Dossier contenant eval_sceneA/B_100/75/50/25.h5",
    )
    parser.add_argument(
        "--output-dir",
        default="predictions/eval",
        help="Dossier racine des CSV de sortie",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pth",
        help="Checkpoint a utiliser pour inference.py",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Executable Python a utiliser pour lancer inference.py",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Passe --no-tta a inference.py",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Relance meme si le CSV de sortie existe deja",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les commandes sans les executer",
    )
    return parser.parse_args()
def expected_eval_files(eval_dir: Path) -> list[Path]:
    return [
        eval_dir / f"eval_scene{scene}_{density}.h5"
        for scene in SCENES
        for density in DENSITIES
    ]


def build_command(
    python_exe: str,
    input_path: Path,
    output_path: Path,
    checkpoint: Path,
    no_tta: bool,
) -> list[str]:
    cmd = [
        python_exe,
        "inference.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--checkpoint",
        str(checkpoint),
    ]
    if no_tta:
        cmd.append("--no-tta")
    return cmd


def summarize_predictions(csv_path: Path) -> dict[str, int]:
    counts = Counter()
    total = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            counts[row["class_label"]] += 1

    return {
        "num_detections": total,
        "antenna": counts["Antenna"],
        "cable": counts["Cable"],
        "electric_pole": counts["Electric Pole"],
        "wind_turbine": counts["Wind Turbine"],
    }


def write_summary(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    summary_path = output_dir / "summary.csv"
    fieldnames = [
        "mode",
        "file",
        "scene",
        "density",
        "num_detections",
        "antenna",
        "cable",
        "electric_pole",
        "wind_turbine",
        "elapsed_sec",
        "status",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary sauvegarde: {summary_path}")


def print_summary(summary_rows: list[dict[str, object]]) -> None:
    print("\nResume des sorties")
    print(
        f"{'File':<20} {'Det':>6} {'Ant':>6} {'Cab':>6} "
        f"{'Pole':>6} {'Turb':>6} {'Status':>10}"
    )
    print("-" * 67)
    for row in summary_rows:
        print(
            f"{row['file']:<20} {row['num_detections']:>6} "
            f"{row['antenna']:>6} {row['cable']:>6} {row['electric_pole']:>6} "
            f"{row['wind_turbine']:>6} {row['status']:>10}"
        )


def main() -> None:
    args = parse_args()

    eval_dir = Path(args.eval_dir)
    output_root = Path(args.output_dir)
    checkpoint = Path(args.checkpoint)

    missing = [p for p in expected_eval_files(eval_dir) if not p.exists()]
    if missing:
        missing_str = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(f"Fichiers eval manquants:\n{missing_str}")

    if not checkpoint.exists() and not args.dry_run:
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint}")
    if not checkpoint.exists() and args.dry_run:
        print(f"WARNING: checkpoint introuvable en dry-run: {checkpoint}")

    summary_rows: list[dict[str, object]] = []

    output_root.mkdir(parents=True, exist_ok=True)

    for scene in SCENES:
        for density in DENSITIES:
            input_path = eval_dir / f"eval_scene{scene}_{density}.h5"
            output_path = output_root / f"eval_scene{scene}_{density}.csv"
            cmd = build_command(
                python_exe=args.python,
                input_path=input_path,
                output_path=output_path,
                checkpoint=checkpoint,
                no_tta=args.no_tta,
            )

            status = "skipped"
            elapsed = 0.0

            print(f"\n{input_path.name} -> {output_path}")
            if output_path.exists() and not args.overwrite:
                print("CSV deja present, skip (utilise --overwrite pour relancer).")
            else:
                print("Commande:", " ".join(cmd))
                if not args.dry_run:
                    t0 = time.time()
                    subprocess.run(cmd, check=True)
                    elapsed = time.time() - t0
                    status = "done"
                else:
                    status = "dry-run"

            if output_path.exists() and not args.dry_run:
                stats = summarize_predictions(output_path)
            else:
                stats = {
                    "num_detections": 0,
                    "antenna": 0,
                    "cable": 0,
                    "electric_pole": 0,
                    "wind_turbine": 0,
                }

            summary_rows.append(
                {
                    "mode": "probabilistic",
                    "file": input_path.name,
                    "scene": scene,
                    "density": density,
                    "elapsed_sec": round(elapsed, 2),
                    "status": status,
                    **stats,
                }
            )

    write_summary(summary_rows, output_root)
    print_summary(summary_rows)


if __name__ == "__main__":
    main()

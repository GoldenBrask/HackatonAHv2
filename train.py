"""
Entraînement RandLA-Net pour segmentation LiDAR 3D.

Features:
  - Mixed precision (AMP) pour vitesse x2 sur L4
  - Cosine annealing LR avec warmup
  - Focal + Lovász-Softmax loss
  - Early stopping sur val mIoU moyen 4 densités
  - Checkpointing sur mIoU moyen (25%+50%+75%+100%) / 4
  - Logging détaillé par epoch
"""
import os
import sys
import json
import time
from copy import copy

# Evite l'oversubscription CPU dans les workers DataLoader quand NumPy/SciPy
# lancent eux-memes des threads BLAS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from config import Config
from model import RandLANet, count_parameters
from dataset import LidarDataset, build_dataloaders
from losses import CombinedLoss


def compute_iou(pred, target, num_classes):
    """Calcule l'IoU par classe."""
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return ious


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg):
    """Un epoch d'entraînement."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in loader:
        xyz = batch["xyz"].to(device)
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        # Préparer les indices (déjà dans le batch via DataLoader)
        batch_data = {}
        for key in batch:
            if key.startswith(("neigh_", "sub_", "up_")):
                batch_data[key] = batch[key].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=cfg.use_amp):
            logits = model(xyz, features, batch_data)  # (B, N, C)
            B, N, C = logits.shape
            logits_flat = logits.reshape(B * N, C)
            labels_flat = labels.reshape(B * N)
            loss = criterion(logits_flat, labels_flat)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        # Predictions pour métriques
        with torch.no_grad():
            preds = logits.argmax(dim=-1).cpu().numpy().flatten()
            labs = labels.cpu().numpy().flatten()
            all_preds.append(preds)
            all_labels.append(labs)

    avg_loss = total_loss / max(num_batches, 1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    ious = compute_iou(all_preds, all_labels, cfg.num_classes)
    miou = np.nanmean(ious)

    return avg_loss, miou, ious


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    """Validation à la densité courante de cfg.val_density."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in loader:
        xyz = batch["xyz"].to(device)
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)

        batch_data = {}
        for key in batch:
            if key.startswith(("neigh_", "sub_", "up_")):
                batch_data[key] = batch[key].to(device)

        with autocast("cuda", enabled=cfg.use_amp):
            logits = model(xyz, features, batch_data)
            B, N, C = logits.shape
            logits_flat = logits.reshape(B * N, C)
            labels_flat = labels.reshape(B * N)
            loss = criterion(logits_flat, labels_flat)

        total_loss += loss.item()
        num_batches += 1

        preds = logits.argmax(dim=-1).cpu().numpy().flatten()
        labs = labels.cpu().numpy().flatten()
        all_preds.append(preds)
        all_labels.append(labs)

    avg_loss = total_loss / max(num_batches, 1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    ious = compute_iou(all_preds, all_labels, cfg.num_classes)
    miou = np.nanmean(ious)

    return avg_loss, miou, ious


def validate_multi_density(model, val_loader, criterion, device, cfg):
    """Validation sur 4 densités fixes (25%, 50%, 75%, 100%).

    Crée un DataLoader single-process par densité pour contourner le problème
    de persistent_workers : les workers multiprocess ont leur propre copie de cfg
    et ne voient pas le patch cfg.val_density fait dans le process principal.
    num_workers=0 évite ce bug (validation = non-bottleneck, pas critique pour la vitesse).
    """
    from torch.utils.data import DataLoader as _DL

    densities = [0.25, 0.50, 0.75, 1.00]
    original_density = cfg.val_density
    results = {}

    for d in densities:
        # Patcher sur le dataset directement (accessible dans le process principal)
        val_loader.dataset.cfg.val_density = d
        # DataLoader single-process → lit cfg.val_density du process principal
        single_loader = _DL(
            val_loader.dataset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=0,   # pas de workers → pas de copie isolée de cfg
            pin_memory=False,
            drop_last=False,
        )
        _, miou_d, _ = validate(model, single_loader, criterion, device, cfg)
        results[d] = miou_d

    # Restaurer
    val_loader.dataset.cfg.val_density = original_density
    cfg.val_density = original_density

    mean_miou = float(np.mean(list(results.values())))
    return mean_miou, results


def validate_multi_density_fast(model, val_file_list, criterion, device, cfg):
    """Validation 4 densites sans mutation de cfg partage.

    Chaque densite recoit son propre Dataset avec une copie de config fixe.
    On peut ainsi garder des workers multiprocess sans bug de copie de cfg,
    tout en evitant la validation 50% en double dans la boucle principale.
    """
    densities = [0.25, 0.50, 0.75, 1.00]
    results = {}
    details = {}

    for d in densities:
        density_cfg = copy(cfg)
        density_cfg.val_density = d
        density_dataset = LidarDataset(val_file_list, training=False, cfg=density_cfg)

        loader_kwargs = {
            "batch_size": cfg.batch_size,
            "shuffle": False,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
            "drop_last": False,
        }
        if cfg.num_workers > 0:
            loader_kwargs["persistent_workers"] = False
            loader_kwargs["prefetch_factor"] = 3

        density_loader = DataLoader(density_dataset, **loader_kwargs)
        loss_d, miou_d, ious_d = validate(
            model, density_loader, criterion, device, cfg
        )
        results[d] = miou_d
        details[d] = {
            "loss": loss_d,
            "miou": miou_d,
            "ious": ious_d,
        }

    mean_miou = float(np.mean(list(results.values())))
    return mean_miou, results, details


def get_cosine_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    """Cosine annealing avec warmup linéaire."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))


def main():
    cfg = Config()
    cfg.ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Charger les class weights ──
    if getattr(cfg, 'class_weights_override', None) is not None:
        class_weights = cfg.class_weights_override
        print(f"Class weights (override): {[f'{w:.2f}' for w in class_weights]}")
    else:
        stats_path = os.path.join(cfg.processed_dir, "dataset_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            class_weights = stats["class_weights"]
            print(f"Class weights from data: {[f'{w:.2f}' for w in class_weights]}")
        else:
            class_weights = cfg.default_class_weights
            print(f"Using default class weights")

    # ── Data ──
    print("\nChargement des données...")
    train_loader, val_loader = build_dataloaders(cfg)
    val_file_list = list(val_loader.dataset.file_list)

    # ── Modèle ──
    model = RandLANet(
        d_in=cfg.d_in,
        num_classes=cfg.num_classes,
        d_encoder=cfg.d_encoder,
        num_layers=cfg.num_layers,
    ).to(device)
    n_params = count_parameters(model)
    print(f"\nModèle: RandLA-Net")
    print(f"Paramètres: {n_params:,} ({n_params/1e6:.2f}M)")

    # torch.compile : +10-20% throughput sans changement de qualité (PyTorch 2.x)
    # Première epoch plus lente (compilation JIT) — normal
    try:
        model = torch.compile(model)
        print("torch.compile activé")
    except Exception as e:
        print(f"torch.compile non disponible ({e}) — mode eager")

    # ── Loss ──
    criterion = CombinedLoss(
        focal_gamma=cfg.focal_gamma,
        class_weights=class_weights,
        focal_weight=cfg.focal_weight,
        lovasz_weight=cfg.lovasz_weight,
    )

    # ── Optimizer + Scheduler ──
    # Run 8: Adam → AdamW (decoupled weight decay correct, Loshchilov & Hutter 2019)
    # Avec Adam + weight_decay = L2 reg (biaise les moments adaptatifs)
    # Avec AdamW + weight_decay = decoupled (mathématiquement correct)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = GradScaler("cuda", enabled=cfg.use_amp)

    # ── Warm start depuis le meilleur checkpoint ──
    start_epoch = 0
    best_miou = 0.0
    patience_counter = 0

    warm_start_path = os.path.join(cfg.checkpoint_dir, "best_model.pth")
    if getattr(cfg, 'warm_start', False) and os.path.exists(warm_start_path):
        ckpt = torch.load(warm_start_path, map_location=device, weights_only=False)
        # Vérifier compatibilité architecture
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            # NE PAS hériter du best_miou du checkpoint : si les features ont changé
            # (distribution différente → val_miou démarre plus bas), on ne veut pas
            # bloquer le checkpointing sur un seuil inatteignable.
            # Run 9 repart de 0 pour le critère de sauvegarde.
            best_miou = 0.0
            ckpt_miou = ckpt.get("val_miou", 0.0)
            print(f"\n★ Warm start depuis epoch {ckpt.get('epoch', '?')} "
                  f"(val mIoU checkpoint={ckpt_miou:.4f})")
            print(f"  Reprise à l'epoch {start_epoch + 1} | best_miou reset à 0.0")
            # Backup du checkpoint Run 8 (sécurité avant Run 9)
            import shutil
            backup_path = os.path.join(cfg.checkpoint_dir, "best_model_run8.pth")
            if not os.path.exists(backup_path):
                shutil.copy2(warm_start_path, backup_path)
                print(f"  Backup sauvegardé: {backup_path}")
        except RuntimeError as e:
            print(f"\n⚠ Warm start échoué (architecture incompatible): {e}")
            print("  Démarrage depuis zéro.")
            start_epoch = 0
            best_miou = 0.0
    else:
        if getattr(cfg, 'warm_start', False):
            print(f"\n⚠ Warm start activé mais aucun checkpoint trouvé. Démarrage depuis zéro.")

    # ── Training loop ──
    history = []

    class_names = ["Antenna", "Cable", "Pole", "Turbine", "Background"]

    val_multi_freq = getattr(cfg, 'val_multi_density_freq', 5)
    print(f"\n{'='*70}")
    print(f"Début de l'entraînement — {cfg.epochs} epochs max")
    print(f"Checkpoint sur mIoU moyen 4 densités (25/50/75/100%) "
          f"— validation complète toutes les {val_multi_freq} epochs")
    print(f"{'='*70}")

    # Seuils garde-fous Run 11 (cold start — seuils réalistes)
    # Run 9 utilisait 0.90/0.70 calibrés pour un warm start → trop restrictifs pour cold start
    # Cold start epoch 30 : val_miou attendu 0.50-0.65, epoch 50 : IoU classe 0.40-0.60
    MIOU_FLOOR = 0.65       # Warning si val_miou@50% < 0.65 après epoch 30 (cold start réaliste)
    CLASS_IOU_FLOOR = 0.55  # Warning si classe obstacle < 0.55 après epoch 50 (cold start réaliste)

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        # Curriculum: informer le dataset de l'epoch courante (density_min adaptatif)
        train_loader.dataset.current_epoch = epoch

        # LR scheduling (continue la courbe cosine depuis start_epoch)
        lr = get_cosine_lr(epoch, cfg.warmup_epochs, cfg.epochs, cfg.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Train
        train_loss, train_miou, train_ious = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg
        )

        # Validate — val simple à cfg.val_density (rapide, chaque epoch)
        run_multi = ((epoch + 1) % val_multi_freq == 0) or (epoch == start_epoch)
        if not run_multi:
            val_loss, val_miou, val_ious = validate(
                model, val_loader, criterion, device, cfg
            )
        else:
            val_loss = val_miou = val_ious = None

        # Validate multi-densité (toutes les val_multi_freq epochs)
        # C'est ce critère qui pilote le checkpointing.
        run_multi = ((epoch + 1) % val_multi_freq == 0) or (epoch == start_epoch)
        if run_multi:
            mean_miou_4d, miou_per_density, density_details = validate_multi_density_fast(
                model, val_file_list, criterion, device, cfg
            )
            val_detail = density_details.get(cfg.val_density)
            if val_detail is not None:
                val_loss = val_detail["loss"]
                val_miou = val_detail["miou"]
                val_ious = val_detail["ious"]
        else:
            mean_miou_4d = None
            miou_per_density = None

        dt = time.time() - t0

        # ── Logging ──
        print(f"\nEpoch {epoch+1:3d}/{cfg.epochs} ({dt:.0f}s) | "
              f"LR={lr:.6f}")
        print(f"  Train — Loss: {train_loss:.4f} | mIoU: {train_miou:.4f}")
        print(f"  Val@{cfg.val_density:.0%}  — Loss: {val_loss:.4f} | mIoU: {val_miou:.4f}")
        if run_multi:
            print(f"  Val 4-densités : "
                  f"25%={miou_per_density[0.25]:.4f}  "
                  f"50%={miou_per_density[0.50]:.4f}  "
                  f"75%={miou_per_density[0.75]:.4f}  "
                  f"100%={miou_per_density[1.00]:.4f}  "
                  f"| moyenne={mean_miou_4d:.4f}")
        print(f"  Val IoU par classe (@{cfg.val_density:.0%}):")
        for i, name in enumerate(class_names):
            iou_str = f"{val_ious[i]:.4f}" if not np.isnan(val_ious[i]) else "  N/A "
            print(f"    {name:12s}: {iou_str}")

        # ── Garde-fous régression (seuils cold start) ──
        if epoch >= 30 and val_miou < MIOU_FLOOR:
            print(f"  WARNING: val_miou@{cfg.val_density:.0%}={val_miou:.4f} < seuil cold start {MIOU_FLOOR}")
        if epoch >= 50:
            for i, (name, iou) in enumerate(zip(class_names[:4], val_ious[:4])):
                if not np.isnan(iou) and iou < CLASS_IOU_FLOOR:
                    print(f"  CRITICAL: {name} IoU={iou:.4f} < {CLASS_IOU_FLOOR}")

        # ── Checkpointing sur mIoU moyen 4 densités ──
        # Run 13 : critère aligné sur l'évaluation hackathon (4 densités séparées).
        # On ne checkpointe que lors des epochs de validation complète.
        # Les epochs intermédiaires ne modifient pas best_miou → patience_counter monte normalement.
        if run_multi:
            checkpoint_score = mean_miou_4d
            is_best = checkpoint_score > best_miou + cfg.min_delta
            if is_best:
                best_miou = checkpoint_score
                patience_counter = 0
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_miou": mean_miou_4d,          # mIoU moyen 4 densités
                    "val_miou_per_density": miou_per_density,
                    "val_ious": val_ious,               # IoU par classe @cfg.val_density
                    "config": {
                        "d_in": cfg.d_in,
                        "num_classes": cfg.num_classes,
                        "d_encoder": cfg.d_encoder,
                        "num_layers": cfg.num_layers,
                        "num_points": cfg.num_points,
                        "k_neighbors": cfg.k_neighbors,
                    },
                }
                torch.save(ckpt, os.path.join(cfg.checkpoint_dir, "best_model.pth"))
                print(f"  ★ Nouveau meilleur modèle sauvegardé "
                      f"(mIoU_avg4d={mean_miou_4d:.4f})")
            else:
                patience_counter += 1
        else:
            # Pas de validation multi-densité cet epoch : patience monte
            patience_counter += 1

        # Sauvegarder dernier modèle aussi
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_miou": val_miou,
        }, os.path.join(cfg.checkpoint_dir, "last_model.pth"))

        # History
        history_entry = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            "train_miou": train_miou,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "val_ious": val_ious,
        }
        if run_multi:
            history_entry["val_miou_avg4d"] = mean_miou_4d
            history_entry["val_miou_25"] = miou_per_density[0.25]
            history_entry["val_miou_50"] = miou_per_density[0.50]
            history_entry["val_miou_75"] = miou_per_density[0.75]
            history_entry["val_miou_100"] = miou_per_density[1.00]
        history.append(history_entry)

        # ── Early stopping ──
        if patience_counter >= cfg.patience:
            print(f"\nEarly stopping à l'epoch {epoch+1} "
                  f"(pas d'amélioration depuis {cfg.patience} epochs)")
            break

    # ── Résumé final ──
    print(f"\n{'='*70}")
    print(f"ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*70}")
    print(f"Meilleur mIoU moyen 4 densités: {best_miou:.4f}")
    print(f"Modèle sauvegardé: {cfg.checkpoint_dir}/best_model.pth")
    print(f"Paramètres: {n_params:,}")

    # Sauvegarder l'historique
    with open(os.path.join(cfg.log_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)


if __name__ == "__main__":
    main()

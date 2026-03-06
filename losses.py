"""
Focal Loss + Lovász-Softmax pour segmentation LiDAR.
- Focal Loss : gère le déséquilibre de classes (99% background vs 1% câbles)
- Lovász-Softmax : optimise directement le IoU (métrique d'évaluation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) avec class weights.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    Focalise l'apprentissage sur les exemples difficiles.
    """

    def __init__(self, gamma=2.0, alpha=None, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.FloatTensor(alpha)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        logits: (B*N, C) — prédictions brutes
        targets: (B*N,) — labels
        """
        # Masquer les points ignorés
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            logits = logits[mask]
            targets = targets[mask]

        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p = torch.exp(-ce_loss)  # probabilité de la vraie classe
        focal_weight = (1 - p) ** self.gamma

        loss = focal_weight * ce_loss

        # Appliquer les poids par classe
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            loss = alpha_t * loss

        return loss.mean()


# ═══════════════════════════════════════════════
# Lovász-Softmax (Berman et al., 2018)
# Optimise directement le Jaccard Index (IoU)
# ═══════════════════════════════════════════════

def lovasz_grad(gt_sorted):
    """Gradient du Lovász extension w.r.t. erreurs triées."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes="present"):
    """
    Lovász-Softmax loss multi-classe (aplati).
    probas: (P, C) — probabilités softmax
    labels: (P,) — labels ground truth
    classes: 'present' pour ignorer les classes absentes du batch
    """
    if probas.numel() == 0:
        return probas * 0.0

    C = probas.size(1)
    losses = []

    for c in range(C):
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            fg_class = 1.0 - probas[:, 0]
        else:
            fg_class = probas[:, c]
        errors = (fg - fg_class).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))

    if not losses:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)
    return torch.stack(losses).mean()


class LovaszSoftmax(nn.Module):
    """Wrapper module pour Lovász-Softmax."""

    def __init__(self, classes="present", ignore_index=-1):
        super().__init__()
        self.classes = classes
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: (B*N, C)
        targets: (B*N,)
        """
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            logits = logits[mask]
            targets = targets[mask]

        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probas = F.softmax(logits, dim=1)
        return lovasz_softmax_flat(probas, targets, classes=self.classes)


class CombinedLoss(nn.Module):
    """
    Loss combinée : Focal + Lovász-Softmax.
    Les deux se complètent :
    - Focal gère le déséquilibre via les poids
    - Lovász optimise directement l'IoU
    """

    def __init__(self, focal_gamma=2.0, class_weights=None,
                 focal_weight=1.0, lovasz_weight=1.0, ignore_index=-1):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=class_weights,
                               ignore_index=ignore_index)
        self.lovasz = LovaszSoftmax(classes="present",
                                     ignore_index=ignore_index)
        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, targets):
        fl = self.focal(logits, targets)
        ls = self.lovasz(logits, targets)
        return self.focal_weight * fl + self.lovasz_weight * ls

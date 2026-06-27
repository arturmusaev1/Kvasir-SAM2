from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return alpha_t * (1.0 - pt).pow(self.gamma) * bce


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.float().flatten(1)
        inter = (probs * targets).sum(1)
        denom = probs.sum(1) + targets.sum(1)
        return 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)


class IoULoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.float().flatten(1)
        inter = (probs * targets).sum(1)
        union = probs.sum(1) + targets.sum(1) - inter
        return 1.0 - ((inter + self.eps) / (union + self.eps)).mean()


class BCEIoULoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.iou = IoULoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, scores: Optional[torch.Tensor] = None):
        if logits.ndim == 4 and logits.shape[1] > 1:
            logits = logits[:, 0]
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        bce = self.bce(logits, targets.float())
        iou = self.iou(logits, targets)
        loss = bce + iou
        return loss, {"loss": float(loss.detach().cpu()), "bce": float(bce.detach().cpu()), "iou": float(iou.detach().cpu())}


class SAMOriginalLoss(nn.Module):
    def __init__(self, focal_weight: float = 20.0, dice_weight: float = 1.0, score_weight: float = 1.0):
        super().__init__()
        self.focal = SigmoidFocalLoss()
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.score_weight = score_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, scores: Optional[torch.Tensor] = None):
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if logits.ndim == 4 and logits.shape[1] > 1:
            return self._multi(logits, targets, scores)
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]
        return self._single(logits, targets)

    def _single(self, logits: torch.Tensor, targets: torch.Tensor):
        focal = self.focal(logits, targets).flatten(1).mean(1)
        dice = self.dice(logits, targets)
        loss = (self.focal_weight * focal + self.dice_weight * dice).mean()
        return loss, {"loss": float(loss.detach().cpu()), "focal": float(focal.mean().detach().cpu()), "dice": float(dice.mean().detach().cpu())}

    def _multi(self, logits: torch.Tensor, targets: torch.Tensor, scores: Optional[torch.Tensor]):
        b, m, h, w = logits.shape
        targets_exp = targets[:, None].expand(b, m, h, w)
        logits_flat = logits.reshape(b * m, h, w)
        targets_flat = targets_exp.reshape(b * m, h, w)
        focal = self.focal(logits_flat, targets_flat).flatten(1).mean(1)
        dice = self.dice(logits_flat, targets_flat)
        per_mask = (self.focal_weight * focal + self.dice_weight * dice).view(b, m)
        best_loss, best_idx = per_mask.min(1)
        loss = best_loss.mean()
        if scores is not None:
            target_iou = self._soft_iou(logits, targets)
            selected_scores = scores.gather(1, best_idx[:, None]).squeeze(1)
            selected_iou = target_iou.gather(1, best_idx[:, None]).squeeze(1)
            score_loss = F.mse_loss(selected_scores, selected_iou)
            loss = loss + self.score_weight * score_loss
            return loss, {"loss": float(loss.detach().cpu()), "score": float(score_loss.detach().cpu())}
        return loss, {"loss": float(loss.detach().cpu())}

    @torch.no_grad()
    def _soft_iou(self, logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
        probs = torch.sigmoid(logits)
        targets = targets[:, None].expand_as(probs)
        inter = (probs * targets).sum((2, 3))
        union = probs.sum((2, 3)) + targets.sum((2, 3)) - inter
        return (inter + eps) / (union + eps)

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class DistillLossOutput:
    loss: torch.Tensor
    top1: torch.Tensor
    valid_ratio: torch.Tensor


def clip_alignment_loss(image_emb: torch.Tensor, text_emb: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric CLIP loss over in-batch pairs."""
    logits_i2t = (image_emb @ text_emb.t()) / tau
    logits_t2i = logits_i2t.t()
    labels = torch.arange(logits_i2t.size(0), device=logits_i2t.device)

    loss_i2t = F.cross_entropy(logits_i2t, labels)
    loss_t2i = F.cross_entropy(logits_t2i, labels)
    return 0.5 * (loss_i2t + loss_t2i)


def clip_hard_negative_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    tau: float,
    margin: float = 0.0,
) -> torch.Tensor:
    """In-batch hard negative hinge loss for CLIP embeddings.

    For each positive pair, push the hardest negative below the positive score.
    """
    logits = (image_emb @ text_emb.t()) / tau
    bsz = int(logits.size(0))
    pos = torch.diag(logits)

    if bsz <= 1:
        return torch.tensor(0.0, device=logits.device)

    neg_mask = ~torch.eye(bsz, device=logits.device, dtype=torch.bool)
    neg_i2t = logits.masked_fill(~neg_mask, float("-inf")).max(dim=1).values
    neg_t2i = logits.t().masked_fill(~neg_mask, float("-inf")).max(dim=1).values

    loss_i2t = F.relu(margin + neg_i2t - pos)
    loss_t2i = F.relu(margin + neg_t2i - pos)

    return 0.5 * (loss_i2t.mean() + loss_t2i.mean())


def distill_logits(query: torch.Tensor, bank: torch.Tensor, tau: float) -> torch.Tensor:
    return (query @ bank.t()) / tau


def distill_cross_entropy_loss(
    query: torch.Tensor,
    bank: torch.Tensor,
    labels: torch.Tensor,
    tau: float,
    ignore_index: int = -1,
    class_weights: torch.Tensor | None = None,
) -> DistillLossOutput:
    """Cross-entropy distillation with optional ignore-index support."""
    logits = distill_logits(query=query, bank=bank, tau=tau)
    labels = labels.long()

    valid_mask = labels != ignore_index
    valid_ratio = valid_mask.float().mean()

    if not valid_mask.any():
        zero = torch.tensor(0.0, device=query.device)
        return DistillLossOutput(loss=zero, top1=zero, valid_ratio=valid_ratio)

    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]

    weight = None
    if class_weights is not None:
        weight = class_weights.to(valid_logits.device, dtype=valid_logits.dtype)

    loss = F.cross_entropy(valid_logits, valid_labels, weight=weight)
    top1 = (valid_logits.argmax(dim=1) == valid_labels).float().mean()

    return DistillLossOutput(loss=loss, top1=top1, valid_ratio=valid_ratio)

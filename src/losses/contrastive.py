from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    """Simple and robust InfoNCE-style loss for Stage 1 MVP.

    Expected shapes:
      - anchor: [B, D]
      - positives: [B, N_pos, D]
      - negatives: [B, N_neg, D]

    The loss treats all positives for each anchor as positive set,
    and all negatives for each anchor as negative set.
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-12) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(
        self,
        anchor: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        self._validate_shapes(anchor, positives, negatives)

        # Cosine-style similarity under L2-normalized embeddings.
        # (If upstream already normalizes, this is still safe.)
        anchor = self._l2_normalize(anchor)  # [B, D]
        positives = self._l2_normalize(positives)  # [B, N_pos, D]
        negatives = self._l2_normalize(negatives)  # [B, N_neg, D]

        # Positive similarities: [B, N_pos]
        pos_sim = torch.einsum("bd,bnd->bn", anchor, positives)
        # Negative similarities: [B, N_neg]
        neg_sim = torch.einsum("bd,bnd->bn", anchor, negatives)

        pos_logits = pos_sim / self.temperature
        neg_logits = neg_sim / self.temperature

        # Set InfoNCE:
        # L_i = -log( sum(exp(pos_logits_i)) / (sum(exp(pos_logits_i)) + sum(exp(neg_logits_i))) )
        pos_logsumexp = torch.logsumexp(pos_logits, dim=1)  # [B]
        neg_logsumexp = torch.logsumexp(neg_logits, dim=1)  # [B]

        # log(sum_pos + sum_neg) in stable form
        all_logsumexp = torch.logaddexp(pos_logsumexp, neg_logsumexp)  # [B]

        loss = -(pos_logsumexp - all_logsumexp).mean()

        if not return_debug:
            return loss

        debug = {
            "mean_positive_similarity": float(pos_sim.mean().detach().item()),
            "mean_negative_similarity": float(neg_sim.mean().detach().item()),
        }
        return loss, debug

    def _validate_shapes(
        self,
        anchor: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> None:
        if anchor.dim() != 2:
            raise ValueError(f"anchor must be 2D [B, D], got shape {tuple(anchor.shape)}")
        if positives.dim() != 3:
            raise ValueError(
                f"positives must be 3D [B, N_pos, D], got shape {tuple(positives.shape)}"
            )
        if negatives.dim() != 3:
            raise ValueError(
                f"negatives must be 3D [B, N_neg, D], got shape {tuple(negatives.shape)}"
            )

        b, d = anchor.shape
        b_pos, n_pos, d_pos = positives.shape
        b_neg, n_neg, d_neg = negatives.shape

        if b_pos != b or b_neg != b:
            raise ValueError(
                "Batch size mismatch: "
                f"anchor={b}, positives={b_pos}, negatives={b_neg}"
            )
        if d_pos != d or d_neg != d:
            raise ValueError(
                "Embedding dim mismatch: "
                f"anchor={d}, positives={d_pos}, negatives={d_neg}"
            )
        if n_pos <= 0:
            raise ValueError("positives must have N_pos > 0")
        if n_neg <= 0:
            raise ValueError("negatives must have N_neg > 0")

    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
            return x / denom
        if x.dim() == 3:
            denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
            return x / denom
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")


class Stage1Loss(nn.Module):
    """Small wrapper for Stage 1 MVP contrastive loss."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.criterion = InfoNCELoss(temperature=temperature)

    def forward(
        self,
        anchor: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        return self.criterion(
            anchor=anchor,
            positives=positives,
            negatives=negatives,
            return_debug=return_debug,
        )

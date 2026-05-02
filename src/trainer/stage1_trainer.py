from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.losses.contrastive import Stage1Loss
from src.models.teacher import Stage1TeacherModel
from src.ontology.schema import ConceptType, Stage1Batch


@dataclass(slots=True)
class Stage1TrainerConfig:
    """Minimal config for Stage 1 trainer."""

    device: str = "cuda"
    grad_clip_norm: float = 1.0
    seed: int = 42


class Stage1Trainer:
    """Simple and reproducible Stage 1 trainer (MVP).

    Design:
      - Train one head type at a time using homogeneous batches.
      - Supports train_epoch and validate.
      - Logs:
          * total loss
          * mean positive similarity
          * mean negative similarity
    """

    def __init__(
        self,
        model: Stage1TeacherModel,
        criterion: Stage1Loss,
        optimizer: torch.optim.Optimizer,
        config: Stage1TrainerConfig | None = None,
    ) -> None:
        self.config = config or Stage1TrainerConfig()
        self.device = torch.device(self.config.device)

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer

        self._set_seed(self.config.seed)

    def train_epoch(
        self,
        dataloader: DataLoader,
        expected_head_type: ConceptType,
    ) -> Dict[str, float]:
        """Train for one epoch on one homogeneous head type.

        Args:
            dataloader: must yield Stage1Batch with homogeneous concept_types
            expected_head_type: target head type for this epoch

        Returns:
            dict with averaged metrics.
        """
        self.model.train()

        total_samples = 0
        total_loss = 0.0
        total_pos_sim = 0.0
        total_neg_sim = 0.0

        for batch in dataloader:
            if not isinstance(batch, Stage1Batch):
                raise TypeError("dataloader must yield Stage1Batch")

            self._assert_batch_head_type(batch, expected_head_type)
            batch = batch.to(self.device)

            encoded = self.model.encode_batch(batch)
            loss, debug = self.criterion(
                anchor=encoded["anchor_embeddings"],
                positives=encoded["positive_embeddings"],
                negatives=encoded["negative_embeddings"],
                return_debug=True,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            self.optimizer.step()

            bsz = int(encoded["anchor_embeddings"].shape[0])
            total_samples += bsz
            total_loss += float(loss.item()) * bsz
            total_pos_sim += float(debug["mean_positive_similarity"]) * bsz
            total_neg_sim += float(debug["mean_negative_similarity"]) * bsz

        return self._finalize_metrics(
            total_samples=total_samples,
            total_loss=total_loss,
            total_pos_sim=total_pos_sim,
            total_neg_sim=total_neg_sim,
            prefix="train",
        )

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        expected_head_type: ConceptType,
    ) -> Dict[str, float]:
        """Validate for one epoch on one homogeneous head type."""
        self.model.eval()

        total_samples = 0
        total_loss = 0.0
        total_pos_sim = 0.0
        total_neg_sim = 0.0

        for batch in dataloader:
            if not isinstance(batch, Stage1Batch):
                raise TypeError("dataloader must yield Stage1Batch")

            self._assert_batch_head_type(batch, expected_head_type)
            batch = batch.to(self.device)

            encoded = self.model.encode_batch(batch)
            loss, debug = self.criterion(
                anchor=encoded["anchor_embeddings"],
                positives=encoded["positive_embeddings"],
                negatives=encoded["negative_embeddings"],
                return_debug=True,
            )

            bsz = int(encoded["anchor_embeddings"].shape[0])
            total_samples += bsz
            total_loss += float(loss.item()) * bsz
            total_pos_sim += float(debug["mean_positive_similarity"]) * bsz
            total_neg_sim += float(debug["mean_negative_similarity"]) * bsz

        return self._finalize_metrics(
            total_samples=total_samples,
            total_loss=total_loss,
            total_pos_sim=total_pos_sim,
            total_neg_sim=total_neg_sim,
            prefix="val",
        )

    def _assert_batch_head_type(self, batch: Stage1Batch, expected_head_type: ConceptType) -> None:
        if not batch.concept_types:
            raise ValueError("Batch has empty concept_types")

        first = batch.concept_types[0]
        for ct in batch.concept_types:
            if ct != first:
                raise ValueError(
                    "Batch is not homogeneous. "
                    f"Found mixed concept types: {first} and {ct}."
                )

        if first != expected_head_type:
            raise ValueError(
                f"Expected head type {expected_head_type}, but got {first}."
            )

    @staticmethod
    def _finalize_metrics(
        total_samples: int,
        total_loss: float,
        total_pos_sim: float,
        total_neg_sim: float,
        prefix: str,
    ) -> Dict[str, float]:
        if total_samples <= 0:
            raise ValueError("No samples processed in epoch")

        mean_loss = total_loss / total_samples
        mean_pos = total_pos_sim / total_samples
        mean_neg = total_neg_sim / total_samples

        return {
            f"{prefix}/loss": mean_loss,
            f"{prefix}/mean_positive_similarity": mean_pos,
            f"{prefix}/mean_negative_similarity": mean_neg,
            f"{prefix}/num_samples": float(total_samples),
        }

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Reproducibility flags
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

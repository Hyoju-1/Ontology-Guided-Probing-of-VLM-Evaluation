from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from src.ontology.schema import ConceptType, Stage1Batch


class Stage1TeacherModel(nn.Module):
    """Stage 1 multi-head teacher model.

    Architecture:
      - One shared HuggingFace text encoder
      - Three lightweight heads:
          * frame_head
          * role_head
          * tuple_head

    Notes:
      - Designed for homogeneous batches only.
      - Output embeddings are L2-normalized.
    """

    def __init__(
        self,
        text_encoder_name: str = "bert-base-uncased",
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.text_encoder_name = text_encoder_name
        self.embedding_dim = int(embedding_dim)

        config = AutoConfig.from_pretrained(text_encoder_name)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name, config=config)
        hidden_size = int(config.hidden_size)

        self.frame_head = self._build_head(hidden_size, self.embedding_dim, dropout)
        self.role_head = self._build_head(hidden_size, self.embedding_dim, dropout)
        self.tuple_head = self._build_head(hidden_size, self.embedding_dim, dropout)

    @staticmethod
    def _build_head(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Dropout(p=dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        concept_type: ConceptType | str,
    ) -> torch.FloatTensor:
        """Encode text and project to a head-specific embedding space.

        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            concept_type: ConceptType or one of {"frame", "role", "typed_tuple", "tuple"}

        Returns:
            embeddings: [B, D] where D = embedding_dim (L2-normalized)
        """
        if input_ids.dim() != 2 or attention_mask.dim() != 2:
            raise ValueError(
                "forward expects 2D tensors: input_ids [B, L], attention_mask [B, L]"
            )

        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape")

        head_key = self._normalize_concept_type(concept_type)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # CLS pooling for encoder outputs: [B, L, H] -> [B, H]
        pooled = outputs.last_hidden_state[:, 0, :]

        if head_key == ConceptType.FRAME.value:
            embedding = self.frame_head(pooled)
        elif head_key == ConceptType.ROLE.value:
            embedding = self.role_head(pooled)
        else:
            embedding = self.tuple_head(pooled)

        # L2 normalize for contrastive learning stability
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding

    def encode_batch(self, batch: Stage1Batch) -> Dict[str, torch.FloatTensor]:
        """Encode a homogeneous Stage1Batch.

        Input tensor shapes from Stage1Batch:
          - anchor_input_ids: [B, L]
          - positive_input_ids: [B, P, L]
          - negative_input_ids: [B, N, L]

        Returns:
          {
            "anchor_embeddings": [B, D],
            "positive_embeddings": [B, P, D],
            "negative_embeddings": [B, N, D],
          }
        """
        self._validate_homogeneous_batch(batch.concept_types)

        concept_type = batch.concept_types[0]

        # Anchor: [B, L] -> [B, D]
        anchor_embeddings = self.forward(
            input_ids=batch.anchor_input_ids,
            attention_mask=batch.anchor_attention_mask,
            concept_type=concept_type,
        )

        # Positive: [B, P, L] -> [B*P, L] -> [B*P, D] -> [B, P, D]
        bsz, num_pos, seq_len = batch.positive_input_ids.shape
        pos_input_ids = batch.positive_input_ids.reshape(bsz * num_pos, seq_len)
        pos_attention_mask = batch.positive_attention_mask.reshape(bsz * num_pos, seq_len)
        pos_embeddings_flat = self.forward(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            concept_type=concept_type,
        )
        positive_embeddings = pos_embeddings_flat.reshape(bsz, num_pos, self.embedding_dim)

        # Negative: [B, N, L] -> [B*N, L] -> [B*N, D] -> [B, N, D]
        bsz_n, num_neg, seq_len_n = batch.negative_input_ids.shape
        if bsz_n != bsz:
            raise ValueError("positive and negative batch sizes must match")

        neg_input_ids = batch.negative_input_ids.reshape(bsz_n * num_neg, seq_len_n)
        neg_attention_mask = batch.negative_attention_mask.reshape(bsz_n * num_neg, seq_len_n)
        neg_embeddings_flat = self.forward(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            concept_type=concept_type,
        )
        negative_embeddings = neg_embeddings_flat.reshape(bsz_n, num_neg, self.embedding_dim)

        return {
            "anchor_embeddings": anchor_embeddings,
            "positive_embeddings": positive_embeddings,
            "negative_embeddings": negative_embeddings,
        }

    @staticmethod
    def _normalize_concept_type(concept_type: ConceptType | str) -> str:
        if isinstance(concept_type, ConceptType):
            return concept_type.value

        value = str(concept_type).strip().lower()
        if value == "tuple":
            return ConceptType.TYPED_TUPLE.value

        valid = {
            ConceptType.FRAME.value,
            ConceptType.ROLE.value,
            ConceptType.TYPED_TUPLE.value,
        }
        if value not in valid:
            raise ValueError(
                f"Unknown concept_type '{concept_type}'. "
                f"Expected one of: {sorted(valid)} or 'tuple'."
            )
        return value

    @staticmethod
    def _validate_homogeneous_batch(concept_types: Sequence[ConceptType]) -> None:
        if not concept_types:
            raise ValueError("Batch concept_types must not be empty")

        first = concept_types[0]
        for ct in concept_types:
            if ct != first:
                raise ValueError(
                    "encode_batch supports homogeneous batches only. "
                    f"Found mixed types: {first} and {ct}."
                )

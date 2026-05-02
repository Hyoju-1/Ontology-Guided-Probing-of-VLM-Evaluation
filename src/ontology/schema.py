from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import torch


class ConceptType(str, Enum):
    """Concept taxonomy for Stage 1 teacher space."""

    FRAME = "frame"
    ROLE = "role"
    TYPED_TUPLE = "typed_tuple"


@dataclass(slots=True)
class FrameConcept:
    """Frame concept (e.g., on, in, holding, riding)."""

    concept_id: str
    concept_type: ConceptType = ConceptType.FRAME

    name: str = ""
    definition: str = ""
    paraphrases: List[str] = field(default_factory=list)

    inverse_frame_id: Optional[str] = None
    confusing_frame_ids: List[str] = field(default_factory=list)

    typed_examples: List[str] = field(default_factory=list)
    role_descriptions: List[str] = field(default_factory=list)


@dataclass(slots=True)
class RoleConcept:
    """Role/order concept (e.g., subject, object, directionality)."""

    concept_id: str
    concept_type: ConceptType = ConceptType.ROLE

    name: str = ""
    definition: str = ""
    paraphrases: List[str] = field(default_factory=list)

    inverse_role_id: Optional[str] = None
    confusing_role_ids: List[str] = field(default_factory=list)

    applicable_frame_ids: List[str] = field(default_factory=list)
    role_descriptions: List[str] = field(default_factory=list)
    typed_examples: List[str] = field(default_factory=list)


@dataclass(slots=True)
class TypedTupleConcept:
    """Typed tuple concept: (subject_type, predicate, object_type)."""

    concept_id: str
    concept_type: ConceptType = ConceptType.TYPED_TUPLE

    subject_type: str = ""
    predicate: str = ""
    object_type: str = ""

    name: str = ""
    definition: str = ""
    paraphrases: List[str] = field(default_factory=list)

    swapped_tuple_id: Optional[str] = None
    inverse_tuple_ids: List[str] = field(default_factory=list)
    confusing_tuple_ids: List[str] = field(default_factory=list)

    typed_examples: List[str] = field(default_factory=list)
    role_descriptions: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ConceptTrainingItem:
    """One Stage 1 training sample for contrastive learning."""

    concept_id: str
    concept_type: ConceptType

    concept_name: str
    definition: str

    paraphrases: List[str] = field(default_factory=list)
    equivalent_descriptions: List[str] = field(default_factory=list)

    inverse_concept_id: Optional[str] = None
    confusing_concept_ids: List[str] = field(default_factory=list)

    typed_examples: List[str] = field(default_factory=list)
    role_descriptions: List[str] = field(default_factory=list)

    # Optional pre-built negatives by ID for deterministic experiments
    negative_concept_ids: List[str] = field(default_factory=list)


@dataclass(slots=True)
class Stage1Batch:
    """Tokenized batch for Stage 1 training.

    Tensor shapes:
      - anchor_input_ids: [B, L]
      - positive_input_ids: [B, P, L]
      - negative_input_ids: [B, N, L]
    """

    concept_ids: List[str]
    concept_types: List[ConceptType]

    anchor_input_ids: torch.LongTensor
    anchor_attention_mask: torch.LongTensor

    positive_input_ids: torch.LongTensor
    positive_attention_mask: torch.LongTensor

    negative_input_ids: torch.LongTensor
    negative_attention_mask: torch.LongTensor

    # IDs for analysis/debugging
    positive_source_ids: List[List[str]] = field(default_factory=list)
    negative_source_ids: List[List[str]] = field(default_factory=list)

    # Optional weighting for negatives (shape: [B, N])
    negative_weights: Optional[torch.FloatTensor] = None

    def batch_size(self) -> int:
        return int(self.anchor_input_ids.size(0))

    def to(self, device: torch.device | str) -> "Stage1Batch":
        return Stage1Batch(
            concept_ids=self.concept_ids,
            concept_types=self.concept_types,
            anchor_input_ids=self.anchor_input_ids.to(device),
            anchor_attention_mask=self.anchor_attention_mask.to(device),
            positive_input_ids=self.positive_input_ids.to(device),
            positive_attention_mask=self.positive_attention_mask.to(device),
            negative_input_ids=self.negative_input_ids.to(device),
            negative_attention_mask=self.negative_attention_mask.to(device),
            positive_source_ids=self.positive_source_ids,
            negative_source_ids=self.negative_source_ids,
            negative_weights=None if self.negative_weights is None else self.negative_weights.to(device),
        )


def ensure_non_empty_ids(ids: Sequence[str], field_name: str) -> None:
    """Utility validator for caller-side checks."""
    if any((not x) for x in ids):
        raise ValueError(f"{field_name} contains empty ID values")

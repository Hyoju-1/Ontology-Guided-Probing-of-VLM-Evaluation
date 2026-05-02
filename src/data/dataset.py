from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from src.ontology.schema import (
    ConceptTrainingItem,
    ConceptType,
    FrameConcept,
    RoleConcept,
    TypedTupleConcept,
)


@dataclass(slots=True)
class TrainingItemSplits:
    """Homogeneous training item lists by concept head type."""

    frame_items: List[ConceptTrainingItem]
    role_items: List[ConceptTrainingItem]
    tuple_items: List[ConceptTrainingItem]


class Stage1ConceptDataset:
    """Simple dataset wrapper that returns ConceptTrainingItem.

    This dataset must be homogeneous by concept type.
    """

    def __init__(self, items: Sequence[ConceptTrainingItem]):
        self._items: List[ConceptTrainingItem] = list(items)
        if not self._items:
            raise ValueError("Stage1ConceptDataset requires at least one item")

        first_type = self._items[0].concept_type
        for item in self._items:
            if item.concept_type != first_type:
                raise ValueError(
                    "Stage1ConceptDataset must be homogeneous by concept_type. "
                    f"Found mixed types: {first_type} and {item.concept_type}."
                )

        self._concept_type = first_type

    @property
    def concept_type(self) -> ConceptType:
        return self._concept_type

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> ConceptTrainingItem:
        return self._items[index]


def build_training_item_splits(
    frame_concepts_by_id: Mapping[str, FrameConcept],
    role_concepts_by_id: Mapping[str, RoleConcept],
    tuple_concepts_by_id: Mapping[str, TypedTupleConcept],
) -> TrainingItemSplits:
    """Build separate training item lists for frame/role/tuple heads."""

    frame_items = [
        _frame_to_training_item(concept) for concept in frame_concepts_by_id.values()
    ]
    role_items = [
        _role_to_training_item(concept) for concept in role_concepts_by_id.values()
    ]
    tuple_items = [
        _tuple_to_training_item(concept) for concept in tuple_concepts_by_id.values()
    ]

    return TrainingItemSplits(
        frame_items=frame_items,
        role_items=role_items,
        tuple_items=tuple_items,
    )


def build_concept_training_lookup(
    splits: TrainingItemSplits,
) -> Dict[str, ConceptTrainingItem]:
    """Build concept_id -> ConceptTrainingItem lookup dictionary."""

    lookup: Dict[str, ConceptTrainingItem] = {}
    for item in splits.frame_items + splits.role_items + splits.tuple_items:
        if item.concept_id in lookup:
            raise ValueError(f"Duplicate concept_id in training items: {item.concept_id}")
        lookup[item.concept_id] = item
    return lookup


def _frame_to_training_item(concept: FrameConcept) -> ConceptTrainingItem:
    negatives = _unique_ids(
        [concept.inverse_frame_id] + concept.confusing_frame_ids
    )
    return ConceptTrainingItem(
        concept_id=concept.concept_id,
        concept_type=ConceptType.FRAME,
        concept_name=concept.name,
        definition=concept.definition,
        paraphrases=list(concept.paraphrases),
        equivalent_descriptions=[],
        inverse_concept_id=concept.inverse_frame_id,
        confusing_concept_ids=list(concept.confusing_frame_ids),
        typed_examples=list(concept.typed_examples),
        role_descriptions=list(concept.role_descriptions),
        negative_concept_ids=negatives,
    )


def _role_to_training_item(concept: RoleConcept) -> ConceptTrainingItem:
    negatives = _unique_ids(
        [concept.inverse_role_id] + concept.confusing_role_ids
    )
    return ConceptTrainingItem(
        concept_id=concept.concept_id,
        concept_type=ConceptType.ROLE,
        concept_name=concept.name,
        definition=concept.definition,
        paraphrases=list(concept.paraphrases),
        equivalent_descriptions=[],
        inverse_concept_id=concept.inverse_role_id,
        confusing_concept_ids=list(concept.confusing_role_ids),
        typed_examples=list(concept.typed_examples),
        role_descriptions=list(concept.role_descriptions),
        negative_concept_ids=negatives,
    )


def _tuple_to_training_item(concept: TypedTupleConcept) -> ConceptTrainingItem:
    primary_inverse: Optional[str] = concept.swapped_tuple_id
    if primary_inverse is None and concept.inverse_tuple_ids:
        primary_inverse = concept.inverse_tuple_ids[0]

    negatives = _unique_ids(
        [concept.swapped_tuple_id] + concept.inverse_tuple_ids + concept.confusing_tuple_ids
    )

    return ConceptTrainingItem(
        concept_id=concept.concept_id,
        concept_type=ConceptType.TYPED_TUPLE,
        concept_name=concept.name,
        definition=concept.definition,
        paraphrases=list(concept.paraphrases),
        equivalent_descriptions=[
            _tuple_structural_text(concept)
        ],
        inverse_concept_id=primary_inverse,
        confusing_concept_ids=list(concept.confusing_tuple_ids),
        typed_examples=list(concept.typed_examples),
        role_descriptions=list(concept.role_descriptions),
        negative_concept_ids=negatives,
    )


def _tuple_structural_text(concept: TypedTupleConcept) -> str:
    return f"({concept.subject_type}, {concept.predicate}, {concept.object_type})"


def _unique_ids(values: Iterable[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure "src" imports work when running: python scripts/quick_test.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.batch import Stage1Collator, Stage1CollatorConfig
from src.data.dataset import build_concept_training_lookup, build_training_item_splits
from src.losses.contrastive import Stage1Loss
from src.models.teacher import Stage1TeacherModel
from src.ontology.schema import FrameConcept, RoleConcept, TypedTupleConcept


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Tiny toy ontology in memory (no file I/O)
    # ------------------------------------------------------------------
    frame_concepts_by_id = {
        "frame_on": FrameConcept(
            concept_id="frame_on",
            name="on",
            definition="A is on B",
            paraphrases=["A rests on B", "A is located on B"],
            inverse_frame_id="frame_under",
            confusing_frame_ids=["frame_above"],
        ),
        "frame_under": FrameConcept(
            concept_id="frame_under",
            name="under",
            definition="A is under B",
            paraphrases=["A is below B"],
            inverse_frame_id="frame_on",
            confusing_frame_ids=["frame_below"],
        ),
        "frame_above": FrameConcept(
            concept_id="frame_above",
            name="above",
            definition="A is above B without contact",
            paraphrases=["A is higher than B"],
            inverse_frame_id=None,
            confusing_frame_ids=["frame_on"],
        ),
        "frame_below": FrameConcept(
            concept_id="frame_below",
            name="below",
            definition="A is below B",
            paraphrases=["A is lower than B"],
            inverse_frame_id=None,
            confusing_frame_ids=["frame_under"],
        ),
    }

    role_concepts_by_id = {
        "role_subject": RoleConcept(
            concept_id="role_subject",
            name="subject",
            definition="the primary entity in a relation",
            paraphrases=["main actor"],
            inverse_role_id="role_object",
            confusing_role_ids=[],
            applicable_frame_ids=["frame_on", "frame_under"],
        ),
        "role_object": RoleConcept(
            concept_id="role_object",
            name="object",
            definition="the secondary entity in a relation",
            paraphrases=["target entity"],
            inverse_role_id="role_subject",
            confusing_role_ids=[],
            applicable_frame_ids=["frame_on", "frame_under"],
        ),
    }

    tuple_concepts_by_id = {
        "tuple_person_on_chair": TypedTupleConcept(
            concept_id="tuple_person_on_chair",
            subject_type="person",
            predicate="on",
            object_type="chair",
            name="person_on_chair",
            definition="a person is on a chair",
            paraphrases=["person sits on chair"],
            swapped_tuple_id="tuple_chair_on_person",
            inverse_tuple_ids=[],
            confusing_tuple_ids=["tuple_person_above_chair"],
        ),
        "tuple_chair_on_person": TypedTupleConcept(
            concept_id="tuple_chair_on_person",
            subject_type="chair",
            predicate="on",
            object_type="person",
            name="chair_on_person",
            definition="a chair is on a person",
            paraphrases=["chair placed on person"],
            swapped_tuple_id="tuple_person_on_chair",
            inverse_tuple_ids=[],
            confusing_tuple_ids=[],
        ),
        "tuple_person_above_chair": TypedTupleConcept(
            concept_id="tuple_person_above_chair",
            subject_type="person",
            predicate="above",
            object_type="chair",
            name="person_above_chair",
            definition="a person is above a chair",
            paraphrases=["person higher than chair"],
            swapped_tuple_id=None,
            inverse_tuple_ids=[],
            confusing_tuple_ids=["tuple_person_on_chair"],
        ),
    }

    # ------------------------------------------------------------------
    # 2) Build ConceptTrainingItem samples
    # ------------------------------------------------------------------
    splits = build_training_item_splits(
        frame_concepts_by_id=frame_concepts_by_id,
        role_concepts_by_id=role_concepts_by_id,
        tuple_concepts_by_id=tuple_concepts_by_id,
    )
    concept_lookup = build_concept_training_lookup(splits)

    # Homogeneous batch only -> use frame samples only
    frame_items = splits.frame_items[:2]
    assert len(frame_items) == 2

    # ------------------------------------------------------------------
    # 3) Run collator
    # ------------------------------------------------------------------
    collator = Stage1Collator(
        concept_lookup_by_id=concept_lookup,
        config=Stage1CollatorConfig(
            tokenizer_name="bert-base-uncased",
            max_length=24,
            max_positive_texts=2,
            max_negative_texts=2,
        ),
    )
    batch = collator(frame_items)

    bsz = len(frame_items)
    assert batch.anchor_input_ids.ndim == 2
    assert batch.anchor_input_ids.shape[0] == bsz
    assert batch.positive_input_ids.ndim == 3
    assert batch.negative_input_ids.ndim == 3

    # ------------------------------------------------------------------
    # 4) Run teacher forward / encode_batch
    # ------------------------------------------------------------------
    model = Stage1TeacherModel(
        text_encoder_name="bert-base-uncased",
        embedding_dim=64,
        dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        encoded = model.encode_batch(batch)

    anchor = encoded["anchor_embeddings"]
    positives = encoded["positive_embeddings"]
    negatives = encoded["negative_embeddings"]

    assert anchor.shape == (bsz, 64)
    assert positives.shape[:2] == (bsz, 2)
    assert positives.shape[2] == 64
    assert negatives.shape[:2] == (bsz, 2)
    assert negatives.shape[2] == 64

    # ------------------------------------------------------------------
    # 5) Run contrastive loss
    # ------------------------------------------------------------------
    loss_fn = Stage1Loss(temperature=0.07)
    loss, debug = loss_fn(
        anchor=anchor,
        positives=positives,
        negatives=negatives,
        return_debug=True,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss).item(), "Loss must be finite"
    assert "mean_positive_similarity" in debug
    assert "mean_negative_similarity" in debug

    # ------------------------------------------------------------------
    # 6) Success message
    # ------------------------------------------------------------------
    print(
        "[OK] Stage1 MVP quick test passed | "
        f"loss={loss.item():.6f}, "
        f"mean_pos={debug['mean_positive_similarity']:.6f}, "
        f"mean_neg={debug['mean_negative_similarity']:.6f}"
    )


if __name__ == "__main__":
    main()

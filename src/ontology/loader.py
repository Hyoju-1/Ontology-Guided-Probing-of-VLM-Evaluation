from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import yaml

from .schema import FrameConcept, RoleConcept, TypedTupleConcept


@dataclass(slots=True)
class OntologyStore:
    """Container for ontology concepts indexed by ID."""

    frame_concepts_by_id: Dict[str, FrameConcept]
    role_concepts_by_id: Dict[str, RoleConcept]
    tuple_concepts_by_id: Dict[str, TypedTupleConcept]
    all_concepts_by_id: Dict[str, Any]


def load_ontology_yaml(yaml_path: str | Path) -> OntologyStore:
    """Load ontology YAML and build concept dictionaries.

    Expected top-level keys:
      - frames
      - roles
      - typed_tuples (or tuples)

    Each section supports either:
      1) dict format: {concept_id: {...fields...}}
      2) list format: [{concept_id: "...", ...fields...}, ...]
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Ontology YAML not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, Mapping):
        raise ValueError("Ontology YAML root must be a mapping/dictionary")

    frame_concepts_by_id = _build_frame_concepts(raw.get("frames", {}))
    role_concepts_by_id = _build_role_concepts(raw.get("roles", {}))
    tuple_section = raw.get("typed_tuples", raw.get("tuples", {}))
    tuple_concepts_by_id = _build_tuple_concepts(tuple_section)

    all_concepts_by_id: Dict[str, Any] = {}
    all_concepts_by_id.update(frame_concepts_by_id)
    all_concepts_by_id.update(role_concepts_by_id)
    all_concepts_by_id.update(tuple_concepts_by_id)

    _validate_unique_ids(
        frame_concepts_by_id,
        role_concepts_by_id,
        tuple_concepts_by_id,
    )
    _validate_references(
        frame_concepts_by_id=frame_concepts_by_id,
        role_concepts_by_id=role_concepts_by_id,
        tuple_concepts_by_id=tuple_concepts_by_id,
    )

    return OntologyStore(
        frame_concepts_by_id=frame_concepts_by_id,
        role_concepts_by_id=role_concepts_by_id,
        tuple_concepts_by_id=tuple_concepts_by_id,
        all_concepts_by_id=all_concepts_by_id,
    )


def _build_frame_concepts(section: Any) -> Dict[str, FrameConcept]:
    concepts: Dict[str, FrameConcept] = {}
    for concept_id, payload in _iter_section(section):
        concepts[concept_id] = FrameConcept(
            concept_id=concept_id,
            name=_as_str(payload.get("name", "")),
            definition=_as_str(payload.get("definition", "")),
            paraphrases=_as_str_list(payload.get("paraphrases")),
            inverse_frame_id=_as_optional_str(payload.get("inverse_frame_id")),
            confusing_frame_ids=_as_str_list(payload.get("confusing_frame_ids")),
            typed_examples=_as_str_list(payload.get("typed_examples")),
            role_descriptions=_as_str_list(payload.get("role_descriptions")),
        )
    return concepts


def _build_role_concepts(section: Any) -> Dict[str, RoleConcept]:
    concepts: Dict[str, RoleConcept] = {}
    for concept_id, payload in _iter_section(section):
        concepts[concept_id] = RoleConcept(
            concept_id=concept_id,
            name=_as_str(payload.get("name", "")),
            definition=_as_str(payload.get("definition", "")),
            paraphrases=_as_str_list(payload.get("paraphrases")),
            inverse_role_id=_as_optional_str(payload.get("inverse_role_id")),
            confusing_role_ids=_as_str_list(payload.get("confusing_role_ids")),
            applicable_frame_ids=_as_str_list(payload.get("applicable_frame_ids")),
            role_descriptions=_as_str_list(payload.get("role_descriptions")),
            typed_examples=_as_str_list(payload.get("typed_examples")),
        )
    return concepts


def _build_tuple_concepts(section: Any) -> Dict[str, TypedTupleConcept]:
    concepts: Dict[str, TypedTupleConcept] = {}
    for concept_id, payload in _iter_section(section):
        concepts[concept_id] = TypedTupleConcept(
            concept_id=concept_id,
            subject_type=_as_str(payload.get("subject_type", "")),
            predicate=_as_str(payload.get("predicate", "")),
            object_type=_as_str(payload.get("object_type", "")),
            name=_as_str(payload.get("name", "")),
            definition=_as_str(payload.get("definition", "")),
            paraphrases=_as_str_list(payload.get("paraphrases")),
            swapped_tuple_id=_as_optional_str(payload.get("swapped_tuple_id")),
            inverse_tuple_ids=_as_str_list(payload.get("inverse_tuple_ids")),
            confusing_tuple_ids=_as_str_list(payload.get("confusing_tuple_ids")),
            typed_examples=_as_str_list(payload.get("typed_examples")),
            role_descriptions=_as_str_list(payload.get("role_descriptions")),
        )
    return concepts


def _iter_section(section: Any) -> Iterator[Tuple[str, Mapping[str, Any]]]:
    """Iterate section entries in either dict or list format."""
    if section is None:
        return iter(())

    if isinstance(section, Mapping):
        for concept_id, payload in section.items():
            cid = _as_str(concept_id)
            if not cid:
                raise ValueError("Empty concept_id found in mapping section")
            if not isinstance(payload, Mapping):
                raise ValueError(f"Concept payload for '{cid}' must be a mapping")
            yield cid, payload
        return

    if isinstance(section, list):
        for idx, item in enumerate(section):
            if not isinstance(item, Mapping):
                raise ValueError(f"List section item at index {idx} must be a mapping")
            cid = _as_str(item.get("concept_id", ""))
            if not cid:
                raise ValueError(f"Missing concept_id in list section item at index {idx}")
            yield cid, item
        return

    raise ValueError("Section must be either mapping or list")


def _validate_unique_ids(
    frame_concepts_by_id: Mapping[str, FrameConcept],
    role_concepts_by_id: Mapping[str, RoleConcept],
    tuple_concepts_by_id: Mapping[str, TypedTupleConcept],
) -> None:
    frame_ids = set(frame_concepts_by_id.keys())
    role_ids = set(role_concepts_by_id.keys())
    tuple_ids = set(tuple_concepts_by_id.keys())

    duplicate = (frame_ids & role_ids) | (frame_ids & tuple_ids) | (role_ids & tuple_ids)
    if duplicate:
        joined = ", ".join(sorted(duplicate))
        raise ValueError(f"Duplicate concept_id across sections: {joined}")


def _validate_references(
    frame_concepts_by_id: Mapping[str, FrameConcept],
    role_concepts_by_id: Mapping[str, RoleConcept],
    tuple_concepts_by_id: Mapping[str, TypedTupleConcept],
) -> None:
    errors: List[str] = []

    frame_ids = set(frame_concepts_by_id.keys())
    role_ids = set(role_concepts_by_id.keys())
    tuple_ids = set(tuple_concepts_by_id.keys())

    # Frame refs
    for cid, concept in frame_concepts_by_id.items():
        if concept.inverse_frame_id and concept.inverse_frame_id not in frame_ids:
            errors.append(
                f"Frame '{cid}' inverse_frame_id '{concept.inverse_frame_id}' not found"
            )
        for ref in concept.confusing_frame_ids:
            if ref not in frame_ids:
                errors.append(f"Frame '{cid}' confusing_frame_id '{ref}' not found")

    # Role refs
    for cid, concept in role_concepts_by_id.items():
        if concept.inverse_role_id and concept.inverse_role_id not in role_ids:
            errors.append(
                f"Role '{cid}' inverse_role_id '{concept.inverse_role_id}' not found"
            )
        for ref in concept.confusing_role_ids:
            if ref not in role_ids:
                errors.append(f"Role '{cid}' confusing_role_id '{ref}' not found")

    # Tuple refs
    for cid, concept in tuple_concepts_by_id.items():
        if concept.swapped_tuple_id and concept.swapped_tuple_id not in tuple_ids:
            errors.append(
                f"TypedTuple '{cid}' swapped_tuple_id '{concept.swapped_tuple_id}' not found"
            )
        for ref in concept.inverse_tuple_ids:
            if ref not in tuple_ids:
                errors.append(f"TypedTuple '{cid}' inverse_tuple_id '{ref}' not found")
        for ref in concept.confusing_tuple_ids:
            if ref not in tuple_ids:
                errors.append(f"TypedTuple '{cid}' confusing_tuple_id '{ref}' not found")

    if errors:
        msg = "\n".join(errors)
        raise ValueError(f"Ontology reference validation failed:\n{msg}")


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_optional_str(value: Any) -> Optional[str]:
    v = _as_str(value)
    return v if v else None


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_as_str(v) for v in value if _as_str(v)]
    # Allow scalar fallback for robustness
    single = _as_str(value)
    return [single] if single else []

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from transformers import AutoTokenizer

from src.models.teacher import Stage1TeacherModel
from src.ontology.loader import OntologyStore
from src.ontology.schema import ConceptType


@dataclass(slots=True)
class PrototypeExportConfig:
    """Configuration for Stage 1 prototype export."""

    max_length: int = 128
    batch_size: int = 128
    device: str = "cuda"


def export_stage1_prototype_banks(
    model: Stage1TeacherModel,
    ontology_store: OntologyStore,
    output_dir: str | Path,
    config: PrototypeExportConfig | None = None,
) -> Dict[str, str]:
    """Export separate prototype banks and mapping files for Stage 2.

    Output files:
      - frame_prototypes.pt
      - role_prototypes.pt
      - tuple_prototypes.pt
      - frame_mapping.json
      - role_mapping.json
      - tuple_mapping.json
      - export_config.json
    """
    cfg = config or PrototypeExportConfig()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model.text_encoder_name)

    frame_ids = sorted(ontology_store.frame_concepts_by_id.keys())
    role_ids = sorted(ontology_store.role_concepts_by_id.keys())
    tuple_ids = sorted(ontology_store.tuple_concepts_by_id.keys())

    frame_texts = [
        _build_concept_text(
            ontology_store.frame_concepts_by_id[cid].name,
            ontology_store.frame_concepts_by_id[cid].definition,
        )
        for cid in frame_ids
    ]
    role_texts = [
        _build_concept_text(
            ontology_store.role_concepts_by_id[cid].name,
            ontology_store.role_concepts_by_id[cid].definition,
        )
        for cid in role_ids
    ]
    tuple_texts = [
        _build_concept_text(
            ontology_store.tuple_concepts_by_id[cid].name,
            ontology_store.tuple_concepts_by_id[cid].definition,
        )
        for cid in tuple_ids
    ]

    frame_prototypes = _encode_texts_to_prototypes(
        model=model,
        tokenizer=tokenizer,
        texts=frame_texts,
        concept_type=ConceptType.FRAME,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        device=device,
    )
    role_prototypes = _encode_texts_to_prototypes(
        model=model,
        tokenizer=tokenizer,
        texts=role_texts,
        concept_type=ConceptType.ROLE,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        device=device,
    )
    tuple_prototypes = _encode_texts_to_prototypes(
        model=model,
        tokenizer=tokenizer,
        texts=tuple_texts,
        concept_type=ConceptType.TYPED_TUPLE,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        device=device,
    )

    frame_proto_path = out_dir / "frame_prototypes.pt"
    role_proto_path = out_dir / "role_prototypes.pt"
    tuple_proto_path = out_dir / "tuple_prototypes.pt"

    torch.save(frame_prototypes.cpu(), frame_proto_path)
    torch.save(role_prototypes.cpu(), role_proto_path)
    torch.save(tuple_prototypes.cpu(), tuple_proto_path)

    frame_mapping_path = out_dir / "frame_mapping.json"
    role_mapping_path = out_dir / "role_mapping.json"
    tuple_mapping_path = out_dir / "tuple_mapping.json"

    _save_mapping_json(frame_ids, frame_mapping_path, ConceptType.FRAME.value)
    _save_mapping_json(role_ids, role_mapping_path, ConceptType.ROLE.value)
    _save_mapping_json(tuple_ids, tuple_mapping_path, ConceptType.TYPED_TUPLE.value)

    config_path = out_dir / "export_config.json"
    _save_export_config(
        path=config_path,
        model=model,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        frame_count=len(frame_ids),
        role_count=len(role_ids),
        tuple_count=len(tuple_ids),
    )

    return {
        "frame_prototypes": str(frame_proto_path),
        "role_prototypes": str(role_proto_path),
        "tuple_prototypes": str(tuple_proto_path),
        "frame_mapping": str(frame_mapping_path),
        "role_mapping": str(role_mapping_path),
        "tuple_mapping": str(tuple_mapping_path),
        "config": str(config_path),
    }


def _encode_texts_to_prototypes(
    model: Stage1TeacherModel,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    concept_type: ConceptType,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode concept texts into normalized prototype embeddings.

    Returns:
      Tensor with shape [N, D]
    """
    if not texts:
        return torch.empty((0, model.embedding_dim), dtype=torch.float32)

    chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            tokens = tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            emb = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                concept_type=concept_type,
            )
            chunks.append(emb.detach().to(dtype=torch.float32).cpu())

    return torch.cat(chunks, dim=0)


def _build_concept_text(name: str, definition: str) -> str:
    name = (name or "").strip()
    definition = (definition or "").strip()

    if name and definition:
        return f"{name}: {definition}"
    if definition:
        return definition
    if name:
        return name
    return "unknown concept"


def _save_mapping_json(concept_ids: Sequence[str], path: Path, head_type: str) -> None:
    mapping = {
        "head_type": head_type,
        "concept_ids": list(concept_ids),
        "concept_id_to_index": {cid: idx for idx, cid in enumerate(concept_ids)},
        "num_concepts": len(concept_ids),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def _save_export_config(
    path: Path,
    model: Stage1TeacherModel,
    max_length: int,
    batch_size: int,
    frame_count: int,
    role_count: int,
    tuple_count: int,
) -> None:
    data = {
        "version": "stage1_mvp_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "text_encoder_name": model.text_encoder_name,
        "embedding_dim": int(model.embedding_dim),
        "l2_normalized": True,
        "tokenizer_max_length": int(max_length),
        "export_batch_size": int(batch_size),
        "prototype_files": {
            "frame": "frame_prototypes.pt",
            "role": "role_prototypes.pt",
            "typed_tuple": "tuple_prototypes.pt",
        },
        "mapping_files": {
            "frame": "frame_mapping.json",
            "role": "role_mapping.json",
            "typed_tuple": "tuple_mapping.json",
        },
        "num_concepts": {
            "frame": int(frame_count),
            "role": int(role_count),
            "typed_tuple": int(tuple_count),
        },
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class PrototypeBanks:
    frame_bank: torch.Tensor
    tuple_bank: torch.Tensor
    role_bank: Optional[torch.Tensor]
    frame_id_to_index: Dict[str, int]
    tuple_id_to_index: Dict[str, int]
    role_id_to_index: Optional[Dict[str, int]]


def load_stage1_export_banks(
    export_dir: str | Path,
    device: torch.device,
    use_role_head: bool = False,
    frame_proto_file: str = "frame_prototypes.pt",
    tuple_proto_file: str = "tuple_prototypes.pt",
    role_proto_file: str = "role_prototypes.pt",
    frame_map_file: str = "frame_mapping.json",
    tuple_map_file: str = "tuple_mapping.json",
    role_map_file: str = "role_mapping.json",
) -> PrototypeBanks:
    root = Path(export_dir)

    frame_bank = _load_bank_tensor(root / frame_proto_file, device)
    tuple_bank = _load_bank_tensor(root / tuple_proto_file, device)

    frame_map = _load_mapping(root / frame_map_file)
    tuple_map = _load_mapping(root / tuple_map_file)

    _validate_bank_vs_mapping(frame_bank, frame_map, "frame")
    _validate_bank_vs_mapping(tuple_bank, tuple_map, "typed_tuple")

    role_bank: Optional[torch.Tensor] = None
    role_id_to_index: Optional[Dict[str, int]] = None

    if use_role_head:
        role_bank_path = root / role_proto_file
        role_map_path = root / role_map_file
        if not role_bank_path.exists() or not role_map_path.exists():
            raise FileNotFoundError(
                "Role head is enabled, but role prototype/mapping files are missing: "
                f"{role_bank_path}, {role_map_path}"
            )
        role_bank = _load_bank_tensor(role_bank_path, device)
        role_map = _load_mapping(role_map_path)
        _validate_bank_vs_mapping(role_bank, role_map, "role")
        role_id_to_index = dict(role_map["concept_id_to_index"])

    return PrototypeBanks(
        frame_bank=frame_bank,
        tuple_bank=tuple_bank,
        role_bank=role_bank,
        frame_id_to_index=dict(frame_map["concept_id_to_index"]),
        tuple_id_to_index=dict(tuple_map["concept_id_to_index"]),
        role_id_to_index=role_id_to_index,
    )


def load_stage1_mapping_dict(path: str | Path) -> Dict[str, int]:
    data = _load_mapping(Path(path))
    return dict(data["concept_id_to_index"])


def _load_bank_tensor(path: Path, device: torch.device) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Prototype bank not found: {path}")

    bank = torch.load(path, map_location="cpu")
    if not isinstance(bank, torch.Tensor):
        raise ValueError(f"Prototype bank must be torch.Tensor: {path}")

    bank = F.normalize(bank.to(dtype=torch.float32), p=2, dim=-1)
    bank = bank.to(device)
    bank.requires_grad_(False)
    return bank


def _load_mapping(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Mapping JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "concept_id_to_index" not in data:
        raise ValueError(f"Invalid mapping JSON (missing concept_id_to_index): {path}")
    if "num_concepts" not in data:
        raise ValueError(f"Invalid mapping JSON (missing num_concepts): {path}")
    return data


def _validate_bank_vs_mapping(bank: torch.Tensor, mapping: Dict, name: str) -> None:
    num = int(mapping["num_concepts"])
    ids = mapping.get("concept_ids", [])
    m = mapping.get("concept_id_to_index", {})

    if bank.ndim != 2:
        raise ValueError(f"{name} prototype bank must be rank-2, got shape={tuple(bank.shape)}")

    if bank.size(0) != num:
        raise ValueError(
            f"{name} bank row mismatch: tensor_rows={bank.size(0)} vs mapping.num_concepts={num}"
        )

    if len(ids) != num or len(m) != num:
        raise ValueError(
            f"{name} mapping mismatch: num_concepts={num}, len(concept_ids)={len(ids)}, len(id_to_index)={len(m)}"
        )

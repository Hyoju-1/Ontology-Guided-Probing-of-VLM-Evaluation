from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor


@dataclass(slots=True)
class Stage2LabelContractReport:
    total_records: int
    valid_records: int
    invalid_records: int
    missing_required_fields: int
    unknown_frame_ids: int
    unknown_tuple_ids: int
    unknown_role_ids: int
    missing_role_ids: int
    missing_images: int
    sample_errors: List[str]


@dataclass(slots=True)
class Stage2IndexedSample:
    image_path: Path
    text: str
    frame_label: int
    tuple_label: int
    role_label: int  # -1 means missing


def load_jsonl_records(path: str | Path) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")

    records: List[Dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {p}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be object at {p}:{line_no}")
            records.append(obj)

    if not records:
        raise ValueError(f"No records found in JSONL file: {p}")
    return records


def validate_stage2_label_contract(
    records: Sequence[Dict],
    jsonl_path: str | Path,
    frame_id_to_index: Dict[str, int],
    tuple_id_to_index: Dict[str, int],
    role_id_to_index: Optional[Dict[str, int]] = None,
    require_role: bool = False,
    check_image_exists: bool = True,
    max_errors_to_keep: int = 20,
) -> Stage2LabelContractReport:
    base_dir = Path(jsonl_path).parent

    total = 0
    valid = 0
    missing_required_fields = 0
    unknown_frame_ids = 0
    unknown_tuple_ids = 0
    unknown_role_ids = 0
    missing_role_ids = 0
    missing_images = 0
    sample_errors: List[str] = []

    for idx, row in enumerate(records):
        total += 1
        row_id = row.get("sample_id", f"row_{idx}")
        row_errors: List[str] = []

        image_path_raw = row.get("image_path")
        text = row.get("text")
        frame_id = row.get("frame_id")
        tuple_id = row.get("tuple_id")
        role_id = row.get("role_id")

        if not isinstance(image_path_raw, str) or not image_path_raw.strip():
            row_errors.append("missing image_path")
            missing_required_fields += 1

        if not isinstance(text, str) or not text.strip():
            row_errors.append("missing text")
            missing_required_fields += 1

        if not isinstance(frame_id, str) or not frame_id.strip():
            row_errors.append("missing frame_id")
            missing_required_fields += 1
        elif frame_id not in frame_id_to_index:
            row_errors.append(f"unknown frame_id='{frame_id}'")
            unknown_frame_ids += 1

        if not isinstance(tuple_id, str) or not tuple_id.strip():
            row_errors.append("missing tuple_id")
            missing_required_fields += 1
        elif tuple_id not in tuple_id_to_index:
            row_errors.append(f"unknown tuple_id='{tuple_id}'")
            unknown_tuple_ids += 1

        if require_role and (not isinstance(role_id, str) or not role_id.strip()):
            row_errors.append("missing role_id (required)")
            missing_role_ids += 1

        if isinstance(role_id, str) and role_id.strip():
            if role_id_to_index is None:
                row_errors.append("role_id is provided but role mapping is unavailable")
                unknown_role_ids += 1
            elif role_id not in role_id_to_index:
                row_errors.append(f"unknown role_id='{role_id}'")
                unknown_role_ids += 1

        if check_image_exists and isinstance(image_path_raw, str) and image_path_raw.strip():
            img_path = Path(image_path_raw)
            if not img_path.is_absolute():
                img_path = (base_dir / img_path).resolve()
            if not img_path.exists():
                row_errors.append(f"missing image file='{img_path}'")
                missing_images += 1

        if row_errors:
            if len(sample_errors) < max_errors_to_keep:
                sample_errors.append(f"{row_id}: {', '.join(row_errors)}")
        else:
            valid += 1

    return Stage2LabelContractReport(
        total_records=total,
        valid_records=valid,
        invalid_records=total - valid,
        missing_required_fields=missing_required_fields,
        unknown_frame_ids=unknown_frame_ids,
        unknown_tuple_ids=unknown_tuple_ids,
        unknown_role_ids=unknown_role_ids,
        missing_role_ids=missing_role_ids,
        missing_images=missing_images,
        sample_errors=sample_errors,
    )


def index_stage2_samples(
    records: Sequence[Dict],
    jsonl_path: str | Path,
    frame_id_to_index: Dict[str, int],
    tuple_id_to_index: Dict[str, int],
    role_id_to_index: Optional[Dict[str, int]] = None,
) -> List[Stage2IndexedSample]:
    base_dir = Path(jsonl_path).parent
    indexed: List[Stage2IndexedSample] = []

    for row in records:
        image_path_raw = row["image_path"]
        text = row["text"].strip()
        frame_id = row["frame_id"]
        tuple_id = row["tuple_id"]
        role_id = row.get("role_id")

        img_path = Path(image_path_raw)
        if not img_path.is_absolute():
            img_path = (base_dir / img_path).resolve()

        role_label = -1
        if isinstance(role_id, str) and role_id.strip() and role_id_to_index is not None:
            role_label = int(role_id_to_index[role_id])

        indexed.append(
            Stage2IndexedSample(
                image_path=img_path,
                text=text,
                frame_label=int(frame_id_to_index[frame_id]),
                tuple_label=int(tuple_id_to_index[tuple_id]),
                role_label=role_label,
            )
        )

    return indexed


class Stage2DistillDataset(Dataset):
    """Stage 2 dataset with explicit ontology-label contract.

    Expected JSONL keys per row:
      - image_path: str
      - text: str
      - frame_id: str
      - tuple_id: str
      - role_id: str (optional)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        frame_id_to_index: Dict[str, int],
        tuple_id_to_index: Dict[str, int],
        role_id_to_index: Optional[Dict[str, int]] = None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        require_role: bool = False,
        check_image_exists: bool = True,
        fail_on_invalid: bool = True,
        max_errors_to_keep: int = 20,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        records = load_jsonl_records(self.jsonl_path)
        report = validate_stage2_label_contract(
            records=records,
            jsonl_path=self.jsonl_path,
            frame_id_to_index=frame_id_to_index,
            tuple_id_to_index=tuple_id_to_index,
            role_id_to_index=role_id_to_index,
            require_role=require_role,
            check_image_exists=check_image_exists,
            max_errors_to_keep=max_errors_to_keep,
        )
        self.contract_report = report

        if fail_on_invalid and report.invalid_records > 0:
            lines = [
                f"Stage2 label contract failed for {self.jsonl_path}",
                f"- total: {report.total_records}",
                f"- invalid: {report.invalid_records}",
                f"- missing_required_fields: {report.missing_required_fields}",
                f"- unknown_frame_ids: {report.unknown_frame_ids}",
                f"- unknown_tuple_ids: {report.unknown_tuple_ids}",
                f"- unknown_role_ids: {report.unknown_role_ids}",
                f"- missing_role_ids: {report.missing_role_ids}",
                f"- missing_images: {report.missing_images}",
            ]
            lines.extend([f"  * {msg}" for msg in report.sample_errors])
            raise ValueError("\n".join(lines))

        if report.valid_records == 0:
            raise ValueError(f"No valid records after contract validation: {self.jsonl_path}")

        self.samples = index_stage2_samples(
            records=records,
            jsonl_path=self.jsonl_path,
            frame_id_to_index=frame_id_to_index,
            tuple_id_to_index=tuple_id_to_index,
            role_id_to_index=role_id_to_index,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")

        encoded = self.processor(
            text=sample.text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "frame_labels": torch.tensor(sample.frame_label, dtype=torch.long),
            "tuple_labels": torch.tensor(sample.tuple_label, dtype=torch.long),
            "role_labels": torch.tensor(sample.role_label, dtype=torch.long),
        }


class Stage2Collator:
    """Simple collator for Stage 2 tensors."""

    def __call__(self, batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Empty Stage2 batch")

        keys = batch[0].keys()
        out: Dict[str, torch.Tensor] = {}
        for key in keys:
            out[key] = torch.stack([item[key] for item in batch], dim=0)
        return out

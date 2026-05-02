from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure "src" imports work when running: python scripts/export_teacher.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.teacher import Stage1TeacherModel
from src.ontology.loader import load_ontology_yaml
from src.utils.checkpoint import PrototypeExportConfig, export_stage1_prototype_banks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage 1 teacher prototype banks")

    parser.add_argument("--ontology-yaml", type=str, required=True, help="Path to ontology YAML")
    parser.add_argument("--teacher-checkpoint", type=str, required=True, help="Path to trained teacher checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save exported prototype banks")

    parser.add_argument("--text-encoder-name", type=str, default="bert-base-uncased")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def _load_teacher_checkpoint(model: Stage1TeacherModel, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Support both raw state_dict and wrapped checkpoint format
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        raise ValueError("Invalid checkpoint format: state_dict not found")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise ValueError(f"Missing keys when loading teacher checkpoint: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected keys when loading teacher checkpoint: {unexpected}")


def main() -> None:
    args = parse_args()

    # Accept common alias used in CLI examples.
    if args.device.lower() == "gpu":
        args.device = "cuda"

    ontology_store = load_ontology_yaml(args.ontology_yaml)

    model = Stage1TeacherModel(
        text_encoder_name=args.text_encoder_name,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )
    _load_teacher_checkpoint(model, args.teacher_checkpoint)

    export_cfg = PrototypeExportConfig(
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    paths = export_stage1_prototype_banks(
        model=model,
        ontology_store=ontology_store,
        output_dir=args.output_dir,
        config=export_cfg,
    )

    print("Stage 1 export completed")
    for key, value in paths.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()

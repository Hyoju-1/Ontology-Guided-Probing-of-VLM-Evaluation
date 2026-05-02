from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

# Ensure "src" imports work when running: python scripts/train_stage1.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.batch import Stage1Collator, Stage1CollatorConfig
from src.data.dataset import (
    Stage1ConceptDataset,
    build_concept_training_lookup,
    build_training_item_splits,
)
from src.losses.contrastive import Stage1Loss
from src.models.teacher import Stage1TeacherModel
from src.ontology.loader import load_ontology_yaml
from src.ontology.schema import ConceptType
from src.trainer.stage1_trainer import Stage1Trainer, Stage1TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1 MVP teacher model")

    parser.add_argument("--ontology-yaml", type=str, required=True, help="Path to ontology YAML")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Output checkpoint path (.pt)")

    parser.add_argument("--text-encoder-name", type=str, default="bert-base-uncased")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-positive-texts", type=int, default=4)
    parser.add_argument("--max-negative-texts", type=int, default=8)

    parser.add_argument("--epochs-per-head", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace) -> Dict[ConceptType, DataLoader]:
    ontology_store = load_ontology_yaml(args.ontology_yaml)

    splits = build_training_item_splits(
        frame_concepts_by_id=ontology_store.frame_concepts_by_id,
        role_concepts_by_id=ontology_store.role_concepts_by_id,
        tuple_concepts_by_id=ontology_store.tuple_concepts_by_id,
    )
    concept_lookup = build_concept_training_lookup(splits)

    collator = Stage1Collator(
        concept_lookup_by_id=concept_lookup,
        config=Stage1CollatorConfig(
            tokenizer_name=args.text_encoder_name,
            max_length=args.max_length,
            max_positive_texts=args.max_positive_texts,
            max_negative_texts=args.max_negative_texts,
        ),
    )

    frame_ds = Stage1ConceptDataset(splits.frame_items)
    role_ds = Stage1ConceptDataset(splits.role_items)
    tuple_ds = Stage1ConceptDataset(splits.tuple_items)

    dataloaders: Dict[ConceptType, DataLoader] = {
        ConceptType.FRAME: DataLoader(
            frame_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=False,
        ),
        ConceptType.ROLE: DataLoader(
            role_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=False,
        ),
        ConceptType.TYPED_TUPLE: DataLoader(
            tuple_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=False,
        ),
    }
    return dataloaders


def main() -> None:
    args = parse_args()

    # Accept common alias used in CLI examples.
    if args.device.lower() == "gpu":
        args.device = "cuda"

    dataloaders = build_dataloaders(args)

    model = Stage1TeacherModel(
        text_encoder_name=args.text_encoder_name,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )
    criterion = Stage1Loss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    trainer = Stage1Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        config=Stage1TrainerConfig(
            device=args.device,
            grad_clip_norm=args.grad_clip_norm,
            seed=args.seed,
        ),
    )

    head_order = [ConceptType.FRAME, ConceptType.ROLE, ConceptType.TYPED_TUPLE]

    for head_type in head_order:
        loader = dataloaders[head_type]
        for epoch in range(1, args.epochs_per_head + 1):
            metrics = trainer.train_epoch(loader, expected_head_type=head_type)
            print(
                f"[train] head={head_type.value} epoch={epoch}/{args.epochs_per_head} "
                f"loss={metrics['train/loss']:.6f} "
                f"pos_sim={metrics['train/mean_positive_similarity']:.6f} "
                f"neg_sim={metrics['train/mean_negative_similarity']:.6f}"
            )

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_args": vars(args),
            "head_order": [h.value for h in head_order],
        },
        checkpoint_path,
    )

    print(f"[done] checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()

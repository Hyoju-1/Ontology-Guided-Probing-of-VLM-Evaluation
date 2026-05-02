from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Allow running this script from repository root without package install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.stage2_dataset import Stage2Collator, Stage2DistillDataset
from src.trainer.stage2_trainer import Stage2Trainer, Stage2TrainerConfig
from src.utils.prototype_bank import load_stage1_mapping_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 fused student model")

    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)

    parser.add_argument("--export-dir", type=str, default="outputs/stage1_export")
    parser.add_argument("--output-checkpoint", type=str, default="outputs/stage2_student.pt")
    parser.add_argument("--save-dir", type=str, default="outputs/stage2")

    parser.add_argument("--ablation-mode", type=str, default="clip_plus_frame")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--fused-dim", type=int, default=512)
    parser.add_argument("--query-dim", type=int, default=256)
    parser.add_argument("--freeze-policy", type=str, default="full_freeze")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tau-clip", type=float, default=0.07)
    parser.add_argument("--tau-distill", type=float, default=0.07)
    parser.add_argument("--lambda-hard-negative", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.0)
    parser.add_argument("--lambda-frame", type=float, default=1.0)
    parser.add_argument("--lambda-tuple", type=float, default=1.0)
    parser.add_argument("--lambda-role", type=float, default=1.0)
    parser.add_argument("--tuple-warmup-epochs", type=int, default=0)
    parser.add_argument("--lambda-tuple-warmup", type=float, default=None)

    parser.add_argument("--use-frame-balanced-sampler", action="store_true")
    parser.add_argument("--frame-sampler-alpha", type=float, default=1.0)
    parser.add_argument("--frame-sampler-replacement", action="store_true")
    parser.add_argument(
        "--frame-loss-weighting",
        type=str,
        default="none",
        choices=["none", "inverse_frequency", "clipped_inverse_frequency"],
    )
    parser.add_argument("--frame-loss-weight-clip-max", type=float, default=5.0)
    parser.add_argument("--frame-loss-weight-eps", type=float, default=1e-6)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-role", action="store_true")

    return parser.parse_args()


def normalize_device(device: str) -> str:
    d = device.lower().strip()
    if d == "gpu":
        return "cuda"
    return d


def build_frame_histogram(samples) -> Dict[int, int]:
    counter = Counter(int(s.frame_label) for s in samples)
    return dict(counter)


def build_frame_loss_weights(
    frame_hist: Dict[int, int],
    num_classes: int,
    mode: str,
    eps: float,
    clip_max: float,
) -> Optional[torch.Tensor]:
    if mode == "none":
        return None

    counts = torch.ones(num_classes, dtype=torch.float32)
    for i in range(num_classes):
        counts[i] = float(frame_hist.get(i, 0))

    inv = 1.0 / torch.clamp(counts, min=eps)
    inv = inv / inv.mean().clamp_min(eps)

    if mode == "clipped_inverse_frequency":
        inv = torch.clamp(inv, max=clip_max)
        inv = inv / inv.mean().clamp_min(eps)

    return inv


def build_frame_balanced_sampler(
    frame_labels: List[int],
    frame_hist: Dict[int, int],
    alpha: float,
    replacement: bool,
) -> Tuple[WeightedRandomSampler, Dict[int, float], torch.Tensor]:
    sample_weights = []
    for y in frame_labels:
        count = max(int(frame_hist.get(int(y), 0)), 1)
        sample_weights.append(float(count) ** (-float(alpha)))

    sample_weights_t = torch.tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights_t,
        num_samples=len(frame_labels),
        replacement=bool(replacement),
    )

    effective_counts = Counter()
    for y, w in zip(frame_labels, sample_weights):
        effective_counts[int(y)] += float(w)
    total_eff = sum(effective_counts.values())
    if total_eff <= 0:
        effective_dist = {k: 0.0 for k in sorted(effective_counts.keys())}
    else:
        effective_dist = {k: float(v / total_eff) for k, v in sorted(effective_counts.items())}

    return sampler, effective_dist, sample_weights_t


def main() -> None:
    args = parse_args()
    device = normalize_device(args.device)

    export_dir = Path(args.export_dir)
    frame_map_path = export_dir / "frame_mapping.json"
    tuple_map_path = export_dir / "tuple_mapping.json"
    role_map_path = export_dir / "role_mapping.json"

    frame_id_to_index = load_stage1_mapping_dict(frame_map_path)
    tuple_id_to_index = load_stage1_mapping_dict(tuple_map_path)

    role_id_to_index = None
    if role_map_path.exists():
        role_id_to_index = load_stage1_mapping_dict(role_map_path)

    train_ds = Stage2DistillDataset(
        jsonl_path=args.train_jsonl,
        frame_id_to_index=frame_id_to_index,
        tuple_id_to_index=tuple_id_to_index,
        role_id_to_index=role_id_to_index,
        clip_model_name=args.clip_model_name,
        require_role=args.require_role,
        check_image_exists=True,
        fail_on_invalid=True,
    )
    val_ds = Stage2DistillDataset(
        jsonl_path=args.val_jsonl,
        frame_id_to_index=frame_id_to_index,
        tuple_id_to_index=tuple_id_to_index,
        role_id_to_index=role_id_to_index,
        clip_model_name=args.clip_model_name,
        require_role=args.require_role,
        check_image_exists=True,
        fail_on_invalid=True,
    )

    print("[Stage2 Contract] train:", json.dumps(asdict(train_ds.contract_report), ensure_ascii=False))
    print("[Stage2 Contract] val  :", json.dumps(asdict(val_ds.contract_report), ensure_ascii=False))

    frame_hist = build_frame_histogram(train_ds.samples)
    num_frame_classes = int(len(frame_id_to_index))

    frame_labels = [int(s.frame_label) for s in train_ds.samples]
    frame_sampler = None
    effective_sampling_dist = {
        k: float(v / max(len(train_ds.samples), 1)) for k, v in sorted(frame_hist.items())
    }
    if args.use_frame_balanced_sampler:
        frame_sampler, effective_sampling_dist, _ = build_frame_balanced_sampler(
            frame_labels=frame_labels,
            frame_hist=frame_hist,
            alpha=args.frame_sampler_alpha,
            replacement=args.frame_sampler_replacement,
        )

    collator = Stage2Collator()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(frame_sampler is None),
        sampler=frame_sampler,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collator,
    )

    cfg = Stage2TrainerConfig(
        seed=args.seed,
        device=device,
        clip_model_name=args.clip_model_name,
        fused_dim=args.fused_dim,
        query_dim=args.query_dim,
        freeze_policy=args.freeze_policy,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tau_clip=args.tau_clip,
        tau_distill=args.tau_distill,
        lambda_hard_negative=args.lambda_hard_negative,
        hard_negative_margin=args.hard_negative_margin,
        lambda_frame=args.lambda_frame,
        lambda_tuple=args.lambda_tuple,
        lambda_role=args.lambda_role,
        frame_loss_weighting=args.frame_loss_weighting,
        frame_loss_weight_clip_max=args.frame_loss_weight_clip_max,
        ablation_mode=args.ablation_mode,
        export_dir=str(export_dir),
    )

    trainer = Stage2Trainer(config=cfg)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    frame_loss_weights = build_frame_loss_weights(
        frame_hist=frame_hist,
        num_classes=num_frame_classes,
        mode=args.frame_loss_weighting,
        eps=args.frame_loss_weight_eps,
        clip_max=args.frame_loss_weight_clip_max,
    )
    trainer.set_frame_loss_class_weights(frame_loss_weights)

    frame_hist_path = save_dir / "frame_label_histogram.json"
    frame_hist_path.write_text(
        json.dumps(
            {
                "num_samples": len(train_ds.samples),
                "num_frame_classes": num_frame_classes,
                "frame_label_histogram": {str(k): int(v) for k, v in sorted(frame_hist.items())},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    frame_sampling_path = save_dir / "frame_effective_sampling_distribution.json"
    frame_sampling_path.write_text(
        json.dumps(
            {
                "use_frame_balanced_sampler": bool(args.use_frame_balanced_sampler),
                "frame_sampler_alpha": float(args.frame_sampler_alpha),
                "frame_sampler_replacement": bool(args.frame_sampler_replacement),
                "frame_effective_sampling_distribution": {
                    str(k): float(v) for k, v in sorted(effective_sampling_dist.items())
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    frame_loss_weight_path = save_dir / "frame_loss_weights.json"
    if frame_loss_weights is None:
        frame_loss_payload = {
            "frame_loss_weighting": "none",
            "frame_loss_weight_clip_max": float(args.frame_loss_weight_clip_max),
            "frame_loss_weights": None,
        }
    else:
        frame_loss_payload = {
            "frame_loss_weighting": args.frame_loss_weighting,
            "frame_loss_weight_clip_max": float(args.frame_loss_weight_clip_max),
            "frame_loss_weights": [float(x) for x in frame_loss_weights.tolist()],
        }
    frame_loss_weight_path.write_text(
        json.dumps(frame_loss_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[Frame Balance] histogram: {frame_hist_path}")
    print(f"[Frame Balance] sampling : {frame_sampling_path}")
    print(f"[Frame Balance] weights  : {frame_loss_weight_path}")

    best_val = float("inf")
    best_epoch = -1
    history = []
    frame_coverage_history = []

    for epoch in range(1, args.epochs + 1):
        if args.tuple_warmup_epochs > 0 and args.lambda_tuple_warmup is not None:
            if epoch <= args.tuple_warmup_epochs:
                trainer.cfg.lambda_tuple = args.lambda_tuple_warmup
            else:
                trainer.cfg.lambda_tuple = args.lambda_tuple

        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        frame_counts = trainer.last_train_frame_counts
        if frame_counts is None:
            per_epoch_cov = {
                "epoch": epoch,
                "frame_counts": [],
                "frame_num_classes_seen": 0,
                "frame_num_classes_total": num_frame_classes,
                "frame_coverage_ratio": 0.0,
            }
        else:
            counts_list = [int(x) for x in frame_counts.tolist()]
            seen = int(sum(1 for x in counts_list if x > 0))
            per_epoch_cov = {
                "epoch": epoch,
                "frame_counts": counts_list,
                "frame_num_classes_seen": seen,
                "frame_num_classes_total": int(len(counts_list)),
                "frame_coverage_ratio": float(seen / max(len(counts_list), 1)),
            }
        frame_coverage_history.append(per_epoch_cov)

        train_line = " ".join([f"{k}={v:.6f}" for k, v in train_metrics.items() if isinstance(v, float)])
        val_line = " ".join([f"{k}={v:.6f}" for k, v in val_metrics.items() if isinstance(v, float)])

        print(f"[Epoch {epoch}] TRAIN {train_line}")
        print(f"[Epoch {epoch}] VAL   {val_line}")
        print(
            "[Epoch {}] LAMBDAS frame={:.4f} tuple={:.4f} role={:.4f}".format(
                epoch,
                float(trainer.cfg.lambda_frame),
                float(trainer.cfg.lambda_tuple),
                float(trainer.cfg.lambda_role),
            )
        )
        print(
            "[Epoch {}] FRAME_COVERAGE seen={}/{} ratio={:.4f}".format(
                epoch,
                int(per_epoch_cov["frame_num_classes_seen"]),
                int(per_epoch_cov["frame_num_classes_total"]),
                float(per_epoch_cov["frame_coverage_ratio"]),
            )
        )

        val_loss = float(val_metrics["val/loss_total"])
        trainer.save_checkpoint(
            path=save_dir / f"stage2_epoch_{epoch}.pt",
            epoch=epoch,
            extra={"train": train_metrics, "val": val_metrics},
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            trainer.save_checkpoint(
                path=args.output_checkpoint,
                epoch=epoch,
                extra={"train": train_metrics, "val": val_metrics, "best": True},
            )

    history_path = save_dir / "metrics_history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    frame_cov_path = save_dir / "frame_coverage_per_epoch.json"
    frame_cov_path.write_text(json.dumps(frame_coverage_history, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Stage 2 training completed. best_epoch={best_epoch}, best_val_loss={best_val:.6f}")
    print(f"Best checkpoint: {args.output_checkpoint}")
    print(f"History: {history_path}")
    print(f"Frame coverage: {frame_cov_path}")


if __name__ == "__main__":
    main()

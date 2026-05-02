from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.losses.fused_distill import clip_alignment_loss, clip_hard_negative_loss, distill_cross_entropy_loss
from src.models.student_stage2 import Stage2StudentModel
from src.utils.prototype_bank import load_stage1_export_banks


AblationMode = Literal[
    "clip_only",
    "clip_plus_frame",
    "clip_plus_frame_tuple",
    "clip_plus_frame_tuple_role",
]


@dataclass(slots=True)
class Stage2TrainerConfig:
    # Repro / device
    seed: int = 42
    device: str = "cuda"

    # Model
    clip_model_name: str = "openai/clip-vit-base-patch32"
    fused_dim: int = 512
    query_dim: int = 256
    use_role_head: bool = False
    freeze_policy: str = "full_freeze"  # full_freeze | partial_unfreeze | full_finetune
    dropout: float = 0.1

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Loss weights / temperatures
    tau_clip: float = 0.07
    tau_distill: float = 0.07
    lambda_clip: float = 1.0
    lambda_hard_negative: float = 0.0
    hard_negative_margin: float = 0.0
    lambda_frame: float = 1.0
    lambda_tuple: float = 1.0
    lambda_role: float = 1.0
    frame_loss_weighting: str = "none"  # none | inverse_frequency | clipped_inverse_frequency
    frame_loss_weight_clip_max: float = 5.0

    # Ablation control
    ablation_mode: AblationMode = "clip_plus_frame_tuple"

    # Prototype export files
    export_dir: str = "outputs/stage1_export"
    frame_proto_file: str = "frame_prototypes.pt"
    tuple_proto_file: str = "tuple_prototypes.pt"
    role_proto_file: str = "role_prototypes.pt"
    frame_map_file: str = "frame_mapping.json"
    tuple_map_file: str = "tuple_mapping.json"
    role_map_file: str = "role_mapping.json"


class Stage2Trainer:
    """Minimal Stage 2 trainer for fused multimodal distillation.

    Supports:
      - CLIP alignment loss
      - Frame distillation loss (optional)
      - Tuple distillation loss (optional)
      - Role distillation loss (optional)
      - Ablation modes
      - Validation and checkpointing
    """

    def __init__(self, config: Stage2TrainerConfig) -> None:
        self.cfg = config
        self._set_seed(self.cfg.seed)

        self.device = torch.device("cuda" if self.cfg.device.lower() == "gpu" else self.cfg.device)

        self._apply_ablation_mode()

        self.model = Stage2StudentModel(
            clip_model_name=self.cfg.clip_model_name,
            fused_dim=self.cfg.fused_dim,
            query_dim=self.cfg.query_dim,
            use_role_head=self.cfg.use_role_head,
            freeze_policy=self.cfg.freeze_policy,
            dropout=self.cfg.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        banks = load_stage1_export_banks(
            export_dir=self.cfg.export_dir,
            device=self.device,
            use_role_head=self.cfg.use_role_head,
            frame_proto_file=self.cfg.frame_proto_file,
            tuple_proto_file=self.cfg.tuple_proto_file,
            role_proto_file=self.cfg.role_proto_file,
            frame_map_file=self.cfg.frame_map_file,
            tuple_map_file=self.cfg.tuple_map_file,
            role_map_file=self.cfg.role_map_file,
        )
        self.frame_bank = banks.frame_bank  # [Nf, Dq]
        self.tuple_bank = banks.tuple_bank  # [Nt, Dq]
        self.role_bank = banks.role_bank  # [Nr, Dq] or None

        self.frame_id_to_index = banks.frame_id_to_index
        self.tuple_id_to_index = banks.tuple_id_to_index
        self.role_id_to_index = banks.role_id_to_index

        self.frame_loss_class_weights: Optional[torch.Tensor] = None
        self.last_train_frame_counts: Optional[torch.Tensor] = None

    def set_frame_loss_class_weights(self, class_weights: Optional[torch.Tensor]) -> None:
        if class_weights is None:
            self.frame_loss_class_weights = None
            return

        if class_weights.dim() != 1:
            raise ValueError("frame class weights must be 1D tensor")
        if int(class_weights.numel()) != int(self.frame_bank.shape[0]):
            raise ValueError(
                f"frame class weights length mismatch: expected {int(self.frame_bank.shape[0])}, got {int(class_weights.numel())}"
            )
        self.frame_loss_class_weights = class_weights.detach().to(self.device, dtype=torch.float32)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()

        agg = self._new_meter_dict(prefix="train")
        frame_counts = torch.zeros(int(self.frame_bank.shape[0]), dtype=torch.long, device=self.device)

        for batch in dataloader:
            batch = self._move_batch_to_device(batch)

            out = self.model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )

            losses, metrics = self._compute_losses_and_metrics(out, batch)

            labels = batch.get("frame_labels")
            if labels is not None:
                valid_labels = labels[labels >= 0]
                if valid_labels.numel() > 0:
                    frame_counts += torch.bincount(valid_labels, minlength=frame_counts.numel())

            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()

            if self.cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

            self.optimizer.step()

            bsz = int(batch["input_ids"].shape[0])
            self._accumulate(agg, losses, metrics, bsz)

        self.last_train_frame_counts = frame_counts.detach().to("cpu")

        seen = int((frame_counts > 0).sum().item())
        total_classes = int(frame_counts.numel())
        agg["frame_num_classes_seen"] = float(seen)
        agg["frame_num_classes_total"] = float(total_classes)
        agg["frame_coverage_ratio"] = float(seen / total_classes) if total_classes > 0 else 0.0

        return self._finalize_meter_dict(agg)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()

        agg = self._new_meter_dict(prefix="val")

        for batch in dataloader:
            batch = self._move_batch_to_device(batch)

            out = self.model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )

            losses, metrics = self._compute_losses_and_metrics(out, batch)
            bsz = int(batch["input_ids"].shape[0])
            self._accumulate(agg, losses, metrics, bsz)

        return self._finalize_meter_dict(agg)

    def save_checkpoint(self, path: str | Path, epoch: int, extra: Optional[Dict] = None) -> None:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
        }
        if extra:
            payload["extra"] = extra

        torch.save(payload, ckpt_path)

    # ---------------------------------------------------------------------
    # Internal: losses / metrics
    # ---------------------------------------------------------------------
    def _compute_losses_and_metrics(self, out, batch: Dict[str, torch.Tensor]):
        loss_clip = clip_alignment_loss(out.image_embeddings, out.text_embeddings, tau=self.cfg.tau_clip)
        loss_hard_negative = torch.tensor(0.0, device=self.device)

        loss_frame = torch.tensor(0.0, device=self.device)
        loss_tuple = torch.tensor(0.0, device=self.device)
        loss_role = torch.tensor(0.0, device=self.device)

        frame_top1 = torch.tensor(0.0, device=self.device)
        tuple_top1 = torch.tensor(0.0, device=self.device)
        role_top1 = torch.tensor(0.0, device=self.device)
        role_valid_ratio = torch.tensor(0.0, device=self.device)

        # frame distill
        if self.cfg.lambda_frame > 0:
            frame_res = distill_cross_entropy_loss(
                query=out.frame_query,
                bank=self.frame_bank,
                labels=batch["frame_labels"],
                tau=self.cfg.tau_distill,
                ignore_index=-1,
                class_weights=self.frame_loss_class_weights,
            )
            loss_frame = frame_res.loss
            frame_top1 = frame_res.top1

        # tuple distill
        if self.cfg.lambda_tuple > 0:
            tuple_res = distill_cross_entropy_loss(
                query=out.tuple_query,
                bank=self.tuple_bank,
                labels=batch["tuple_labels"],
                tau=self.cfg.tau_distill,
                ignore_index=-1,
            )
            loss_tuple = tuple_res.loss
            tuple_top1 = tuple_res.top1

        # role distill (optional)
        if self.cfg.use_role_head and self.cfg.lambda_role > 0 and out.role_query is not None and self.role_bank is not None:
            role_labels = batch.get("role_labels")
            if role_labels is None:
                # no labels in this batch
                loss_role = torch.tensor(0.0, device=self.device)
                role_top1 = torch.tensor(0.0, device=self.device)
                role_valid_ratio = torch.tensor(0.0, device=self.device)
            else:
                role_res = distill_cross_entropy_loss(
                    query=out.role_query,
                    bank=self.role_bank,
                    labels=role_labels,
                    tau=self.cfg.tau_distill,
                    ignore_index=-1,
                )
                loss_role = role_res.loss
                role_top1 = role_res.top1
                role_valid_ratio = role_res.valid_ratio

        if self.cfg.lambda_hard_negative > 0:
            loss_hard_negative = clip_hard_negative_loss(
                image_emb=out.image_embeddings,
                text_emb=out.text_embeddings,
                tau=self.cfg.tau_clip,
                margin=self.cfg.hard_negative_margin,
            )

        total = (
            self.cfg.lambda_clip * loss_clip
            + self.cfg.lambda_hard_negative * loss_hard_negative
            + self.cfg.lambda_frame * loss_frame
            + self.cfg.lambda_tuple * loss_tuple
            + self.cfg.lambda_role * loss_role
        )

        # simple behavior metric: in-batch CLIP retrieval top1
        retrieval_i2t, retrieval_t2i = self._clip_retrieval_metrics(out.image_embeddings, out.text_embeddings, self.cfg.tau_clip)

        losses = {
            "total": total,
            "clip": loss_clip,
            "hard_negative": loss_hard_negative,
            "frame": loss_frame,
            "tuple": loss_tuple,
            "role": loss_role,
        }
        metrics = {
            "retrieval_i2t_top1": retrieval_i2t,
            "retrieval_t2i_top1": retrieval_t2i,
            "frame_top1": frame_top1,
            "tuple_top1": tuple_top1,
            "role_top1": role_top1,
            "role_valid_ratio": role_valid_ratio,
        }
        return losses, metrics

    @staticmethod
    def _clip_retrieval_metrics(image_emb: torch.Tensor, text_emb: torch.Tensor, tau: float):
        logits_i2t = (image_emb @ text_emb.t()) / tau
        logits_t2i = logits_i2t.t()
        labels = torch.arange(logits_i2t.size(0), device=logits_i2t.device)

        i2t_top1 = (logits_i2t.argmax(dim=1) == labels).float().mean()
        t2i_top1 = (logits_t2i.argmax(dim=1) == labels).float().mean()
        return i2t_top1, t2i_top1

    # ---------------------------------------------------------------------
    # Internal: metering / utils
    # ---------------------------------------------------------------------
    def _new_meter_dict(self, prefix: str) -> Dict[str, float]:
        return {
            "prefix": prefix,
            "count": 0.0,
            "loss_total": 0.0,
            "loss_clip": 0.0,
            "loss_hard_negative": 0.0,
            "loss_frame": 0.0,
            "loss_tuple": 0.0,
            "loss_role": 0.0,
            "retrieval_i2t_top1": 0.0,
            "retrieval_t2i_top1": 0.0,
            "frame_top1": 0.0,
            "tuple_top1": 0.0,
            "role_top1": 0.0,
            "role_valid_ratio": 0.0,
            "frame_num_classes_seen": 0.0,
            "frame_num_classes_total": 0.0,
            "frame_coverage_ratio": 0.0,
        }

    @staticmethod
    def _accumulate(agg: Dict[str, float], losses: Dict[str, torch.Tensor], metrics: Dict[str, torch.Tensor], bsz: int) -> None:
        agg["count"] += bsz
        agg["loss_total"] += float(losses["total"].item()) * bsz
        agg["loss_clip"] += float(losses["clip"].item()) * bsz
        agg["loss_hard_negative"] += float(losses["hard_negative"].item()) * bsz
        agg["loss_frame"] += float(losses["frame"].item()) * bsz
        agg["loss_tuple"] += float(losses["tuple"].item()) * bsz
        agg["loss_role"] += float(losses["role"].item()) * bsz

        agg["retrieval_i2t_top1"] += float(metrics["retrieval_i2t_top1"].item()) * bsz
        agg["retrieval_t2i_top1"] += float(metrics["retrieval_t2i_top1"].item()) * bsz
        agg["frame_top1"] += float(metrics["frame_top1"].item()) * bsz
        agg["tuple_top1"] += float(metrics["tuple_top1"].item()) * bsz
        agg["role_top1"] += float(metrics["role_top1"].item()) * bsz
        agg["role_valid_ratio"] += float(metrics["role_valid_ratio"].item()) * bsz

    @staticmethod
    def _finalize_meter_dict(agg: Dict[str, float]) -> Dict[str, float]:
        count = agg["count"]
        if count <= 0:
            raise ValueError("No samples processed")

        p = agg["prefix"]
        return {
            f"{p}/loss_total": agg["loss_total"] / count,
            f"{p}/loss_clip": agg["loss_clip"] / count,
            f"{p}/loss_hard_negative": agg["loss_hard_negative"] / count,
            f"{p}/loss_frame": agg["loss_frame"] / count,
            f"{p}/loss_tuple": agg["loss_tuple"] / count,
            f"{p}/loss_role": agg["loss_role"] / count,
            f"{p}/retrieval_i2t_top1": agg["retrieval_i2t_top1"] / count,
            f"{p}/retrieval_t2i_top1": agg["retrieval_t2i_top1"] / count,
            f"{p}/frame_top1": agg["frame_top1"] / count,
            f"{p}/tuple_top1": agg["tuple_top1"] / count,
            f"{p}/role_top1": agg["role_top1"] / count,
            f"{p}/role_valid_ratio": agg["role_valid_ratio"] / count,
            f"{p}/frame_num_classes_seen": agg["frame_num_classes_seen"],
            f"{p}/frame_num_classes_total": agg["frame_num_classes_total"],
            f"{p}/frame_coverage_ratio": agg["frame_coverage_ratio"],
            f"{p}/num_samples": count,
        }

    def _apply_ablation_mode(self) -> None:
        mode = self.cfg.ablation_mode

        if mode == "clip_only":
            self.cfg.lambda_frame = 0.0
            self.cfg.lambda_tuple = 0.0
            self.cfg.lambda_role = 0.0
            self.cfg.use_role_head = False
        elif mode == "clip_plus_frame":
            if self.cfg.lambda_frame <= 0:
                self.cfg.lambda_frame = 1.0
            self.cfg.lambda_tuple = 0.0
            self.cfg.lambda_role = 0.0
            self.cfg.use_role_head = False
        elif mode == "clip_plus_frame_tuple":
            if self.cfg.lambda_frame <= 0:
                self.cfg.lambda_frame = 1.0
            if self.cfg.lambda_tuple <= 0:
                self.cfg.lambda_tuple = 1.0
            self.cfg.lambda_role = 0.0
            self.cfg.use_role_head = False
        elif mode == "clip_plus_frame_tuple_role":
            if self.cfg.lambda_frame <= 0:
                self.cfg.lambda_frame = 1.0
            if self.cfg.lambda_tuple <= 0:
                self.cfg.lambda_tuple = 1.0
            if self.cfg.lambda_role <= 0:
                self.cfg.lambda_role = 1.0
            self.cfg.use_role_head = True
        else:
            raise ValueError(
                f"Unknown ablation_mode '{mode}'. "
                "Expected one of: clip_only, clip_plus_frame, clip_plus_frame_tuple, clip_plus_frame_tuple_role"
            )

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

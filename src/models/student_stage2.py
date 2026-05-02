from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


FreezePolicy = Literal["full_freeze", "partial_unfreeze", "full_finetune"]


@dataclass(slots=True)
class Stage2StudentOutput:
    """Outputs for Stage 2 fused multimodal distillation.

    Shapes:
      - image_embeddings: [B, D_clip]
      - text_embeddings: [B, D_clip]
      - fused_embeddings: [B, D_fused]
      - frame_query: [B, D_query]
      - tuple_query: [B, D_query]
      - role_query: [B, D_query] or None
    """

    image_embeddings: torch.Tensor
    text_embeddings: torch.Tensor
    fused_embeddings: torch.Tensor
    frame_query: torch.Tensor
    tuple_query: torch.Tensor
    role_query: Optional[torch.Tensor] = None


class FusionMLP(nn.Module):
    """Fusion module: fused = normalize(MLP(concat(image_emb, text_emb)))."""

    def __init__(self, input_dim: int, fused_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # image_emb: [B, D_clip], text_emb: [B, D_clip]
        fused_in = torch.cat([image_emb, text_emb], dim=-1)  # [B, 2*D_clip]
        fused = self.net(fused_in)  # [B, D_fused]
        fused = F.normalize(fused, p=2, dim=-1)
        return fused


class DistillProjectionHead(nn.Module):
    """Lightweight projection head from fused embedding to teacher query space."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D_fused]
        q = self.net(x)  # [B, D_query]
        q = F.normalize(q, p=2, dim=-1)
        return q


class CLIPDualEncoder(nn.Module):
    """Shared CLIP-style image/text dual encoder with configurable freeze policy."""

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze_policy: FreezePolicy = "full_freeze",
    ) -> None:
        super().__init__()
        self.backbone = CLIPModel.from_pretrained(clip_model_name)

        # CLIP projection dimension used by get_image_features/get_text_features
        self.clip_embed_dim = int(self.backbone.config.projection_dim)

        self.apply_freeze_policy(freeze_policy)

    def apply_freeze_policy(self, freeze_policy: FreezePolicy) -> None:
        if freeze_policy == "full_finetune":
            for p in self.backbone.parameters():
                p.requires_grad = True
            return

        # Start from fully frozen for both full_freeze and partial_unfreeze
        for p in self.backbone.parameters():
            p.requires_grad = False

        if freeze_policy == "partial_unfreeze":
            # Minimal partial unfreeze: allow CLIP projection layers to adapt.
            for p in self.backbone.visual_projection.parameters():
                p.requires_grad = True
            for p in self.backbone.text_projection.parameters():
                p.requires_grad = True
        elif freeze_policy != "full_freeze":
            raise ValueError(
                f"Unknown freeze_policy '{freeze_policy}'. "
                "Expected one of: full_freeze, partial_unfreeze, full_finetune."
            )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image/text into CLIP embedding space.

        Args:
          pixel_values: [B, 3, H, W]
          input_ids: [B, L]
          attention_mask: [B, L] or None

        Returns:
          image_embeddings: [B, D_clip]
          text_embeddings: [B, D_clip]
        """
        image_out = self.backbone.get_image_features(pixel_values=pixel_values)
        text_out = self.backbone.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        image_embeddings = self._coerce_feature_tensor(
            feat_output=image_out,
            is_image=True,
        )
        text_embeddings = self._coerce_feature_tensor(
            feat_output=text_out,
            is_image=False,
        )

        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        return image_embeddings, text_embeddings

    def _coerce_feature_tensor(self, feat_output, is_image: bool) -> torch.Tensor:
        """Handle transformers API differences for CLIP feature outputs."""
        if isinstance(feat_output, torch.Tensor):
            return feat_output

        # Some transformers versions return BaseModelOutputWithPooling from get_*_features.
        pooled = getattr(feat_output, "pooler_output", None)
        if pooled is not None:
            # If pooled output is already in CLIP projection space, use as-is.
            if pooled.size(-1) == self.clip_embed_dim:
                return pooled

            if is_image:
                if pooled.size(-1) == self.backbone.visual_projection.in_features:
                    return self.backbone.visual_projection(pooled)
            else:
                if pooled.size(-1) == self.backbone.text_projection.in_features:
                    return self.backbone.text_projection(pooled)

            kind = "image" if is_image else "text"
            raise ValueError(
                f"Unexpected CLIP {kind} pooled dim={pooled.size(-1)}; "
                f"expected one of ({self.clip_embed_dim}, "
                f"{self.backbone.visual_projection.in_features if is_image else self.backbone.text_projection.in_features})."
            )

        if isinstance(feat_output, tuple) and feat_output:
            first = feat_output[0]
            if isinstance(first, torch.Tensor):
                return first

        kind = "image" if is_image else "text"
        raise TypeError(f"Unsupported CLIP {kind} feature output type: {type(feat_output)}")


class Stage2StudentModel(nn.Module):
    """Minimal Stage 2 student model for fused multimodal distillation.

    Components:
      1) shared CLIP-style image/text backbone
      2) fused representation module
      3) distillation projection heads for frame/tuple/(optional role)

    This model is designed for:
      - CLIP-style image-text alignment loss on image_embeddings/text_embeddings
      - fused distillation to frozen Stage 1 prototype banks via query heads
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        fused_dim: int = 512,
        query_dim: int = 256,
        use_role_head: bool = False,
        freeze_policy: FreezePolicy = "full_freeze",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.use_role_head = bool(use_role_head)

        self.dual_encoder = CLIPDualEncoder(
            clip_model_name=clip_model_name,
            freeze_policy=freeze_policy,
        )

        d_clip = self.dual_encoder.clip_embed_dim

        self.fusion = FusionMLP(
            input_dim=2 * d_clip,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        self.frame_head = DistillProjectionHead(
            in_dim=fused_dim,
            out_dim=query_dim,
            dropout=dropout,
        )
        self.tuple_head = DistillProjectionHead(
            in_dim=fused_dim,
            out_dim=query_dim,
            dropout=dropout,
        )

        self.role_head: Optional[DistillProjectionHead]
        if self.use_role_head:
            self.role_head = DistillProjectionHead(
                in_dim=fused_dim,
                out_dim=query_dim,
                dropout=dropout,
            )
        else:
            self.role_head = None

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Stage2StudentOutput:
        """Forward pass with explicit tensor shapes.

        Args:
          pixel_values: [B, 3, H, W]
          input_ids: [B, L]
          attention_mask: [B, L] or None

        Returns:
          Stage2StudentOutput with:
            - image_embeddings: [B, D_clip]
            - text_embeddings: [B, D_clip]
            - fused_embeddings: [B, D_fused]
            - frame_query: [B, D_query]
            - tuple_query: [B, D_query]
            - role_query: [B, D_query] or None
        """
        image_embeddings, text_embeddings = self.dual_encoder(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        fused_embeddings = self.fusion(image_embeddings, text_embeddings)

        frame_query = self.frame_head(fused_embeddings)
        tuple_query = self.tuple_head(fused_embeddings)

        role_query: Optional[torch.Tensor] = None
        if self.role_head is not None:
            role_query = self.role_head(fused_embeddings)

        return Stage2StudentOutput(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            fused_embeddings=fused_embeddings,
            frame_query=frame_query,
            tuple_query=tuple_query,
            role_query=role_query,
        )

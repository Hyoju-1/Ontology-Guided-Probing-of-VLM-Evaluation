from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.ontology.schema import ConceptTrainingItem, Stage1Batch


@dataclass(slots=True)
class Stage1CollatorConfig:
    tokenizer_name: str = "bert-base-uncased"
    max_length: int = 128
    max_positive_texts: int = 4
    max_negative_texts: int = 8


class Stage1Collator:
    """Collator for Stage 1 concept contrastive batches.

    Constraints:
    - Input batch must be homogeneous by concept_type.
    - Negative texts are resolved from concept_lookup_by_id.
    """

    def __init__(
        self,
        concept_lookup_by_id: Mapping[str, ConceptTrainingItem],
        config: Stage1CollatorConfig | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.concept_lookup_by_id: Mapping[str, ConceptTrainingItem] = concept_lookup_by_id
        self.config = config or Stage1CollatorConfig()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.config.tokenizer_name)

    def __call__(self, items: Sequence[ConceptTrainingItem]) -> Stage1Batch:
        if not items:
            raise ValueError("Stage1Collator received an empty batch")

        self._validate_homogeneous_types(items)

        concept_ids = [item.concept_id for item in items]
        concept_types = [item.concept_type for item in items]

        anchor_texts = [self._build_anchor_text(item) for item in items]

        positive_text_rows: List[List[str]] = [self._collect_positive_texts(item) for item in items]
        negative_id_rows: List[List[str]] = [self._collect_negative_ids(item) for item in items]
        negative_text_rows: List[List[str]] = [self._negative_ids_to_texts(ids) for ids in negative_id_rows]

        padded_positive_text_rows = self._pad_2d_text_rows(
            positive_text_rows,
            pad_value=anchor_texts,
            target_len=self.config.max_positive_texts,
        )
        padded_negative_text_rows = self._pad_2d_text_rows(
            negative_text_rows,
            pad_value=anchor_texts,
            target_len=self.config.max_negative_texts,
        )
        padded_negative_id_rows = self._pad_2d_id_rows(
            negative_id_rows,
            target_len=self.config.max_negative_texts,
        )

        anchor_tokens = self._tokenize(anchor_texts)  # [B, L]

        positive_flat = [txt for row in padded_positive_text_rows for txt in row]
        positive_tokens_flat = self._tokenize(positive_flat)

        negative_flat = [txt for row in padded_negative_text_rows for txt in row]
        negative_tokens_flat = self._tokenize(negative_flat)

        batch_size = len(items)
        num_pos = len(padded_positive_text_rows[0])
        num_neg = len(padded_negative_text_rows[0])
        max_len = int(anchor_tokens["input_ids"].shape[1])

        positive_input_ids = positive_tokens_flat["input_ids"].view(batch_size, num_pos, max_len)
        positive_attention_mask = positive_tokens_flat["attention_mask"].view(batch_size, num_pos, max_len)

        negative_input_ids = negative_tokens_flat["input_ids"].view(batch_size, num_neg, max_len)
        negative_attention_mask = negative_tokens_flat["attention_mask"].view(batch_size, num_neg, max_len)

        # MVP: uniform negative weights.
        negative_weights = torch.ones((batch_size, num_neg), dtype=torch.float32)

        positive_source_ids = [
            [f"{item.concept_id}::pos_{idx}" for idx in range(num_pos)]
            for item in items
        ]

        return Stage1Batch(
            concept_ids=concept_ids,
            concept_types=concept_types,
            anchor_input_ids=anchor_tokens["input_ids"].to(dtype=torch.long),
            anchor_attention_mask=anchor_tokens["attention_mask"].to(dtype=torch.long),
            positive_input_ids=positive_input_ids.to(dtype=torch.long),
            positive_attention_mask=positive_attention_mask.to(dtype=torch.long),
            negative_input_ids=negative_input_ids.to(dtype=torch.long),
            negative_attention_mask=negative_attention_mask.to(dtype=torch.long),
            positive_source_ids=positive_source_ids,
            negative_source_ids=padded_negative_id_rows,
            negative_weights=negative_weights,
        )

    def _validate_homogeneous_types(self, items: Sequence[ConceptTrainingItem]) -> None:
        head_type = items[0].concept_type
        for item in items:
            if item.concept_type != head_type:
                raise ValueError(
                    "Batch must be homogeneous by concept head type. "
                    f"Found {head_type} and {item.concept_type}."
                )

    def _build_anchor_text(self, item: ConceptTrainingItem) -> str:
        if item.concept_name:
            return f"{item.concept_name}: {item.definition}".strip()
        return item.definition.strip()

    def _collect_positive_texts(self, item: ConceptTrainingItem) -> List[str]:
        positives: List[str] = []

        for text in item.paraphrases:
            text = text.strip()
            if text:
                positives.append(text)

        for text in item.equivalent_descriptions:
            text = text.strip()
            if text:
                positives.append(text)

        # If no explicit positive text exists, use anchor text itself.
        if not positives:
            positives = [self._build_anchor_text(item)]

        return positives[: self.config.max_positive_texts]

    def _collect_negative_ids(self, item: ConceptTrainingItem) -> List[str]:
        ids: List[str] = []

        if item.inverse_concept_id:
            ids.append(item.inverse_concept_id)

        ids.extend(item.confusing_concept_ids)
        ids.extend(item.negative_concept_ids)

        # Unique + remove self-id
        seen = set()
        unique_ids: List[str] = []
        for cid in ids:
            if not cid or cid == item.concept_id:
                continue
            if cid not in seen:
                seen.add(cid)
                unique_ids.append(cid)

        if not unique_ids:
            raise ValueError(
                f"Concept '{item.concept_id}' has no negative concept IDs. "
                "Provide inverse/confusing/negative_concept_ids."
            )

        return unique_ids[: self.config.max_negative_texts]

    def _negative_ids_to_texts(self, negative_ids: Sequence[str]) -> List[str]:
        texts: List[str] = []
        for cid in negative_ids:
            negative_item = self.concept_lookup_by_id.get(cid)
            if negative_item is None:
                raise KeyError(f"Negative concept ID not found in lookup: {cid}")
            texts.append(self._build_anchor_text(negative_item))
        return texts

    def _pad_2d_text_rows(
        self,
        rows: Sequence[Sequence[str]],
        pad_value: Sequence[str],
        target_len: int,
    ) -> List[List[str]]:
        padded: List[List[str]] = []

        if target_len <= 0:
            raise ValueError("target_len must be > 0")

        for idx, row in enumerate(rows):
            clipped = list(row)[:target_len]
            if not clipped:
                clipped = [pad_value[idx]]

            if len(clipped) < target_len:
                clipped = clipped + [clipped[-1]] * (target_len - len(clipped))

            padded.append(clipped)

        return padded

    def _pad_2d_id_rows(
        self,
        rows: Sequence[Sequence[str]],
        target_len: int,
    ) -> List[List[str]]:
        padded: List[List[str]] = []

        if target_len <= 0:
            raise ValueError("target_len must be > 0")

        for row in rows:
            clipped = list(row)[:target_len]
            if len(clipped) < target_len:
                pad_token = clipped[-1] if clipped else ""
                clipped = clipped + [pad_token] * (target_len - len(clipped))
            padded.append(clipped)

        return padded

    def _tokenize(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            list(texts),
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

#!/usr/bin/env bash
set -euo pipefail

TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-bert-base-uncased}"
CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-openai/clip-vit-base-patch32}"

python scripts/quick_test.py

python scripts/train_stage1.py \
  --ontology-yaml data/ontology_minimal.yaml \
  --checkpoint-path outputs/stage1_demo.pt \
  --text-encoder-name "${TEXT_ENCODER_NAME}" \
  --embedding-dim 256 \
  --batch-size 4 \
  --epochs-per-head 1 \
  --num-workers 0 \
  --device cpu

python scripts/export_teacher.py \
  --ontology-yaml data/ontology_minimal.yaml \
  --teacher-checkpoint outputs/stage1_demo.pt \
  --output-dir outputs/stage1_export_demo \
  --text-encoder-name "${TEXT_ENCODER_NAME}" \
  --embedding-dim 256 \
  --batch-size 16 \
  --device cpu

python scripts/train_stage2.py \
  --train-jsonl data/stage2_samples.example.jsonl \
  --val-jsonl data/stage2_samples.example.jsonl \
  --export-dir outputs/stage1_export_demo \
  --output-checkpoint outputs/stage2_demo/best.pt \
  --save-dir outputs/stage2_demo \
  --ablation-mode clip_plus_frame_tuple \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --clip-model-name "${CLIP_MODEL_NAME}" \
  --freeze-policy full_freeze \
  --device cpu

# Anonymous Submission Package

This folder is a reduced, anonymized package prepared for code submission.

Included:
- Minimal Stage 1 teacher training code
- Stage 1 prototype export code
- Minimal Stage 2 student training code
- Small ontology file
- Small example JSONL dataset with relative image paths
- Two example images for a self-contained smoke run

Not included:
- Local absolute paths from the original workspace
- Large research artifacts, drafts, figures, and logs
- External dataset-specific JSONL files that referenced private local directories

## Folder Layout

- `scripts/quick_test.py`: small in-memory Stage 1 sanity check
- `scripts/train_stage1.py`: Stage 1 teacher training
- `scripts/export_teacher.py`: export prototype banks from a trained Stage 1 checkpoint
- `scripts/train_stage2.py`: Stage 2 student training
- `src/`: core library code required by the scripts
- `data/ontology_minimal.yaml`: small ontology used for the demo
- `data/stage2_samples.example.jsonl`: small example Stage 2 dataset
- `data/images/`: images referenced by the example JSONL

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

If needed, install a system-compatible PyTorch build first and then re-run the command above.

## Quick Start

1. Run the Stage 1 sanity check:

```bash
python scripts/quick_test.py
```

2. Train a small Stage 1 teacher on CPU:

```bash
python scripts/train_stage1.py \
  --ontology-yaml data/ontology_minimal.yaml \
  --checkpoint-path outputs/stage1_demo.pt \
  --text-encoder-name bert-base-uncased \
  --embedding-dim 256 \
  --batch-size 4 \
  --epochs-per-head 1 \
  --num-workers 0 \
  --device cpu
```

3. Export Stage 1 prototype banks:

```bash
python scripts/export_teacher.py \
  --ontology-yaml data/ontology_minimal.yaml \
  --teacher-checkpoint outputs/stage1_demo.pt \
  --output-dir outputs/stage1_export_demo \
  --text-encoder-name bert-base-uncased \
  --embedding-dim 256 \
  --batch-size 16 \
  --device cpu
```

4. Run a small Stage 2 demo:

```bash
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
  --freeze-policy full_freeze \
  --device cpu
```

You can also run the whole sequence with:

```bash
bash run_demo.sh
```

## Notes

- The first run may download pretrained Hugging Face model weights for BERT and CLIP.
- If you already have local Hugging Face model directories, you can pass them instead of model names.
- Example: `--text-encoder-name /path/to/local/bert-base-uncased`
- Example: `--clip-model-name /path/to/local/clip-vit-base-patch32`
- The demo script also supports environment variables:
- `TEXT_ENCODER_NAME=/path/to/local/bert-base-uncased`
- `CLIP_MODEL_NAME=/path/to/local/clip-vit-base-patch32`
- The example Stage 2 JSONL uses relative image paths so the package can run without user-specific directories.
- This package is intended as a minimal runnable submission bundle rather than the full research workspace.

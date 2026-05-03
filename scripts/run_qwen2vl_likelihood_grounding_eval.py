#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "Qwen2-VL-2B-Instruct"
DEFAULT_BENCHMARK_JSON = (
    ROOT
    / "output"
    / "vg20k_coco20k_stratified_scored_4k"
    / "semantic_violation_benchmark_vg20k_coco20k_stratified_extension_scored_onto_img_factorized.json"
)
DEFAULT_OUTDIR = ROOT / "output" / "qwen2vl_likelihood_mixed_4k"

CAPTION_PROMPT = "Describe this image with one short caption."
PHRASE_PROMPT = "Answer with one short grounded phrase from the image."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Qwen2-VL likelihood-based grounding evaluation on factorized benchmarks.")
    p.add_argument("--benchmark-json", default=str(DEFAULT_BENCHMARK_JSON))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--device-map", default="auto")
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--limit-samples", type=int, default=0)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--skip-analysis", action="store_true")
    p.add_argument("--analysis-output-prefix", default="qwen2vl_likelihood_mixed_4k_editaware")
    p.add_argument(
        "--include-recipes",
        nargs="*",
        default=["visual_only", "baseline_relation_only", "relphrase_spatialbind"],
    )
    p.add_argument("--semantic-confidence-threshold", type=float, default=0.6)
    p.add_argument("--lambda-delta", type=float, default=0.35)
    p.add_argument("--lambda-sem", type=float, default=0.3)
    p.add_argument("--lambda-ground", type=float, default=0.2)
    return p.parse_args()


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if math.isnan(out) or math.isinf(out):
        return float("nan")
    return out


def is_finite(value: Any) -> bool:
    return math.isfinite(safe_float(value))


def normalize_space(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_image_path(image_path: Optional[str]) -> Optional[Path]:
    if not image_path:
        return None
    path = Path(image_path)
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    return path


def extract_samples(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return payload["samples"]
    if isinstance(payload, list):
        return payload
    raise TypeError(f"Unsupported payload type: {type(payload)!r}")


def build_prompt_text(processor: AutoProcessor, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    size = max(1, int(size))
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


def score_answer_batch(
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    prompts: Sequence[str],
    answers: Sequence[str],
) -> List[Dict[str, Any]]:
    if not prompts:
        return []
    prefix_texts = [build_prompt_text(processor, prompt) for prompt in prompts]
    full_texts = [prefix + answer for prefix, answer in zip(prefix_texts, answers)]
    images = [image] * len(full_texts)
    prefix_inputs = processor(text=prefix_texts, images=images, return_tensors="pt", padding=True)
    full_inputs = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
    full_inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in full_inputs.items()}
    with torch.inference_mode():
        outputs = model(**full_inputs)
    logits = outputs.logits
    logprobs = torch.log_softmax(logits, dim=-1)
    input_ids = full_inputs["input_ids"]
    attention_mask = full_inputs["attention_mask"]
    results: List[Dict[str, Any]] = []
    for idx in range(input_ids.shape[0]):
        start = int(prefix_lens[idx])
        end = int(attention_mask[idx].sum().item())
        if start <= 0 or end <= start:
            results.append({"mean_logprob": float("nan"), "sum_logprob": float("nan"), "token_count": 0})
            continue
        target_ids = input_ids[idx, start:end]
        pred_lp = logprobs[idx, start - 1:end - 1, :]
        token_lp = pred_lp.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        results.append(
            {
                "mean_logprob": float(token_lp.mean().item()),
                "sum_logprob": float(token_lp.sum().item()),
                "token_count": int(target_ids.numel()),
            }
        )
    return results


def mean_or_nan(values: Sequence[float]) -> float:
    vals = [safe_float(v) for v in values if is_finite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def get_semantic_components(candidate: Dict[str, Any]) -> Dict[str, Any]:
    breakdown = candidate.get("ontology_image_component_breakdown") or {}
    semantic = breakdown.get("semantic_components") or {}
    if isinstance(semantic, dict):
        return semantic
    return {}


def build_relation_phrase(candidate: Dict[str, Any], semantic: Dict[str, Any]) -> str:
    subj = normalize_space(semantic.get("subject") or (candidate.get("triplet") or {}).get("subject"))
    pred = normalize_space(semantic.get("predicate") or (candidate.get("triplet") or {}).get("predicate")).replace("_", " ")
    obj = normalize_space(semantic.get("object") or (candidate.get("triplet") or {}).get("object"))
    if subj and pred and obj:
        return normalize_space(f"{subj} {pred} {obj}")
    return normalize_space(candidate.get("text") or candidate.get("caption"))


def build_spatial_phrases(candidate: Dict[str, Any], semantic: Dict[str, Any]) -> List[str]:
    phrases: List[str] = []
    for item in semantic.get("spatial_relations") or []:
        phrase = normalize_space(item.get("phrase"))
        if phrase:
            phrases.append(phrase)
            continue
        subj = normalize_space(item.get("subject") or semantic.get("subject"))
        spat = normalize_space(item.get("spatial")).replace("_", " ")
        obj = normalize_space(item.get("object") or semantic.get("object"))
        if subj and spat and obj:
            phrases.append(normalize_space(f"{subj} {spat} {obj}"))
    return phrases[:5]


def run_analysis(
    *,
    benchmark_json: Path,
    outdir: Path,
    output_prefix: str,
    include_recipes: Sequence[str],
    semantic_confidence_threshold: float,
    lambda_delta: float,
    lambda_sem: float,
    lambda_ground: float,
) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "edit_aware_contrastive_grounding_analysis.py"),
        "--benchmark-json",
        str(benchmark_json),
        "--outdir",
        str(outdir),
        "--output-prefix",
        output_prefix,
        "--calibration",
        "zscore",
        "--gate-mode",
        "combined",
        "--clip-margin-thresholds",
        "0.0",
        "--lambda-deltas",
        str(lambda_delta),
        "--include-recipes",
        *list(include_recipes),
        "--semantic-confidence-threshold",
        str(semantic_confidence_threshold),
        "--lambda-delta",
        str(lambda_delta),
        "--lambda-sem",
        str(lambda_sem),
        "--lambda-ground",
        str(lambda_ground),
        "--w-relation",
        "1.0",
        "--w-attribute",
        "0.5",
        "--w-spatial",
        "0.8",
        "--w-count",
        "0.5",
        "--w-subj",
        "0.5",
        "--w-obj",
        "0.5",
        "--w-attr-bind",
        "1.0",
        "--w-spatial-bind",
        "1.0",
        "--w-rel-phrase",
        "1.0",
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    return outdir / f"{output_prefix}.json"


def analysis_rows(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = analysis.get("results") or []
    rows: List[Dict[str, Any]] = []
    if isinstance(results, list):
        for row in results:
            if isinstance(row, dict):
                rows.append(row)
        return rows
    if isinstance(results, dict):
        for recipe, payload in results.items():
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics") or {}
            if not isinstance(metrics, dict):
                continue
            overall = ((metrics.get("overall") or {}).get("all") or {}) if isinstance(metrics.get("overall"), dict) else {}
            row = {"recipe": str(recipe), "metrics": metrics}
            if isinstance(overall, dict):
                row.update(overall)
            rows.append(row)
    return rows


def best_result(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rows = [row for row in analysis_rows(analysis) if is_finite(row.get("pairwise_accuracy"))]
    if not rows:
        return None
    return max(rows, key=lambda row: safe_float(row.get("pairwise_accuracy")))


def find_recipe(analysis: Dict[str, Any], recipe_substring: str) -> Optional[Dict[str, Any]]:
    for row in analysis_rows(analysis):
        recipe = str(row.get("recipe") or "")
        if recipe_substring in recipe:
            return row
        if recipe_substring == "baseline_relation_only" and "relation_only_baseline" in recipe:
            return row
    return None


def write_summary(path: Path, payload: Dict[str, Any]) -> None:
    base = payload.get("visual_only") or {}
    baseline = payload.get("baseline_relation_only") or {}
    best = payload.get("best") or {}
    lines = [
        "# Qwen2-VL Likelihood Grounding Experiment",
        "",
        f"- Benchmark: `{payload.get('benchmark')}`",
        f"- Model: `{payload.get('model_name')}`",
        f"- Samples used: `{payload.get('num_samples_used')}`",
        f"- Output factorized JSON: `{payload.get('factorized_json')}`",
        f"- Analysis JSON: `{payload.get('analysis_json')}`",
        "",
        "## Results",
        "",
        "| Setting | Recipe | Pairwise | Rescue | AUROC | Valid pairwise |",
        "|---|---|---:|---:|---:|---:|",
        f"| Visual-only | `{base.get('recipe', 'NA')}` | {safe_float(base.get('pairwise_accuracy')):.4f} | {safe_float(base.get('rescue_rate')):.4f} | {safe_float(base.get('auroc')):.4f} | {base.get('num_valid_pairwise', 'NA')} |",
        f"| Semantic baseline | `{baseline.get('recipe', 'NA')}` | {safe_float(baseline.get('pairwise_accuracy')):.4f} | {safe_float(baseline.get('rescue_rate')):.4f} | {safe_float(baseline.get('auroc')):.4f} | {baseline.get('num_valid_pairwise', 'NA')} |",
        f"| Best selective | `{best.get('recipe', 'NA')}` | {safe_float(best.get('pairwise_accuracy')):.4f} | {safe_float(best.get('rescue_rate')):.4f} | {safe_float(best.get('auroc')):.4f} | {best.get('num_valid_pairwise', 'NA')} |",
        "",
        "## Deltas",
        "",
        f"- Visual-only -> Semantic baseline pairwise: `{safe_float(baseline.get('pairwise_accuracy')) - safe_float(base.get('pairwise_accuracy')):+.4f}`",
        f"- Semantic baseline -> Best selective pairwise: `{safe_float(best.get('pairwise_accuracy')) - safe_float(baseline.get('pairwise_accuracy')):+.4f}`",
        f"- Visual-only -> Best selective pairwise: `{safe_float(best.get('pairwise_accuracy')) - safe_float(base.get('pairwise_accuracy')):+.4f}`",
        "",
        "## Diagnostics",
        "",
        f"- Mean base caption logprob: `{safe_float(payload.get('mean_base_logprob')):.4f}`",
        f"- Mean relation phrase logprob: `{safe_float(payload.get('mean_relation_phrase_logprob')):.4f}`",
        f"- Mean spatial phrase logprob: `{safe_float(payload.get('mean_spatial_phrase_logprob')):.4f}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark_json = Path(args.benchmark_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    factorized_out = outdir / f"{benchmark_json.stem}_qwen2vl_likelihood_factorized.json"
    progress_json = outdir / "progress.json"

    payload = load_json(benchmark_json)
    samples = extract_samples(payload)
    if args.limit_samples > 0:
        samples = samples[: args.limit_samples]
        if isinstance(payload, dict):
            payload["samples"] = samples
        else:
            payload = samples

    start_index = 0
    if factorized_out.exists():
        try:
            resumed_payload = load_json(factorized_out)
            resumed_samples = extract_samples(resumed_payload)
            if len(resumed_samples) == len(samples):
                payload = resumed_payload
                samples = resumed_samples
                if progress_json.exists():
                    progress_payload = load_json(progress_json)
                    start_index = max(0, min(len(samples), int(progress_payload.get("sample_index_completed") or 0)))
                else:
                    start_index = 0
                print(f"[resume] resuming from sample {start_index}/{len(samples)}", flush=True)
        except Exception as exc:
            print(f"[resume] ignored existing checkpoint due to load error: {exc}", flush=True)

    print(f"[setup] loading model from {args.model_dir}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    base_values: List[float] = []
    rel_values: List[float] = []
    spatial_values: List[float] = []
    started = time.time()
    num_scored_candidates = 0

    for sample in samples[:start_index]:
        for candidate in [sample.get("positive", {})] + list(sample.get("negatives") or []):
            scores = candidate.get("scores") or {}
            base_score = safe_float(scores.get("qwen2vl_caption_logprob_mean"))
            rel_score = safe_float(scores.get("qwen2vl_relation_phrase_logprob_mean"))
            spatial_score = safe_float(scores.get("qwen2vl_spatial_phrase_logprob_mean"))
            if is_finite(base_score):
                base_values.append(base_score)
            if is_finite(rel_score):
                rel_values.append(rel_score)
            if is_finite(spatial_score):
                spatial_values.append(spatial_score)
            if any(
                key in scores
                for key in (
                    "qwen2vl_caption_logprob_mean",
                    "qwen2vl_relation_phrase_logprob_mean",
                    "qwen2vl_spatial_phrase_logprob_mean",
                )
            ):
                num_scored_candidates += 1

    for sample_index, sample in enumerate(samples[start_index:], start=start_index + 1):
        image_path = resolve_image_path(sample.get("image_path"))
        if image_path is None or not image_path.exists():
            continue
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
            requests: List[Dict[str, Any]] = []
            candidates = [sample.get("positive", {})] + list(sample.get("negatives") or [])
            for cand_index, candidate in enumerate(candidates):
                semantic = get_semantic_components(candidate)
                caption = normalize_space(candidate.get("text") or candidate.get("caption"))
                if caption:
                    requests.append(
                        {
                            "kind": "base",
                            "candidate_index": cand_index,
                            "prompt": CAPTION_PROMPT,
                            "answer": caption,
                        }
                    )
                rel_phrase = build_relation_phrase(candidate, semantic)
                if rel_phrase:
                    requests.append(
                        {
                            "kind": "rel_phrase",
                            "candidate_index": cand_index,
                            "prompt": PHRASE_PROMPT,
                            "answer": rel_phrase,
                        }
                    )
                for spatial_idx, spatial_phrase in enumerate(build_spatial_phrases(candidate, semantic)):
                    requests.append(
                        {
                            "kind": "spatial_phrase",
                            "candidate_index": cand_index,
                            "spatial_index": spatial_idx,
                            "prompt": PHRASE_PROMPT,
                            "answer": spatial_phrase,
                        }
                    )

            scored_requests: List[Dict[str, Any]] = []
            for chunk in chunked(requests, args.batch_size):
                prompts = [item["prompt"] for item in chunk]
                answers = [item["answer"] for item in chunk]
                chunk_scores = score_answer_batch(model, processor, image, prompts, answers)
                for item, score in zip(chunk, chunk_scores):
                    merged = dict(item)
                    merged.update(score)
                    scored_requests.append(merged)

        per_candidate: Dict[int, Dict[str, Any]] = {}
        for item in scored_requests:
            entry = per_candidate.setdefault(item["candidate_index"], {"spatial_values": []})
            if item["kind"] == "base":
                entry["base"] = item["mean_logprob"]
            elif item["kind"] == "rel_phrase":
                entry["rel_phrase"] = item["mean_logprob"]
            elif item["kind"] == "spatial_phrase" and is_finite(item["mean_logprob"]):
                entry["spatial_values"].append(item["mean_logprob"])

        for cand_index, candidate in enumerate([sample.get("positive", {})] + list(sample.get("negatives") or [])):
            scores = candidate.setdefault("scores", {})
            result = per_candidate.get(cand_index, {})
            base_score = safe_float(result.get("base"))
            rel_score = safe_float(result.get("rel_phrase"))
            spatial_score = mean_or_nan(result.get("spatial_values") or [])
            scores["clip"] = base_score
            scores["phrase_binding_support"] = rel_score
            scores["spatial_visual_support"] = spatial_score
            scores["grounding_support_source"] = "qwen2vl_likelihood"
            scores["qwen2vl_caption_logprob_mean"] = base_score
            scores["qwen2vl_relation_phrase_logprob_mean"] = rel_score
            scores["qwen2vl_spatial_phrase_logprob_mean"] = spatial_score
            if is_finite(base_score):
                base_values.append(base_score)
            if is_finite(rel_score):
                rel_values.append(rel_score)
            if is_finite(spatial_score):
                spatial_values.append(spatial_score)
            num_scored_candidates += 1

        if sample_index % max(1, args.save_every) == 0 or sample_index == len(samples):
            if isinstance(payload, dict):
                payload["samples"] = samples
                payload.setdefault("metadata", {})["qwen2vl_likelihood_eval"] = {
                    "model_name": "Qwen/Qwen2-VL-2B-Instruct",
                    "model_dir": str(args.model_dir),
                    "benchmark_json": str(benchmark_json),
                    "sample_index_completed": sample_index,
                    "num_samples_total": len(samples),
                    "num_scored_candidates": num_scored_candidates,
                    "updated_at_unix": int(time.time()),
                }
            save_json(factorized_out, payload)
            save_json(
                progress_json,
                {
                    "sample_index_completed": sample_index,
                    "num_samples_total": len(samples),
                    "num_scored_candidates": num_scored_candidates,
                    "elapsed_sec": time.time() - started,
                    "factorized_out": str(factorized_out),
                },
            )
            print(f"[progress] samples {sample_index}/{len(samples)}", flush=True)

    analysis_json = None
    analysis = {}
    if not args.skip_analysis:
        analysis_json = run_analysis(
            benchmark_json=factorized_out,
            outdir=outdir,
            output_prefix=args.analysis_output_prefix,
            include_recipes=args.include_recipes,
            semantic_confidence_threshold=args.semantic_confidence_threshold,
            lambda_delta=args.lambda_delta,
            lambda_sem=args.lambda_sem,
            lambda_ground=args.lambda_ground,
        )
        analysis = load_json(analysis_json)

    summary = {
        "benchmark": benchmark_json.stem,
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "factorized_json": str(factorized_out),
        "analysis_json": str(analysis_json) if analysis_json else "",
        "num_samples_used": len(samples),
        "num_scored_candidates": num_scored_candidates,
        "mean_base_logprob": mean_or_nan(base_values),
        "mean_relation_phrase_logprob": mean_or_nan(rel_values),
        "mean_spatial_phrase_logprob": mean_or_nan(spatial_values),
        "visual_only": find_recipe(analysis, "visual_only"),
        "baseline_relation_only": find_recipe(analysis, "baseline_relation_only"),
        "best": best_result(analysis),
    }
    summary_json = outdir / "qwen2vl_likelihood_summary.json"
    summary_md = outdir / "qwen2vl_likelihood_summary.md"
    save_json(summary_json, summary)
    write_summary(summary_md, summary)
    print(f"[done] summary_json={summary_json}", flush=True)
    print(f"[done] summary_md={summary_md}", flush=True)


if __name__ == "__main__":
    main()

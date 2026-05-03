#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import (
    BridgeTowerForImageAndTextRetrieval,
    BridgeTowerImageProcessor,
    BridgeTowerProcessor,
    RobertaTokenizerFast,
)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
MODEL_DIR = PROJECT_ROOT / "models" / "bridgetower-base-itm-mlm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BridgeTower retrieval-style selective correction.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-native", action="store_true")
    parser.add_argument("--skip-execute", action="store_true")
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "cross_model_openclip_bridgetower"))
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def inspect_snapshot(path: Path) -> Dict[str, Any]:
    files = sorted(p.name for p in path.iterdir()) if path.exists() else []
    return {
        "exists": path.exists(),
        "path": str(path),
        "files": files,
        "has_config": (path / "config.json").exists(),
        "has_preprocessor": (path / "preprocessor_config.json").exists(),
        "has_pytorch_bin": (path / "pytorch_model.bin").exists(),
    }


def attempt_load(path: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "processor_loaded": False,
        "model_loaded": False,
        "model_type": None,
        "sanity_score_finite": False,
        "sanity_logits": None,
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }
    try:
        image_processor = BridgeTowerImageProcessor.from_pretrained(path, local_files_only=True)
        tokenizer = RobertaTokenizerFast.from_pretrained(path, local_files_only=True)
        processor = BridgeTowerProcessor(image_processor, tokenizer)
        model = BridgeTowerForImageAndTextRetrieval.from_pretrained(path, local_files_only=True)
        report["processor_loaded"] = True
        report["model_loaded"] = True
        report["model_type"] = type(model).__name__

        image_root = ROOT / "output" / "winoground_materialized_images"
        image_path = sorted(image_root.glob("*.png"))[0]
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            images=[image, image],
            text=["a person is standing", "a dog is sleeping"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        model.eval()
        with torch.no_grad():
            logits = model(**inputs, return_dict=True).logits
        report["sanity_logits"] = logits.detach().cpu().tolist()
        report["sanity_score_finite"] = bool(torch.isfinite(logits).all().item())
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
    return report


def build_jobs() -> List[Dict[str, Any]]:
    return [
        {
            "benchmark": "mixed_vg20k_coco20k_4k",
            "input_json": ROOT / "output" / "vg20k_coco20k_stratified_scored_4k" / "semantic_violation_benchmark_vg20k_coco20k_stratified_extension_scored_onto_img_factorized.json",
            "include_recipes": ["visual_only", "baseline_relation_only", "relphrase_spatialbind", "full_semantic_grounding"],
        },
        {
            "benchmark": "coco5k",
            "input_json": ROOT / "output" / "semantic_violation_benchmark_coco5k_scored_components_scored_onto_img_factorized.json",
            "include_recipes": ["visual_only", "baseline_relation_only", "relphrase_spatialbind", "full_semantic_grounding"],
        },
        {
            "benchmark": "winoground_exploded",
            "input_json": ROOT / "output" / "winoground_exploded_refined_scored" / "semantic_violation_benchmark_winoground_exploded_scored_onto_img_factorized.json",
            "include_recipes": ["visual_only", "baseline_relation_only", "spatialbind_only", "relphrase_spatialbind", "full_semantic_grounding"],
            "native_benchmark_json": ROOT / "output" / "semantic_violation_benchmark_winoground_native.json",
        },
    ]


def run_command(cmd: List[str]) -> None:
    print("RUN:", " ".join(str(item) for item in cmd))
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    subprocess.run(cmd, check=True, env=env)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def parse_editaware_json(path: Path) -> Dict[str, Dict[str, float]]:
    payload = load_json(path)
    out: Dict[str, Dict[str, float]] = {}
    for recipe, result in payload.get("results", {}).items():
        overall = result.get("metrics", {}).get("overall", {}).get("all", {})
        out[recipe] = {
            "pairwise": safe_float(overall.get("pairwise_accuracy")),
            "rescue": safe_float(overall.get("rescue_rate")),
            "AUROC": safe_float(overall.get("auroc")),
        }
    return out


def parse_native_json(path: Path) -> Dict[str, Dict[str, float]]:
    payload = load_json(path)
    name_map = {
        "visual_only": "visual_only",
        "baseline_relation_only": "combined_semantic_relation_only_baseline@+0.000",
        "relphrase_spatialbind": "edit_contrastive_relphrase_spatialbind@combined@+0.000@0.35",
        "full_semantic_grounding": "edit_contrastive_full_semantic_grounding@combined@+0.000@0.35",
        "spatialbind_only": "edit_contrastive_spatialbind_only@combined@+0.000@0.35",
    }
    out: Dict[str, Dict[str, float]] = {}
    for row in payload.get("rows", []):
        recipe = name_map.get(row.get("recipe"))
        if recipe:
            out[recipe] = {
                "native_text": safe_float(row.get("text_score")),
                "native_image": safe_float(row.get("image_score")),
                "native_group": safe_float(row.get("group_score")),
            }
    return out


def summarize_results(manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    recipe_order = [
        "visual_only",
        "combined_semantic_relation_only_baseline@+0.000",
        "edit_contrastive_relphrase_spatialbind@combined@+0.000@0.35",
        "edit_contrastive_full_semantic_grounding@combined@+0.000@0.35",
        "edit_contrastive_spatialbind_only@combined@+0.000@0.35",
    ]
    rows: List[Dict[str, Any]] = []
    for job in manifest:
        benchmark = job["benchmark"]
        parsed = parse_editaware_json(Path(job["analysis_json"]))
        native = parse_native_json(Path(job["native_json"])) if job.get("native_json") else {}
        for recipe in recipe_order:
            if recipe == "edit_contrastive_spatialbind_only@combined@+0.000@0.35" and benchmark != "winoground_exploded":
                continue
            metrics = parsed.get(recipe, {})
            native_metrics = native.get(recipe, {}) if benchmark == "winoground_exploded" else {}
            rows.append(
                {
                    "model_name": "bridgetower-base-itm-mlm",
                    "model_family": "BridgeTower",
                    "benchmark": benchmark,
                    "recipe": recipe,
                    "pairwise": metrics.get("pairwise", float("nan")),
                    "rescue": metrics.get("rescue", float("nan")),
                    "AUROC": metrics.get("AUROC", float("nan")),
                    "native_text": native_metrics.get("native_text", float("nan")),
                    "native_image": native_metrics.get("native_image", float("nan")),
                    "native_group": native_metrics.get("native_group", float("nan")),
                    "analysis_json": job["analysis_json"],
                }
            )
    return rows


def build_loading_report(snapshot: Dict[str, Any], load_report: Dict[str, Any]) -> str:
    lines = [
        "# BridgeTower Loading Report",
        "",
        f"- model dir: `{snapshot['path']}`",
        f"- exists: `{snapshot['exists']}`",
        f"- has `config.json`: `{snapshot['has_config']}`",
        f"- has `preprocessor_config.json`: `{snapshot['has_preprocessor']}`",
        f"- has `pytorch_model.bin`: `{snapshot['has_pytorch_bin']}`",
        f"- processor loaded: `{load_report['processor_loaded']}`",
        f"- model loaded: `{load_report['model_loaded']}`",
        f"- model type: `{load_report['model_type']}`",
        f"- sanity score finite: `{load_report['sanity_score_finite']}`",
        f"- sanity logits: `{load_report['sanity_logits']}`",
    ]
    if load_report["error_type"]:
        lines.extend(["", "## Error", "", f"- error type: `{load_report['error_type']}`", f"- message: `{load_report['error_message']}`"])
    return "\n".join(lines) + "\n"


def build_score_interface_report() -> str:
    return "\n".join(
        [
            "# BridgeTower Score Interface Report",
            "",
            "- model: `bridgetower-base-itm-mlm`",
            "- loader: `BridgeTowerImageProcessor` + `RobertaTokenizerFast` + `BridgeTowerProcessor` + `BridgeTowerForImageAndTextRetrieval`",
            "- base score used in the selective-correction pipeline:",
            "  - `BridgeTowerForImageAndTextRetrieval(...).logits` returns a 2-class image-text matching logit",
            "  - the scalar score is defined as `match_logit - mismatch_logit`",
            "- interpretation:",
            "  - this is an ITM margin, not a CLIP cosine similarity",
            "  - the same scalar is used both for base ranking and for image-phrase support under the current lightweight compatibility layer",
            "- `visual_only` means BridgeTower ITM margin only, with no ontology-guided correction.",
        ]
    ) + "\n"


def build_failure_report(stage: str, message: str) -> str:
    return "\n".join(
        [
            "# BridgeTower Failure Report",
            "",
            f"- failure stage: `{stage}`",
            f"- message: `{message}`",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    snapshot = inspect_snapshot(MODEL_DIR)
    load_report = attempt_load(MODEL_DIR)
    write_json(outdir / "bridgetower_probe.json", {"snapshot": snapshot, "load_report": load_report})
    (outdir / "bridgetower_loading_report.md").write_text(build_loading_report(snapshot, load_report), encoding="utf-8")
    (outdir / "bridgetower_score_interface_report.md").write_text(build_score_interface_report(), encoding="utf-8")
    if not load_report["model_loaded"]:
        (outdir / "bridgetower_failure_report.md").write_text(
            build_failure_report("load", load_report.get("error_message") or "unknown"),
            encoding="utf-8",
        )
        return

    model_slug = "bridgetower_base_itm_mlm"
    model_dir = outdir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    ontology_json = ROOT / "output" / "generalized_ontology" / "generalized_semantic_ontology.json"
    predicate_frames = ROOT / "output" / "generalized_ontology" / "generalized_predicate_frames.json"
    compat_npz = ROOT / "output" / "generalized_ontology" / "generalized_compatibility_tables.npz"

    manifest: List[Dict[str, Any]] = []
    requested = set(args.benchmarks or [])
    for job in build_jobs():
        if requested and job["benchmark"] not in requested:
            continue
        bench = job["benchmark"]
        bench_dir = model_dir / bench
        bench_dir.mkdir(parents=True, exist_ok=True)
        rescored_json = bench_dir / f"{bench}_bridgetower_scored.json"
        factorized_dir = bench_dir / "factorized"
        factorized_json = factorized_dir / f"{rescored_json.stem}_scored_onto_img_factorized.json"
        analysis_prefix = f"{model_slug}_{bench}_editaware"
        commands = [
            [
                sys.executable,
                str(ROOT / "score_semantic_violation_benchmark.py"),
                "--benchmark-json", str(job["input_json"]),
                "--output-json", str(rescored_json),
                "--scorers", "clip",
                "--clip-checkpoint", str(MODEL_DIR),
                "--device", args.device,
                "--images-root", str(PROJECT_ROOT),
            ],
            [
                sys.executable,
                str(ROOT / "score_image_conditioned_semantic_plausibility.py"),
                "--benchmark-json", str(rescored_json),
                "--ontology-json", str(ontology_json),
                "--predicate-frames-json", str(predicate_frames),
                "--compatibility-npz", str(compat_npz),
                "--outdir", str(factorized_dir),
                "--precompute-visual-support",
                "--image-batch-size", "8",
                "--text-batch-size", "16",
                "--save-coverage-summary",
                "--export-for-gate-features",
                "--strict-triplet-fallback",
                "--use-clip-visual-support",
                "--clip-checkpoint", str(MODEL_DIR),
                "--images-root", str(PROJECT_ROOT),
                "--device", args.device,
            ],
            [
                sys.executable,
                str(ROOT / "edit_aware_contrastive_grounding_analysis.py"),
                "--benchmark-json", str(factorized_json),
                "--outdir", str(outdir),
                "--output-prefix", analysis_prefix,
                "--calibration", "zscore",
                "--gate-mode", "combined",
                "--clip-margin-threshold", "0.0",
                "--lambda-delta", "0.35",
                "--include-recipes", *job["include_recipes"],
            ],
        ]
        native_prefix = ""
        if args.run_native and job.get("native_benchmark_json"):
            native_prefix = f"{model_slug}_{bench}_native"
            commands.append(
                [
                    sys.executable,
                    str(ROOT / "evaluate_winoground_native_2x2.py"),
                    "--benchmark-json", str(job["native_benchmark_json"]),
                    "--scored-exploded-json", str(factorized_json),
                    "--outdir", str(outdir),
                    "--output-prefix", native_prefix,
                    "--calibration", "zscore",
                    "--gate-mode", "combined",
                    "--clip-margin-threshold", "0.0",
                    "--lambda-delta", "0.35",
                    "--include-recipes",
                    "visual_only", "baseline_relation_only", "spatialbind_only", "relphrase_spatialbind", "full_semantic_grounding",
                    "--enable-native-delta",
                    "--delta-mode", "relphrase_spatialbind",
                ]
            )
        manifest.append(
            {
                "benchmark": bench,
                "input_json": str(job["input_json"]),
                "rescored_json": str(rescored_json),
                "factorized_json": str(factorized_json),
                "analysis_json": str(outdir / f"{analysis_prefix}.json"),
                "native_json": str(outdir / f"{native_prefix}.json") if native_prefix else "",
            }
        )
        if not args.skip_execute:
            for cmd in commands:
                run_command(cmd)

    write_json(outdir / "bridgetower_run_manifest.json", manifest)
    if args.skip_execute:
        return
    rows = summarize_results(manifest)
    write_json(outdir / "bridgetower_results.json", rows)
    with (outdir / "bridgetower_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

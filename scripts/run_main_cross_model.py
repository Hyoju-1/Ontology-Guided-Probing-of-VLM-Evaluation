#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
OUTDIR = ROOT / "output" / "cross_model_suite"
L14_PATH = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-large-patch14" / "snapshots" / "32bd64288804d66eefd0ccbe215aa642df71cc41"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a lightweight cross-model suite on existing benchmarks.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-execute", action="store_true")
    p.add_argument("--run-native", action="store_true")
    return p.parse_args()


def build_jobs() -> List[Dict[str, object]]:
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
    print("RUN:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    model_slug = "clip_vitl14"
    model_dir = OUTDIR / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    ontology_json = ROOT / "output" / "generalized_ontology" / "generalized_semantic_ontology.json"
    predicate_frames = ROOT / "output" / "generalized_ontology" / "generalized_predicate_frames.json"
    compat_npz = ROOT / "output" / "generalized_ontology" / "generalized_compatibility_tables.npz"

    job_manifest: List[Dict[str, object]] = []
    for job in build_jobs():
        bench = job["benchmark"]
        bench_dir = model_dir / bench
        bench_dir.mkdir(parents=True, exist_ok=True)
        rescored_json = bench_dir / f"{bench}_clip_only_scored.json"
        factorized_dir = bench_dir / "factorized"
        analysis_prefix = f"{model_slug}_{bench}_editaware"
        factorized_json = factorized_dir / f"{rescored_json.stem}_scored_onto_img_factorized.json"
        commands = [
            [
                sys.executable,
                str(ROOT / "score_semantic_violation_benchmark.py"),
                "--benchmark-json",
                str(job["input_json"]),
                "--output-json",
                str(rescored_json),
                "--scorers",
                "clip",
                "--clip-checkpoint",
                str(L14_PATH),
                "--device",
                args.device,
                "--images-root",
                str(PROJECT_ROOT),
            ],
            [
                sys.executable,
                str(ROOT / "score_image_conditioned_semantic_plausibility.py"),
                "--benchmark-json",
                str(rescored_json),
                "--ontology-json",
                str(ontology_json),
                "--predicate-frames-json",
                str(predicate_frames),
                "--compatibility-npz",
                str(compat_npz),
                "--outdir",
                str(factorized_dir),
                "--precompute-visual-support",
                "--image-batch-size",
                "16",
                "--text-batch-size",
                "64",
                "--save-coverage-summary",
                "--export-for-gate-features",
                "--strict-triplet-fallback",
                "--use-clip-visual-support",
                "--clip-checkpoint",
                str(L14_PATH),
                "--images-root",
                str(PROJECT_ROOT),
                "--device",
                args.device,
            ],
            [
                sys.executable,
                str(ROOT / "edit_aware_contrastive_grounding_analysis.py"),
                "--benchmark-json",
                str(factorized_json),
                "--outdir",
                str(OUTDIR),
                "--output-prefix",
                analysis_prefix,
                "--calibration",
                "zscore",
                "--gate-mode",
                "combined",
                "--clip-margin-threshold",
                "0.0",
                "--lambda-delta",
                "0.35",
                "--include-recipes",
                *job["include_recipes"],
            ],
        ]
        native_prefix = None
        if args.run_native and job.get("native_benchmark_json"):
            native_prefix = f"{model_slug}_{bench}_native"
            commands.append(
                [
                    sys.executable,
                    str(ROOT / "evaluate_winoground_native_2x2.py"),
                    "--benchmark-json",
                    str(job["native_benchmark_json"]),
                    "--scored-exploded-json",
                    str(factorized_json),
                    "--outdir",
                    str(OUTDIR),
                    "--output-prefix",
                    native_prefix,
                    "--calibration",
                    "zscore",
                    "--gate-mode",
                    "combined",
                    "--clip-margin-threshold",
                    "0.0",
                    "--lambda-delta",
                    "0.35",
                    "--include-recipes",
                    "visual_only",
                    "baseline_relation_only",
                    "spatialbind_only",
                    "relphrase_spatialbind",
                    "full_semantic_grounding",
                    "--enable-native-delta",
                    "--delta-mode",
                    "relphrase_spatialbind",
                ]
            )

        job_manifest.append(
            {
                "model_slug": model_slug,
                "benchmark": bench,
                "input_json": str(job["input_json"]),
                "rescored_json": str(rescored_json),
                "factorized_json": str(factorized_json),
                "analysis_json": str(OUTDIR / f"{analysis_prefix}.json"),
                "native_json": str(OUTDIR / f"{native_prefix}.json") if native_prefix else "",
                "commands": commands,
            }
        )
        if not args.skip_execute:
            for cmd in commands:
                run_command(cmd)

    (OUTDIR / "run_manifest.json").write_text(json.dumps(job_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTDIR / "run_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_slug", "benchmark", "input_json", "rescored_json", "factorized_json", "analysis_json", "native_json"])
        writer.writeheader()
        for row in job_manifest:
            writer.writerow({k: row[k] for k in writer.fieldnames})


if __name__ == "__main__":
    main()

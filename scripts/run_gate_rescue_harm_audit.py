#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import edit_aware_contrastive_grounding_analysis as edit_analysis


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "output" / "gate_rescue_harm_audit"

RECIPES = {
    "relphrase_spatialbind": "edit_contrastive_relphrase_spatialbind@combined@+0.000@0.35",
    "full_semantic_grounding": "edit_contrastive_full_semantic_grounding@combined@+0.000@0.35",
}

DATASETS = [
    {
        "model": "clip_vitl14",
        "benchmark": "mixed_vg20k_coco20k_4k",
        "path": ROOT / "output" / "cross_model_suite" / "clip_vitl14" / "mixed_vg20k_coco20k_4k" / "factorized" / "mixed_vg20k_coco20k_4k_clip_only_scored_scored_onto_img_factorized.json",
    },
    {
        "model": "clip_vitl14",
        "benchmark": "coco5k",
        "path": ROOT / "output" / "cross_model_suite" / "clip_vitl14" / "coco5k" / "factorized" / "coco5k_clip_only_scored_scored_onto_img_factorized.json",
    },
    {
        "model": "clip_vitl14",
        "benchmark": "winoground_exploded",
        "path": ROOT / "output" / "cross_model_suite" / "clip_vitl14" / "winoground_exploded" / "factorized" / "winoground_exploded_clip_only_scored_scored_onto_img_factorized.json",
    },
    {
        "model": "blip_itm_base_coco",
        "benchmark": "mixed_vg20k_coco20k_4k",
        "path": ROOT / "output" / "cross_model_blip_torch26_retry" / "blip_itm_base_coco" / "mixed_vg20k_coco20k_4k" / "factorized" / "mixed_vg20k_coco20k_4k_blip_scored_scored_onto_img_factorized.json",
    },
    {
        "model": "blip_itm_base_coco",
        "benchmark": "coco5k",
        "path": ROOT / "output" / "cross_model_blip_torch26_retry" / "blip_itm_base_coco" / "coco5k" / "factorized" / "coco5k_blip_scored_scored_onto_img_factorized.json",
    },
    {
        "model": "blip_itm_base_coco",
        "benchmark": "winoground_exploded",
        "path": ROOT / "output" / "cross_model_blip_torch26_retry" / "blip_itm_base_coco" / "winoground_exploded" / "factorized" / "winoground_exploded_blip_scored_scored_onto_img_factorized.json",
    },
    {
        "model": "siglip_base_patch16_224",
        "benchmark": "mixed_vg20k_coco20k_4k",
        "path": ROOT / "output" / "cross_model_siglip_qwen" / "siglip_base_patch16_224" / "mixed_vg20k_coco20k_4k" / "factorized" / "mixed_vg20k_coco20k_4k_siglip_scored_scored_onto_img_factorized.json",
    },
    {
        "model": "siglip_base_patch16_224",
        "benchmark": "coco5k",
        "path": ROOT / "output" / "cross_model_siglip_qwen" / "siglip_base_patch16_224" / "coco5k" / "factorized" / "coco5k_siglip_scored_scored_onto_img_factorized.json",
    },
    {
        "model": "siglip_base_patch16_224",
        "benchmark": "winoground_exploded",
        "path": ROOT / "output" / "cross_model_siglip_qwen" / "siglip_base_patch16_224" / "winoground_exploded" / "factorized" / "winoground_exploded_siglip_scored_scored_onto_img_factorized.json",
    },
    {
        "model": "bridgetower_base_itm_mlm",
        "benchmark": "mixed_vg20k_coco20k_4k",
        "path": ROOT / "output" / "cross_model_openclip_bridgetower" / "bridgetower_base_itm_mlm" / "mixed_vg20k_coco20k_4k" / "factorized" / "mixed_vg20k_coco20k_4k_bridgetower_scored_scored_onto_img_factorized.json",
    },
    {
        "model": "bridgetower_base_itm_mlm",
        "benchmark": "coco5k",
        "path": ROOT / "output" / "cross_model_openclip_bridgetower" / "bridgetower_base_itm_mlm" / "coco5k" / "factorized" / "coco5k_bridgetower_scored_scored_onto_img_factorized.json",
    },
    {
        "model": "bridgetower_base_itm_mlm",
        "benchmark": "winoground_exploded",
        "path": ROOT / "output" / "cross_model_openclip_bridgetower" / "bridgetower_base_itm_mlm" / "winoground_exploded" / "factorized" / "winoground_exploded_bridgetower_scored_scored_onto_img_factorized.json",
    },
    {
        "model": "clip",
        "benchmark": "sugarcrepe_pilot",
        "path": ROOT / "output" / "sugarcrepe_pilot" / "factorized" / "sugarcrepe_pilot_clip_scored_scored_onto_img_factorized.json",
    },
]


class EvalArgs:
    calibration = "zscore"
    gate_mode = "combined"
    clip_margin_threshold = 0.0
    semantic_confidence_threshold = 0.6
    lambda_delta = 0.35
    lambda_sem = 0.3
    lambda_ground = 0.2
    w_relation = 1.0
    w_attribute = 0.5
    w_spatial = 0.8
    w_count = 0.5
    w_subj = 0.5
    w_obj = 0.5
    w_attr_bind = 1.0
    w_spatial_bind = 1.0
    w_rel_phrase = 1.0
    role_reversal_boost = 1.0
    enable_role_reversal_delta_rule = False
    role_reversal_a = 1.0
    role_reversal_b = 0.7
    role_reversal_c = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate rescue-vs-harm audit for selective correction recipes.")
    p.add_argument("--outdir", default=str(OUT_ROOT))
    p.add_argument("--models", nargs="+", default=["clip_vitl14", "blip_itm_base_coco", "siglip_base_patch16_224", "bridgetower_base_itm_mlm", "clip"])
    p.add_argument("--benchmarks", nargs="+", default=["mixed_vg20k_coco20k_4k", "coco5k", "winoground_exploded", "sugarcrepe_pilot"])
    p.add_argument("--recipes", nargs="+", default=["relphrase_spatialbind", "full_semantic_grounding"])
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def visual_only_summary(sample: Dict[str, Any]) -> Dict[str, Any]:
    pos_score = safe_float(sample.get("positive", {}).get("scores", {}).get("clip"))
    neg_scores = [safe_float(neg.get("scores", {}).get("clip")) for neg in sample.get("negatives", [])]
    neg_scores = [score for score in neg_scores if math.isfinite(score)]
    clip_margin = pos_score - max(neg_scores) if math.isfinite(pos_score) and neg_scores else float("nan")
    neg_severity_pairs = []
    for neg in sample.get("negatives", []):
        sev = safe_float(neg.get("severity"))
        score = safe_float(neg.get("scores", {}).get("clip"))
        if math.isfinite(sev) and math.isfinite(score):
            neg_severity_pairs.append((sev, score))
    return {
        "sample_id": sample.get("sample_id"),
        "source": edit_analysis.source_name(sample),
        "rule": str(sample.get("rule") or "unknown"),
        "recipe": "visual_only",
        "pos_score": pos_score,
        "neg_scores": neg_scores,
        "margin": clip_margin,
        "clip_margin": clip_margin,
        "neg_severity_pairs": neg_severity_pairs,
        "changed_slot_available": False,
        "delta_confidence": 0.0,
        "gate_active": False,
        "grounding_available": False,
        "avg_delta_magnitude": float("nan"),
    }


def build_recipe_summary(sample: Dict[str, Any], recipe_key: str, calibration: Dict[str, Any]) -> Dict[str, Any]:
    args = EvalArgs()
    semantic_weights = {
        "relation": args.w_relation,
        "attribute": args.w_attribute,
        "spatial": args.w_spatial,
        "count": args.w_count,
    }
    grounding_weights = {
        "subj": args.w_subj,
        "obj": args.w_obj,
        "attr_bind": args.w_attr_bind,
        "spatial_bind": args.w_spatial_bind,
        "rel_phrase": args.w_rel_phrase,
    }
    role_weights = {"a": args.role_reversal_a, "b": args.role_reversal_b, "c": args.role_reversal_c}
    if recipe_key == "relphrase_spatialbind":
        branches = ["rel_phrase", "spatial_bind"]
    elif recipe_key == "full_semantic_grounding":
        branches = ["relation", "attribute", "spatial", "count", "subj", "obj", "attr_bind", "spatial_bind", "rel_phrase"]
    else:
        raise ValueError(f"Unsupported recipe: {recipe_key}")
    return edit_analysis.build_contrastive_summary(
        sample=sample,
        calibration=calibration,
        semantic_weights=semantic_weights,
        grounding_weights=grounding_weights,
        clip_margin_threshold=args.clip_margin_threshold,
        conf_threshold=args.semantic_confidence_threshold,
        lambda_delta=args.lambda_delta,
        gate_mode=args.gate_mode,
        config_name=RECIPES[recipe_key],
        branch_subset=branches,
        semantic_subset=None,
        role_reversal_boost=args.role_reversal_boost,
        enable_role_reversal_delta_rule=args.enable_role_reversal_delta_rule,
        role_reversal_weights=role_weights,
    )


def margin_correct(summary: Dict[str, Any]) -> bool:
    value = safe_float(summary.get("margin"))
    return math.isfinite(value) and value > 0.0


def audit_records(visual: Sequence[Dict[str, Any]], corrected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_valid = 0
    activated = 0
    rescue = 0
    harmful = 0
    preserved = 0
    unchanged_wrong = 0
    base_wrong_active = 0
    base_correct_active = 0
    active_delta_mags: List[float] = []
    active_conf: List[float] = []
    examples: Dict[str, List[Dict[str, Any]]] = {"rescue": [], "harmful_flip": []}

    for idx, (base, after) in enumerate(zip(visual, corrected)):
        base_margin = safe_float(base.get("margin"))
        after_margin = safe_float(after.get("margin"))
        if not (math.isfinite(base_margin) and math.isfinite(after_margin)):
            continue
        total_valid += 1
        if not after.get("gate_active"):
            continue
        activated += 1
        base_ok = base_margin > 0.0
        after_ok = after_margin > 0.0
        if base_ok:
            base_correct_active += 1
        else:
            base_wrong_active += 1
        if math.isfinite(safe_float(after.get("avg_delta_magnitude"))):
            active_delta_mags.append(safe_float(after.get("avg_delta_magnitude")))
        if math.isfinite(safe_float(after.get("delta_confidence"))):
            active_conf.append(safe_float(after.get("delta_confidence")))

        if not base_ok and after_ok:
            rescue += 1
            if len(examples["rescue"]) < 10:
                examples["rescue"].append({"index": idx, "base_margin": base_margin, "after_margin": after_margin})
        elif base_ok and not after_ok:
            harmful += 1
            if len(examples["harmful_flip"]) < 10:
                examples["harmful_flip"].append({"index": idx, "base_margin": base_margin, "after_margin": after_margin})
        elif base_ok and after_ok:
            preserved += 1
        else:
            unchanged_wrong += 1

    def rate(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")

    return {
        "total_valid_samples": total_valid,
        "gate_activated_samples": activated,
        "gate_active_rate": rate(activated, total_valid),
        "base_wrong_activated": base_wrong_active,
        "base_correct_activated": base_correct_active,
        "rescue_count": rescue,
        "harmful_flip_count": harmful,
        "preserved_correct_count": preserved,
        "unchanged_wrong_count": unchanged_wrong,
        "rescue_per_activated": rate(rescue, activated),
        "harm_per_activated": rate(harmful, activated),
        "preserved_per_activated": rate(preserved, activated),
        "unchanged_wrong_per_activated": rate(unchanged_wrong, activated),
        "net_rescue": rescue - harmful,
        "net_rescue_per_activated": rate(rescue - harmful, activated),
        "conversion_rate": rate(rescue, base_wrong_active),
        "harm_rate_among_base_correct_active": rate(harmful, base_correct_active),
        "mean_active_delta_magnitude": sum(active_delta_mags) / len(active_delta_mags) if active_delta_mags else float("nan"),
        "mean_active_delta_confidence": sum(active_conf) / len(active_conf) if active_conf else float("nan"),
        "examples": examples,
    }


def evaluate_artifact(model: str, benchmark: str, path: Path, recipe_keys: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    payload = load_json(path)
    samples = edit_analysis.extract_samples(payload)
    calibration = edit_analysis.collect_calibration_stats(samples, "zscore")
    visual = [visual_only_summary(sample) for sample in samples]
    rows: List[Dict[str, Any]] = []
    examples: Dict[str, Any] = {}
    for recipe_key in recipe_keys:
        corrected = [build_recipe_summary(sample, recipe_key, calibration) for sample in samples]
        audit = audit_records(visual, corrected)
        row = {
            "model": model,
            "benchmark": benchmark,
            "recipe_key": recipe_key,
            "recipe": RECIPES[recipe_key],
            **{key: value for key, value in audit.items() if key != "examples"},
        }
        rows.append(row)
        examples[recipe_key] = audit["examples"]
    diagnostics = {
        "model": model,
        "benchmark": benchmark,
        "path": str(path),
        "num_samples": len(samples),
        "examples": examples,
    }
    return rows, diagnostics


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    v = safe_float(value)
    return "nan" if not math.isfinite(v) else f"{v:.4f}"


def write_markdown(path: Path, rows: List[Dict[str, Any]], diagnostics: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Gate Rescue-vs-Harm Audit\n\n")
        f.write("## Setup\n")
        f.write("- Unit: sample-level hardest-negative margin.\n")
        f.write("- Categories are computed only on gate-activated samples.\n")
        f.write("- Base correctness uses visual-only margin; after correctness uses corrected recipe margin.\n")
        f.write("- SugarCrepe is included where an existing CLIP pilot artifact is available.\n\n")

        f.write("## Main Table\n\n")
        f.write("| Model | Benchmark | Recipe | N | Gate active | Rescue | Harm | Preserved | Unchanged wrong | Net rescue | Rescue/active | Harm/active | Conversion |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['model']} | {row['benchmark']} | {row['recipe_key']} | {row['total_valid_samples']} | "
                f"{fmt(row['gate_active_rate'])} | {row['rescue_count']} | {row['harmful_flip_count']} | "
                f"{row['preserved_correct_count']} | {row['unchanged_wrong_count']} | {row['net_rescue']} | "
                f"{fmt(row['rescue_per_activated'])} | {fmt(row['harm_per_activated'])} | {fmt(row['conversion_rate'])} |\n"
            )

        f.write("\n## Model-Benchmark Notes\n")
        for row in rows:
            if row["recipe_key"] != "relphrase_spatialbind":
                continue
            f.write(
                f"- `{row['model']} / {row['benchmark']}` relphrase: active `{fmt(row['gate_active_rate'])}`, "
                f"rescue `{row['rescue_count']}`, harm `{row['harmful_flip_count']}`, net `{row['net_rescue']}`.\n"
            )

        f.write("\n## Takeaway\n")
        positive = [r for r in rows if r["net_rescue"] > 0]
        negative = [r for r in rows if r["net_rescue"] < 0]
        f.write(f"- Positive net rescue rows: `{len(positive)}/{len(rows)}`.\n")
        f.write(f"- Negative net rescue rows: `{len(negative)}/{len(rows)}`.\n")
        f.write("- This audit separates sparse useful correction from activated-but-not-helpful behavior, which is useful for explaining weak-transfer backbones.\n")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    recipe_keys = [key for key in args.recipes if key in RECIPES]
    selected = [
        spec for spec in DATASETS
        if spec["model"] in set(args.models) and spec["benchmark"] in set(args.benchmarks) and spec["path"].exists()
    ]
    rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    for spec in selected:
        spec_rows, diag = evaluate_artifact(spec["model"], spec["benchmark"], spec["path"], recipe_keys)
        rows.extend(spec_rows)
        diagnostics.append(diag)

    payload = {"rows": rows, "diagnostics": diagnostics}
    write_json(outdir / "gate_rescue_harm_audit_results.json", payload)
    write_csv(outdir / "gate_rescue_harm_audit_results.csv", rows)
    write_markdown(outdir / "gate_rescue_harm_audit_summary.md", rows, diagnostics)
    print(f"[saved] {outdir / 'gate_rescue_harm_audit_summary.md'}")
    print(f"[saved] {outdir / 'gate_rescue_harm_audit_results.json'}")


if __name__ == "__main__":
    main()

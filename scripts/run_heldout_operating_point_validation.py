#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import edit_aware_contrastive_grounding_analysis as edit_analysis


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "output" / "heldout_operating_point_validation"

ARTIFACTS = {
    "mixed_vg20k_coco20k_4k": ROOT / "output" / "cross_model_suite" / "clip_vitl14" / "mixed_vg20k_coco20k_4k" / "factorized" / "mixed_vg20k_coco20k_4k_clip_only_scored_scored_onto_img_factorized.json",
    "coco5k": ROOT / "output" / "cross_model_suite" / "clip_vitl14" / "coco5k" / "factorized" / "coco5k_clip_only_scored_scored_onto_img_factorized.json",
    "winoground_exploded": ROOT / "output" / "cross_model_suite" / "clip_vitl14" / "winoground_exploded" / "factorized" / "winoground_exploded_clip_only_scored_scored_onto_img_factorized.json",
    "sugarcrepe_highcompat_50": ROOT / "output" / "sugarcrepe_pilot_highcompat_50" / "factorized" / "sugarcrepe_pilot_clip_scored_scored_onto_img_factorized.json",
}

RECIPES = {
    "relphrase_spatialbind": ["rel_phrase", "spatial_bind"],
    "full_semantic_grounding": ["relation", "attribute", "spatial", "count", "subj", "obj", "attr_bind", "spatial_bind", "rel_phrase"],
}


class EvalArgs:
    calibration = "zscore"
    gate_mode = "combined"
    semantic_confidence_threshold = 0.6
    lambda_sem = 0.3
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
    p = argparse.ArgumentParser(description="Held-out operating-point validation for selective semantic correction.")
    p.add_argument("--outdir", default=str(OUT_ROOT))
    p.add_argument("--dev-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260502)
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.0, -0.002, -0.005, -0.01])
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.20, 0.35, 0.50])
    p.add_argument("--selection-metric", choices=["pairwise", "rescue", "auroc"], default="pairwise")
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


def split_samples(samples: Sequence[Dict[str, Any]], split_name: str, dev_size: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    keyed = []
    for idx, sample in enumerate(samples):
        key = str(sample.get("sample_id") or idx)
        digest = hashlib.sha1(f"{seed}:{split_name}:{key}".encode("utf-8")).hexdigest()
        keyed.append((digest, idx, sample))
    keyed.sort(key=lambda item: (item[0], item[1]))
    dev = [sample for _, _, sample in keyed[: min(dev_size, len(keyed))]]
    test = [sample for _, _, sample in keyed[min(dev_size, len(keyed)):]]
    return dev, test


def weights() -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
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
    return semantic_weights, grounding_weights, role_weights


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


def evaluate_config(
    samples: Sequence[Dict[str, Any]],
    calibration_samples: Sequence[Dict[str, Any]],
    recipe_key: str,
    threshold: float,
    lambda_delta: float,
) -> List[Dict[str, Any]]:
    args = EvalArgs()
    semantic_weights, grounding_weights, role_weights = weights()
    calibration = edit_analysis.collect_calibration_stats(list(calibration_samples), args.calibration)
    records = []
    for sample in samples:
        records.append(
            edit_analysis.build_contrastive_summary(
                sample=sample,
                calibration=calibration,
                semantic_weights=semantic_weights,
                grounding_weights=grounding_weights,
                clip_margin_threshold=threshold,
                conf_threshold=args.semantic_confidence_threshold,
                lambda_delta=lambda_delta,
                gate_mode=args.gate_mode,
                config_name=f"edit_contrastive_{recipe_key}@combined@{threshold:+.3f}@{lambda_delta:.2f}",
                branch_subset=RECIPES[recipe_key],
                semantic_subset=None,
                role_reversal_boost=args.role_reversal_boost,
                enable_role_reversal_delta_rule=args.enable_role_reversal_delta_rule,
                role_reversal_weights=role_weights,
            )
        )
    return records


def evaluate_semantic_baseline(
    samples: Sequence[Dict[str, Any]],
    calibration_samples: Sequence[Dict[str, Any]],
    threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    args = EvalArgs()
    semantic_weights, grounding_weights, _ = weights()
    calibration = edit_analysis.collect_calibration_stats(list(calibration_samples), args.calibration)
    return [
        edit_analysis.build_baseline_summary(
            sample,
            calibration,
            semantic_weights,
            grounding_weights,
            clip_margin_threshold=threshold,
            conf_threshold=args.semantic_confidence_threshold,
            lambda_sem=args.lambda_sem,
        )
        for sample in samples
    ]


def metric_value(metrics: Dict[str, Any], name: str) -> float:
    key = {
        "pairwise": "pairwise_accuracy",
        "rescue": "rescue_rate",
        "auroc": "auroc",
    }[name]
    return safe_float(metrics.get(key))


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = edit_analysis.evaluate_subset(records)
    diag = edit_analysis.summarize_recipe_diagnostics(records)
    return {
        "num_samples": metrics["num_samples"],
        "num_valid_pairwise": metrics["num_valid_pairwise"],
        "num_clip_fail": metrics["num_clip_fail"],
        "pairwise": metrics["pairwise_accuracy"],
        "rescue": metrics["rescue_rate"],
        "auroc": metrics["auroc"],
        "gate_active_rate": diag["activation_rate"],
    }


def row_from_summary(protocol: str, dataset: str, split: str, recipe: str, threshold: Any, lambda_delta: Any, summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "protocol": protocol,
        "dataset": dataset,
        "split": split,
        "recipe": recipe,
        "threshold": threshold,
        "lambda_delta": lambda_delta,
        **summary,
    }


def select_operating_point(
    protocol_name: str,
    dev_samples: Sequence[Dict[str, Any]],
    thresholds: Sequence[float],
    lambdas: Sequence[float],
    selection_metric: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dev_rows = []
    for recipe_key in RECIPES:
        for threshold in thresholds:
            for lambda_delta in lambdas:
                records = evaluate_config(dev_samples, dev_samples, recipe_key, threshold, lambda_delta)
                summary = summarize_records(records)
                dev_rows.append(row_from_summary(protocol_name, "dev_source", "dev", recipe_key, threshold, lambda_delta, summary))
    def sort_key(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
        return (
            safe_float(row[selection_metric]),
            safe_float(row["rescue"]),
            safe_float(row["auroc"]),
            -abs(safe_float(row["threshold"])),
        )
    best = max(dev_rows, key=sort_key)
    return {
        "recipe": best["recipe"],
        "threshold": float(best["threshold"]),
        "lambda_delta": float(best["lambda_delta"]),
        "selection_metric": selection_metric,
        "dev_pairwise": best["pairwise"],
        "dev_rescue": best["rescue"],
        "dev_auroc": best["auroc"],
        "dev_gate_active_rate": best["gate_active_rate"],
    }, dev_rows


def run_protocol(
    protocol_name: str,
    dev_dataset: str,
    payloads: Dict[str, Any],
    dev_size: int,
    seed: int,
    thresholds: Sequence[float],
    lambdas: Sequence[float],
    selection_metric: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    all_samples = {name: edit_analysis.extract_samples(payload) for name, payload in payloads.items()}
    dev_samples, heldout_dev_source = split_samples(all_samples[dev_dataset], dev_dataset, dev_size, seed)
    selected, dev_grid_rows = select_operating_point(protocol_name, dev_samples, thresholds, lambdas, selection_metric)
    rows: List[Dict[str, Any]] = []

    for dataset, samples in all_samples.items():
        if dataset == dev_dataset:
            eval_samples = heldout_dev_source
            split_name = f"{dataset}_heldout"
        else:
            eval_samples = samples
            split_name = dataset
        if not eval_samples:
            continue
        calibration_samples = samples
        visual = [visual_only_summary(sample) for sample in eval_samples]
        semantic = evaluate_semantic_baseline(eval_samples, calibration_samples, threshold=0.0)
        selected_records = evaluate_config(
            eval_samples,
            calibration_samples,
            selected["recipe"],
            selected["threshold"],
            selected["lambda_delta"],
        )
        canonical_rel = evaluate_config(eval_samples, calibration_samples, "relphrase_spatialbind", 0.0, 0.35)
        canonical_full = evaluate_config(eval_samples, calibration_samples, "full_semantic_grounding", 0.0, 0.35)

        rows.append(row_from_summary(protocol_name, dataset, split_name, "visual_only", "", "", summarize_records(visual)))
        rows.append(row_from_summary(protocol_name, dataset, split_name, "semantic_baseline@+0.000", 0.0, "", summarize_records(semantic)))
        rows.append(row_from_summary(protocol_name, dataset, split_name, f"heldout_selected::{selected['recipe']}", selected["threshold"], selected["lambda_delta"], summarize_records(selected_records)))
        rows.append(row_from_summary(protocol_name, dataset, split_name, "canonical::relphrase_spatialbind", 0.0, 0.35, summarize_records(canonical_rel)))
        rows.append(row_from_summary(protocol_name, dataset, split_name, "canonical::full_semantic_grounding", 0.0, 0.35, summarize_records(canonical_full)))
    selected["protocol"] = protocol_name
    selected["dev_dataset"] = dev_dataset
    selected["dev_size"] = len(dev_samples)
    selected["heldout_dev_source_size"] = len(heldout_dev_source)
    return rows, dev_grid_rows, selected


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


def write_markdown(path: Path, rows: List[Dict[str, Any]], selected_points: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_rows = [r for r in rows if str(r["recipe"]).startswith("heldout_selected::")]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Held-out Operating-point Validation\n\n")
        f.write("## Setup\n")
        f.write("- Backbone/artifacts: CLIP L/14 for Mixed, COCO, Winoground; available CLIP SugarCrepe high-compat pilot.\n")
        f.write("- Dev selection searches only recipe, gate threshold, and lambda on a held-out 1k source split.\n")
        f.write("- The selected operating point is then frozen and transferred to the held-out source remainder plus the other benchmarks.\n\n")
        f.write("## Selected Operating Points\n\n")
        f.write("| Protocol | Dev dataset | Dev N | Recipe | Threshold | Lambda | Dev pairwise | Dev rescue | Dev AUROC |\n")
        f.write("|---|---|---:|---|---:|---:|---:|---:|---:|\n")
        for point in selected_points:
            f.write(
                f"| {point['protocol']} | {point['dev_dataset']} | {point['dev_size']} | {point['recipe']} | "
                f"{fmt(point['threshold'])} | {fmt(point['lambda_delta'])} | {fmt(point['dev_pairwise'])} | "
                f"{fmt(point['dev_rescue'])} | {fmt(point['dev_auroc'])} |\n"
            )
        f.write("\n## Frozen Transfer Results\n\n")
        f.write("| Protocol | Dataset | Split | Recipe | Pairwise | Rescue | AUROC | Gate |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['protocol']} | {row['dataset']} | {row['split']} | {row['recipe']} | "
                f"{fmt(row['pairwise'])} | {fmt(row['rescue'])} | {fmt(row['auroc'])} | {fmt(row['gate_active_rate'])} |\n"
            )
        f.write("\n## Selected Recipe Deltas vs Baselines\n\n")
        f.write("| Protocol | Dataset | Selected - visual pairwise | Selected - semantic pairwise | Selected - visual rescue | Selected - semantic rescue |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for selected in selected_rows:
            base_key = (selected["protocol"], selected["dataset"], selected["split"])
            visual = next((r for r in rows if (r["protocol"], r["dataset"], r["split"]) == base_key and r["recipe"] == "visual_only"), None)
            semantic = next((r for r in rows if (r["protocol"], r["dataset"], r["split"]) == base_key and str(r["recipe"]).startswith("semantic_baseline")), None)
            if not visual or not semantic:
                continue
            f.write(
                f"| {selected['protocol']} | {selected['dataset']} | "
                f"{fmt(safe_float(selected['pairwise']) - safe_float(visual['pairwise']))} | "
                f"{fmt(safe_float(selected['pairwise']) - safe_float(semantic['pairwise']))} | "
                f"{fmt(safe_float(selected['rescue']) - safe_float(visual['rescue']))} | "
                f"{fmt(safe_float(selected['rescue']) - safe_float(semantic['rescue']))} |\n"
            )
        f.write("\n## Takeaway\n")
        wins_vs_visual = sum(
            1 for selected in selected_rows
            for visual in rows
            if visual["protocol"] == selected["protocol"]
            and visual["dataset"] == selected["dataset"]
            and visual["split"] == selected["split"]
            and visual["recipe"] == "visual_only"
            and safe_float(selected["pairwise"]) > safe_float(visual["pairwise"])
        )
        f.write(f"- Held-out selected recipe beats visual-only pairwise in `{wins_vs_visual}/{len(selected_rows)}` frozen transfer evaluations.\n")
        f.write("- The operating point is fixed before transfer, reducing concern that each benchmark is independently tuned post hoc.\n")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    payloads = {name: load_json(path) for name, path in ARTIFACTS.items() if path.exists()}
    all_rows: List[Dict[str, Any]] = []
    all_dev_rows: List[Dict[str, Any]] = []
    selected_points: List[Dict[str, Any]] = []
    protocols = [
        ("mixed_dev_to_transfer", "mixed_vg20k_coco20k_4k"),
        ("coco_dev_to_transfer", "coco5k"),
    ]
    for protocol_name, dev_dataset in protocols:
        rows, dev_rows, selected = run_protocol(
            protocol_name,
            dev_dataset,
            payloads,
            args.dev_size,
            args.seed,
            args.thresholds,
            args.lambdas,
            args.selection_metric,
        )
        all_rows.extend(rows)
        all_dev_rows.extend(dev_rows)
        selected_points.append(selected)
    write_json(outdir / "heldout_operating_point_validation_results.json", {
        "selected_points": selected_points,
        "rows": all_rows,
        "dev_grid_rows": all_dev_rows,
    })
    write_csv(outdir / "heldout_operating_point_validation_results.csv", all_rows)
    write_csv(outdir / "heldout_operating_point_validation_dev_grid.csv", all_dev_rows)
    write_markdown(outdir / "heldout_operating_point_validation_summary.md", all_rows, selected_points)
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()

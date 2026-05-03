#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import edit_aware_contrastive_grounding_analysis as edit_analysis


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "output" / "backbone_calibration_test"

ARTIFACTS = [
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
        "model": "bridgetower_base_itm_mlm",
        "benchmark": "mixed_vg20k_coco20k_4k",
        "path": ROOT / "output" / "cross_model_openclip_bridgetower" / "bridgetower_base_itm_mlm" / "mixed_vg20k_coco20k_4k" / "factorized" / "mixed_vg20k_coco20k_4k_bridgetower_scored_scored_onto_img_factorized.json",
    },
    {
        "model": "bridgetower_base_itm_mlm",
        "benchmark": "coco5k",
        "path": ROOT / "output" / "cross_model_openclip_bridgetower" / "bridgetower_base_itm_mlm" / "coco5k" / "factorized" / "coco5k_bridgetower_scored_scored_onto_img_factorized.json",
    },
]

RECIPES = {
    "relphrase_spatialbind": ["rel_phrase", "spatial_bind"],
    "full_semantic_grounding": ["relation", "attribute", "spatial", "count", "subj", "obj", "attr_bind", "spatial_bind", "rel_phrase"],
    "relphrase_only": ["rel_phrase"],
    "spatialbind_only": ["spatial_bind"],
}

CALIBRATION_MODES = [
    "fixed_lambda",
    "margin_std_scaled_lambda",
    "zscore_margin_correction",
    "delta_std_normalized",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backbone-aware calibration test for selective correction.")
    p.add_argument("--outdir", default=str(OUT_ROOT))
    p.add_argument("--lambda-delta", type=float, default=0.35)
    p.add_argument("--clip-margin-threshold", type=float, default=0.0)
    p.add_argument("--confidence-threshold", type=float, default=0.6)
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_samples(payload: Any) -> List[Dict[str, Any]]:
    return edit_analysis.extract_samples(payload)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    vals = finite(values)
    if not vals:
        return float("nan"), float("nan")
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    return float(mu), float(math.sqrt(var))


def weights() -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    semantic_weights = {"relation": 1.0, "attribute": 0.5, "spatial": 0.8, "count": 0.5}
    grounding_weights = {"subj": 0.5, "obj": 0.5, "attr_bind": 1.0, "spatial_bind": 1.0, "rel_phrase": 1.0}
    role_weights = {"a": 1.0, "b": 0.7, "c": 0.5}
    return semantic_weights, grounding_weights, role_weights


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    return edit_analysis.auroc(list(scores), list(labels))


def visual_margin(sample: Dict[str, Any]) -> float:
    pos = safe_float((sample.get("positive") or {}).get("scores", {}).get("clip"))
    negs = [safe_float((neg.get("scores") or {}).get("clip")) for neg in sample.get("negatives", [])]
    negs = [v for v in negs if math.isfinite(v)]
    return pos - max(negs) if math.isfinite(pos) and negs else float("nan")


def semantic_summary(
    sample: Dict[str, Any],
    calibration: Dict[str, Dict[str, edit_analysis.CalibrationStats]],
    semantic_weights: Dict[str, float],
    grounding_weights: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return edit_analysis.build_baseline_summary(
        sample,
        calibration,
        semantic_weights,
        grounding_weights,
        clip_margin_threshold=args.clip_margin_threshold,
        conf_threshold=args.confidence_threshold,
        lambda_sem=0.3,
    )


def pair_deltas(
    sample: Dict[str, Any],
    calibration: Dict[str, Dict[str, edit_analysis.CalibrationStats]],
    semantic_weights: Dict[str, float],
    grounding_weights: Dict[str, float],
    recipe_branches: Sequence[str],
) -> List[Dict[str, Any]]:
    candidates = [sample.get("positive", {})] + list(sample.get("negatives") or [])
    features = [edit_analysis.build_candidate_features(candidate) for candidate in candidates]
    branches = [edit_analysis.calc_candidate_branch_scores(feat, calibration, semantic_weights, grounding_weights, None) for feat in features]
    out = []
    for neg_feat, neg_branch in zip(features[1:], branches[1:]):
        changed_slots = edit_analysis.infer_changed_slots(str(sample.get("rule") or ""), features[0]["signature"], neg_feat["signature"])
        selected_branches = edit_analysis.select_delta_branches(changed_slots, recipe_branches)
        delta = edit_analysis.compute_delta(
            features[0],
            neg_feat,
            branches[0],
            neg_branch,
            selected_branches,
            semantic_weights,
            grounding_weights,
            str(sample.get("rule") or ""),
            role_reversal_boost=1.0,
        )
        out.append(
            {
                "base_pair_margin": features[0]["clip"] - neg_feat["clip"] if math.isfinite(features[0]["clip"]) and math.isfinite(neg_feat["clip"]) else float("nan"),
                "delta_score": safe_float(delta["delta_score"]),
                "delta_confidence": safe_float(delta["delta_confidence"]),
                "available": bool(delta["available"]),
            }
        )
    return out


def final_margin_for_mode(
    base_pair_margin: float,
    delta_score: float,
    gate: float,
    mode: str,
    lambda_delta: float,
    margin_mu: float,
    margin_std: float,
    delta_std: float,
) -> float:
    if not math.isfinite(base_pair_margin):
        return float("nan")
    if gate <= 0.0 or not math.isfinite(delta_score):
        return base_pair_margin
    if mode == "fixed_lambda":
        correction = lambda_delta * delta_score
        return base_pair_margin + correction
    if mode == "margin_std_scaled_lambda":
        scale = margin_std if math.isfinite(margin_std) and margin_std > 1e-12 else 1.0
        correction = lambda_delta * scale * delta_score
        return base_pair_margin + correction
    if mode == "zscore_margin_correction":
        scale = margin_std if math.isfinite(margin_std) and margin_std > 1e-12 else 1.0
        z_base = (base_pair_margin - margin_mu) / scale
        z_boundary = (0.0 - margin_mu) / scale
        z_final = z_base + lambda_delta * delta_score
        # Return signed distance to the original decision boundary in z-space.
        return z_final - z_boundary
    if mode == "delta_std_normalized":
        scale = margin_std if math.isfinite(margin_std) and margin_std > 1e-12 else 1.0
        dscale = delta_std if math.isfinite(delta_std) and delta_std > 1e-12 else 1.0
        correction = lambda_delta * scale * (delta_score / dscale)
        return base_pair_margin + correction
    raise ValueError(f"Unknown calibration mode: {mode}")


def evaluate_recipe_mode(
    samples: Sequence[Dict[str, Any]],
    calibration: Dict[str, Dict[str, edit_analysis.CalibrationStats]],
    semantic_weights: Dict[str, float],
    grounding_weights: Dict[str, float],
    recipe_name: str,
    mode: str,
    args: argparse.Namespace,
    margin_mu: float,
    margin_std: float,
    delta_std: float,
) -> Dict[str, Any]:
    margins = []
    base_margins = []
    gate_active = 0
    delta_values = []
    pos_scores = []
    neg_scores = []
    for sample in samples:
        pair_payloads = pair_deltas(sample, calibration, semantic_weights, grounding_weights, RECIPES[recipe_name])
        pos_clip = safe_float((sample.get("positive") or {}).get("scores", {}).get("clip"))
        final_pair_margins = []
        adjusted_negs = []
        clip_margin = visual_margin(sample)
        for neg, payload in zip(sample.get("negatives", []), pair_payloads):
            gate = edit_analysis.gate_value("combined", clip_margin, payload["delta_confidence"], args.clip_margin_threshold, args.confidence_threshold)
            gate_active += 1 if gate > 0 else 0
            if math.isfinite(payload["delta_score"]):
                delta_values.append(payload["delta_score"])
            final_margin = final_margin_for_mode(
                payload["base_pair_margin"],
                payload["delta_score"],
                gate,
                mode,
                args.lambda_delta,
                margin_mu,
                margin_std,
                delta_std,
            )
            final_pair_margins.append(final_margin)
            neg_clip = safe_float((neg.get("scores") or {}).get("clip"))
            if mode == "zscore_margin_correction":
                # Candidate AUROC is not comparable after pairwise z-boundary correction.
                adjusted_negs.append(float("nan"))
            else:
                correction_margin = final_margin - payload["base_pair_margin"] if math.isfinite(final_margin) and math.isfinite(payload["base_pair_margin"]) else 0.0
                adjusted_negs.append(neg_clip - correction_margin if math.isfinite(neg_clip) else float("nan"))
        margin = min([m for m in final_pair_margins if math.isfinite(m)], default=float("nan"))
        margins.append(margin)
        base_margins.append(clip_margin)
        if math.isfinite(pos_clip):
            pos_scores.append(pos_clip)
        for score in adjusted_negs:
            if math.isfinite(score):
                neg_scores.append(score)
    valid = [m for m in margins if math.isfinite(m)]
    base_valid = [m for m in base_margins if math.isfinite(m)]
    base_fail_mask = [math.isfinite(m) and m <= 0 for m in base_margins]
    rescue_vals = [1 if margins[i] > 0 else 0 for i, fail in enumerate(base_fail_mask) if fail and math.isfinite(margins[i])]
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    auc = auroc(pos_scores + neg_scores, labels) if pos_scores and neg_scores else float("nan")
    return {
        "pairwise": sum(1 for m in valid if m > 0) / len(valid) if valid else float("nan"),
        "rescue": sum(rescue_vals) / len(rescue_vals) if rescue_vals else float("nan"),
        "auroc": auc,
        "gate_active_rate": gate_active / max(1, sum(len(sample.get("negatives", [])) for sample in samples)),
        "mean_abs_correction_delta": sum(abs(v) for v in delta_values) / len(delta_values) if delta_values else float("nan"),
        "num_valid": len(valid),
        "num_base_fail": sum(base_fail_mask),
    }


def evaluate_artifact(cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    payload = load_json(cfg["path"])
    samples = extract_samples(payload)
    semantic_weights, grounding_weights, _ = weights()
    calibration = edit_analysis.collect_calibration_stats(samples, "zscore")
    base_margins = [visual_margin(sample) for sample in samples]
    margin_mu, margin_std = mean_std(base_margins)
    semantic_records = [semantic_summary(sample, calibration, semantic_weights, grounding_weights, args) for sample in samples]
    semantic_pairwise = edit_analysis.evaluate_subset(semantic_records)["pairwise_accuracy"]
    semantic_rescue = edit_analysis.evaluate_subset(semantic_records)["rescue_rate"]
    rows = []
    for recipe_name, branches in RECIPES.items():
        all_deltas = []
        for sample in samples:
            for payload_delta in pair_deltas(sample, calibration, semantic_weights, grounding_weights, branches):
                if math.isfinite(payload_delta["delta_score"]):
                    all_deltas.append(payload_delta["delta_score"])
        _, delta_std = mean_std(all_deltas)
        for mode in CALIBRATION_MODES:
            metrics = evaluate_recipe_mode(
                samples,
                calibration,
                semantic_weights,
                grounding_weights,
                recipe_name,
                mode,
                args,
                margin_mu,
                margin_std,
                delta_std,
            )
            rows.append(
                {
                    "model": cfg["model"],
                    "benchmark": cfg["benchmark"],
                    "artifact_path": str(cfg["path"]),
                    "recipe": recipe_name,
                    "calibration_mode": mode,
                    "lambda_delta": args.lambda_delta,
                    "base_margin_mean": margin_mu,
                    "base_margin_std": margin_std,
                    "delta_std": delta_std,
                    "semantic_pairwise": semantic_pairwise,
                    "semantic_rescue": semantic_rescue,
                    **metrics,
                    "pairwise_delta_vs_semantic": metrics["pairwise"] - semantic_pairwise if math.isfinite(metrics["pairwise"]) and math.isfinite(semantic_pairwise) else float("nan"),
                    "rescue_delta_vs_semantic": metrics["rescue"] - semantic_rescue if math.isfinite(metrics["rescue"]) and math.isfinite(semantic_rescue) else float("nan"),
                }
            )
    audit = {
        "model": cfg["model"],
        "benchmark": cfg["benchmark"],
        "artifact_path": str(cfg["path"]),
        "num_samples": len(samples),
        "base_margin_mean": margin_mu,
        "base_margin_std": margin_std,
        "semantic_pairwise": semantic_pairwise,
        "semantic_rescue": semantic_rescue,
    }
    return rows, audit


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
    return "NA" if not math.isfinite(v) else f"{v:.4f}"


def write_latex(path: Path, rows: List[Dict[str, Any]]) -> None:
    selected = []
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["benchmark"], row["recipe"]), []).append(row)
    for group in grouped.values():
        selected.extend(sorted(group, key=lambda r: safe_float(r["pairwise"]), reverse=True)[:1])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lllrrrr}\n")
        f.write("\\toprule\n")
        f.write("Model & Benchmark & Recipe & Best mode & Pairwise & Rescue & $\\Delta$ pairwise \\\\\n")
        f.write("\\midrule\n")
        for row in selected:
            model = row["model"].replace("_", "\\_")
            benchmark = row["benchmark"].replace("_", "\\_")
            recipe = row["recipe"].replace("_", "\\_")
            mode = row["calibration_mode"].replace("_", "\\_")
            f.write(
                f"{model} & {benchmark} & {recipe} & "
                f"{mode} & {fmt(row['pairwise'])} & {fmt(row['rescue'])} & {fmt(row['pairwise_delta_vs_semantic'])} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_summary(path: Path, rows: List[Dict[str, Any]], audits: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fixed = [r for r in rows if r["calibration_mode"] == "fixed_lambda"]
    improved = []
    for row in fixed:
        peers = [r for r in rows if r["model"] == row["model"] and r["benchmark"] == row["benchmark"] and r["recipe"] == row["recipe"] and r["calibration_mode"] != "fixed_lambda"]
        best_peer = max(peers, key=lambda r: safe_float(r["pairwise"]), default=None)
        if best_peer and safe_float(best_peer["pairwise"]) > safe_float(row["pairwise"]):
            improved.append((row, best_peer))
    with path.open("w", encoding="utf-8") as f:
        f.write("# Backbone-aware Calibration Test\n\n")
        f.write("## Purpose\n")
        f.write("This experiment tests whether weak transfer on SigLIP and BridgeTower can be explained by score-scale mismatch. It compares fixed correction strength against backbone margin-scale and z-score variants.\n\n")
        f.write("## Margin Geometry Audit\n\n")
        f.write("| Model | Benchmark | N | Base margin mean | Base margin std | Semantic pairwise | Semantic rescue |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for audit in audits:
            f.write(f"| {audit['model']} | {audit['benchmark']} | {audit['num_samples']} | {fmt(audit['base_margin_mean'])} | {fmt(audit['base_margin_std'])} | {fmt(audit['semantic_pairwise'])} | {fmt(audit['semantic_rescue'])} |\n")
        f.write("\n## Calibration Results\n\n")
        f.write("| Model | Benchmark | Recipe | Mode | Pairwise | Rescue | AUROC | Gate | Delta vs semantic |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(f"| {row['model']} | {row['benchmark']} | {row['recipe']} | {row['calibration_mode']} | {fmt(row['pairwise'])} | {fmt(row['rescue'])} | {fmt(row['auroc'])} | {fmt(row['gate_active_rate'])} | {fmt(row['pairwise_delta_vs_semantic'])} |\n")
        f.write("\n## Fixed-vs-Calibrated Comparison\n\n")
        if improved:
            f.write(f"- At least one calibrated mode improves over fixed lambda in `{len(improved)}` recipe/model/benchmark comparisons.\n")
        else:
            f.write("- No calibrated mode improves over fixed lambda on pairwise accuracy.\n")
        for old, new in improved[:12]:
            f.write(f"- {old['model']} / {old['benchmark']} / {old['recipe']}: fixed {fmt(old['pairwise'])} -> {new['calibration_mode']} {fmt(new['pairwise'])}\n")
        f.write("\n## Interpretation\n")
        f.write("If margin-scaled or z-score correction improves SigLIP/BridgeTower, the weak-transfer pattern is partly attributable to score-interface mismatch. If not, the result suggests that simple score-scale calibration is insufficient and that the correction interface itself is backbone-specific.\n")


def write_plots(outdir: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model and r["recipe"] in {"relphrase_spatialbind", "full_semantic_grounding"}]
        labels = [f"{r['benchmark']}\n{r['recipe']}\n{r['calibration_mode']}" for r in sub]
        vals = [safe_float(r["pairwise"]) for r in sub]
        plt.figure(figsize=(max(8, len(sub) * 0.45), 4))
        plt.bar(range(len(sub)), vals)
        plt.xticks(range(len(sub)), labels, rotation=75, ha="right", fontsize=7)
        plt.ylabel("Pairwise")
        plt.title(f"{model} calibration modes")
        plt.tight_layout()
        plt.savefig(outdir / f"{model}_calibration_pairwise.png", dpi=160)
        plt.close()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    skipped = []
    for cfg in ARTIFACTS:
        if not cfg["path"].exists():
            skipped.append({"artifact_path": str(cfg["path"]), "reason": "missing"})
            continue
        artifact_rows, audit = evaluate_artifact(cfg, args)
        rows.extend(artifact_rows)
        audits.append(audit)
    write_csv(outdir / "backbone_calibration_results.csv", rows)
    write_csv(outdir / "backbone_calibration_margin_audit.csv", audits)
    (outdir / "backbone_calibration_results.json").write_text(json.dumps({"rows": rows, "audits": audits, "skipped": skipped}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_latex(outdir / "backbone_calibration_latex_table.tex", rows)
    write_summary(outdir / "backbone_calibration_summary.md", rows, audits)
    write_plots(outdir, rows)
    fixed = [r for r in rows if r["calibration_mode"] == "fixed_lambda"]
    improved = 0
    for row in fixed:
        peers = [r for r in rows if r["model"] == row["model"] and r["benchmark"] == row["benchmark"] and r["recipe"] == row["recipe"] and r["calibration_mode"] != "fixed_lambda"]
        if peers and max(safe_float(r["pairwise"]) for r in peers) > safe_float(row["pairwise"]):
            improved += 1
    print(f"usable_artifacts: {len(audits)}")
    print(f"output_dir: {outdir}")
    print(f"results_csv: {outdir / 'backbone_calibration_results.csv'}")
    print(f"summary_md: {outdir / 'backbone_calibration_summary.md'}")
    print(f"calibrated_improvements_over_fixed: {improved}/{len(fixed)}")
    best = max(rows, key=lambda r: safe_float(r["pairwise_delta_vs_semantic"]), default=None)
    if best:
        print(f"strongest_delta_vs_semantic: {best['model']} {best['benchmark']} {best['recipe']} {best['calibration_mode']} {fmt(best['pairwise_delta_vs_semantic'])}")


if __name__ == "__main__":
    main()

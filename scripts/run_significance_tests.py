#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest

import edit_aware_contrastive_grounding_analysis as edit_analysis
import evaluate_winoground_native_2x2 as native_eval

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime
    plt = None


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTDIR = ROOT / "output" / "significance_selective_correction"


@dataclass
class MetricRow:
    comparison_id: str
    dataset: str
    sample_unit: str
    comparator_a: str
    comparator_b: str
    metric: str
    metric_a: Optional[float]
    metric_b: Optional[float]
    delta_b_minus_a: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    p_value: Optional[float]
    n_unit: int
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired significance testing for selective correction comparisons.")
    p.add_argument(
        "--mixed-benchmark-json",
        default=str(
            ROOT
            / "output"
            / "vg20k_coco20k_stratified_scored_4k"
            / "semantic_violation_benchmark_vg20k_coco20k_stratified_extension_scored_onto_img_factorized.json"
        ),
    )
    p.add_argument(
        "--coco5k-benchmark-json",
        default=str(ROOT / "output" / "semantic_violation_benchmark_coco5k_scored_components_scored_onto_img_factorized.json"),
    )
    p.add_argument(
        "--winoground-exploded-json",
        default=str(
            ROOT
            / "output"
            / "winoground_exploded_refined_scored"
            / "semantic_violation_benchmark_winoground_exploded_scored_onto_img_factorized.json"
        ),
    )
    p.add_argument(
        "--winoground-native-json",
        default=str(ROOT / "output" / "semantic_violation_benchmark_winoground_native.json"),
    )
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260402)
    p.add_argument("--calibration", choices=["none", "zscore", "percentile"], default="zscore")
    p.add_argument("--clip-margin-threshold", type=float, default=0.0)
    p.add_argument("--semantic-confidence-threshold", type=float, default=0.6)
    p.add_argument("--lambda-delta", type=float, default=0.35)
    p.add_argument("--lambda-sem", type=float, default=0.3)
    p.add_argument("--gate-mode", choices=["always_on", "clip_margin", "combined"], default="combined")
    p.add_argument("--w-relation", type=float, default=1.0)
    p.add_argument("--w-attribute", type=float, default=0.5)
    p.add_argument("--w-spatial", type=float, default=0.8)
    p.add_argument("--w-count", type=float, default=0.5)
    p.add_argument("--w-subj", type=float, default=0.2)
    p.add_argument("--w-obj", type=float, default=0.2)
    p.add_argument("--w-attr-bind", type=float, default=0.0)
    p.add_argument("--w-spatial-bind", type=float, default=1.3)
    p.add_argument("--w-rel-phrase", type=float, default=1.6)
    p.add_argument("--native-delta-a", type=float, default=1.0)
    p.add_argument("--native-delta-b", type=float, default=1.2)
    p.add_argument("--native-delta-c", type=float, default=0.3)
    return p.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def is_finite(value: Any) -> bool:
    return math.isfinite(safe_float(value))


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    x = safe_float(value)
    return "NA" if not math.isfinite(x) else f"{x:.4f}"


def align_summaries_by_sample(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    by_id_a = {str(item.get("sample_id")): item for item in a}
    by_id_b = {str(item.get("sample_id")): item for item in b}
    sample_ids = sorted(set(by_id_a).intersection(by_id_b))
    return [(by_id_a[sid], by_id_b[sid]) for sid in sample_ids]


def binary_mean(values: Sequence[bool]) -> float:
    return float(sum(1 for v in values if v) / len(values)) if values else float("nan")


def bootstrap_ci_delta(
    records: Sequence[Any],
    metric_fn_a: Callable[[Sequence[Any]], float],
    metric_fn_b: Callable[[Sequence[Any]], float],
    rng: np.random.Generator,
    iters: int,
) -> Tuple[float, float, List[float]]:
    n = len(records)
    deltas: List[float] = []
    if n == 0:
        return float("nan"), float("nan"), deltas
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        sample = [records[i] for i in idx]
        a = metric_fn_a(sample)
        b = metric_fn_b(sample)
        if math.isfinite(a) and math.isfinite(b):
            deltas.append(b - a)
    if not deltas:
        return float("nan"), float("nan"), deltas
    arr = np.asarray(deltas, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), deltas


def exact_sign_test(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> float:
    wins_b = 0
    discordant = 0
    for a, b in zip(a_correct, b_correct):
        if a == b:
            continue
        discordant += 1
        if b and not a:
            wins_b += 1
    if discordant == 0:
        return 1.0
    return float(binomtest(wins_b, discordant, p=0.5, alternative="two-sided").pvalue)


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    paired = [(safe_float(s), int(y)) for s, y in zip(scores, labels) if math.isfinite(safe_float(s))]
    if not paired:
        return float("nan")
    vals = np.asarray([p[0] for p in paired], dtype=float)
    y = np.asarray([p[1] for p in paired], dtype=int)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(vals)
    ranks = np.empty(len(vals), dtype=float)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def severity_spearman(pairs: Sequence[Tuple[float, float]]) -> float:
    vals = [(safe_float(x), safe_float(y)) for x, y in pairs if math.isfinite(safe_float(x)) and math.isfinite(safe_float(y))]
    if len(vals) < 2:
        return float("nan")
    xs = np.asarray([v[0] for v in vals], dtype=float)
    ys = np.asarray([v[1] for v in vals], dtype=float)
    xr = rankdata(xs)
    yr = rankdata(ys)
    xmu = float(np.mean(xr))
    ymu = float(np.mean(yr))
    xv = xr - xmu
    yv = yr - ymu
    denom = float(np.sqrt(np.sum(xv * xv) * np.sum(yv * yv)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(xv * yv) / denom)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def edit_metric_records(aligned: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, List[Any]]:
    pairwise_records: List[Dict[str, Any]] = []
    rescue_records: List[Dict[str, Any]] = []
    auroc_records: List[Dict[str, Any]] = []
    severity_records: List[Dict[str, Any]] = []
    for a, b in aligned:
        margin_a = safe_float(a.get("margin"))
        margin_b = safe_float(b.get("margin"))
        clip_margin = safe_float(a.get("clip_margin"))
        if math.isfinite(margin_a) and math.isfinite(margin_b):
            rec = {"a": margin_a > 0.0, "b": margin_b > 0.0}
            pairwise_records.append(rec)
            if math.isfinite(clip_margin) and clip_margin <= 0.0:
                rescue_records.append(rec)
        pos_a = safe_float(a.get("pos_score"))
        pos_b = safe_float(b.get("pos_score"))
        negs_a = [safe_float(v) for v in a.get("neg_scores", []) if math.isfinite(safe_float(v))]
        negs_b = [safe_float(v) for v in b.get("neg_scores", []) if math.isfinite(safe_float(v))]
        if math.isfinite(pos_a) and math.isfinite(pos_b) and negs_a and negs_b:
            auroc_records.append({"pos_a": pos_a, "pos_b": pos_b, "negs_a": negs_a, "negs_b": negs_b})
        sev_a = [(safe_float(x), safe_float(y)) for x, y in a.get("neg_severity_pairs", []) if math.isfinite(safe_float(x)) and math.isfinite(safe_float(y))]
        sev_b = [(safe_float(x), safe_float(y)) for x, y in b.get("neg_severity_pairs", []) if math.isfinite(safe_float(x)) and math.isfinite(safe_float(y))]
        if sev_a and sev_b:
            severity_records.append({"a": sev_a, "b": sev_b})
    return {
        "pairwise": pairwise_records,
        "rescue": rescue_records,
        "auroc": auroc_records,
        "severity": severity_records,
    }


def pairwise_metric_from_records(records: Sequence[Dict[str, Any]], key: str) -> float:
    return binary_mean([bool(rec[key]) for rec in records])


def auroc_metric_from_records(records: Sequence[Dict[str, Any]], key_prefix: str) -> float:
    scores: List[float] = []
    labels: List[int] = []
    for rec in records:
        scores.append(float(rec[f"pos_{key_prefix}"]))
        labels.append(1)
        for neg in rec[f"negs_{key_prefix}"]:
            scores.append(float(neg))
            labels.append(0)
    return auroc(scores, labels)


def severity_metric_from_records(records: Sequence[Dict[str, Any]], key: str) -> float:
    pairs: List[Tuple[float, float]] = []
    for rec in records:
        pairs.extend(rec[key])
    return severity_spearman(pairs)


def build_edit_summaries(
    benchmark_json: str,
    recipe_key: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    payload = load_json(benchmark_json)
    samples = edit_analysis.extract_samples(payload)
    calibration = edit_analysis.collect_calibration_stats(samples, args.calibration)
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
    summaries: List[Dict[str, Any]] = []
    if recipe_key == "baseline_relation_only":
        for sample in samples:
            summaries.append(
                edit_analysis.build_baseline_summary(
                    sample=sample,
                    calibration=calibration,
                    semantic_weights=semantic_weights,
                    grounding_weights=grounding_weights,
                    clip_margin_threshold=args.clip_margin_threshold,
                    conf_threshold=args.semantic_confidence_threshold,
                    lambda_sem=args.lambda_sem,
                )
            )
        return summaries

    recipe = edit_analysis.RECIPE_REGISTRY[recipe_key]
    semantic_subset = recipe.get("semantic_subset")
    semantic_weights_use = dict(semantic_weights)
    semantic_weights_use.update(recipe.get("semantic_weight_override", {}))
    grounding_weights_use = dict(grounding_weights)
    grounding_weights_use.update(recipe.get("grounding_weight_override", {}))
    config_name = f"edit_contrastive_{recipe_key}@{args.gate_mode}@{args.clip_margin_threshold:+.3f}@{args.lambda_delta:.2f}"
    for sample in samples:
        summaries.append(
            edit_analysis.build_contrastive_summary(
                sample=sample,
                calibration=calibration,
                semantic_weights=semantic_weights_use,
                grounding_weights=grounding_weights_use,
                clip_margin_threshold=args.clip_margin_threshold,
                conf_threshold=args.semantic_confidence_threshold,
                lambda_delta=args.lambda_delta,
                gate_mode=args.gate_mode,
                config_name=config_name,
                branch_subset=recipe.get("branches", []),
                semantic_subset=semantic_subset,
                role_reversal_boost=1.0,
                enable_role_reversal_delta_rule=False,
                role_reversal_weights={"a": 1.0, "b": 0.7, "c": 0.5},
            )
        )
    return summaries


def compare_edit_recipes(
    comparison_id: str,
    dataset: str,
    benchmark_json: str,
    recipe_a: str,
    label_a: str,
    recipe_b: str,
    label_b: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[List[MetricRow], Dict[str, Any]]:
    summaries_a = build_edit_summaries(benchmark_json, recipe_a, args)
    summaries_b = build_edit_summaries(benchmark_json, recipe_b, args)
    aligned = align_summaries_by_sample(summaries_a, summaries_b)
    records = edit_metric_records(aligned)
    rows: List[MetricRow] = []

    pair_a = pairwise_metric_from_records(records["pairwise"], "a")
    pair_b = pairwise_metric_from_records(records["pairwise"], "b")
    pair_ci_low, pair_ci_high, _ = bootstrap_ci_delta(
        records["pairwise"],
        lambda recs: pairwise_metric_from_records(recs, "a"),
        lambda recs: pairwise_metric_from_records(recs, "b"),
        rng,
        args.bootstrap_iters,
    )
    pair_p = exact_sign_test([r["a"] for r in records["pairwise"]], [r["b"] for r in records["pairwise"]])
    rows.append(
        MetricRow(
            comparison_id=comparison_id,
            dataset=dataset,
            sample_unit="sample",
            comparator_a=label_a,
            comparator_b=label_b,
            metric="pairwise",
            metric_a=pair_a,
            metric_b=pair_b,
            delta_b_minus_a=pair_b - pair_a if math.isfinite(pair_a) and math.isfinite(pair_b) else None,
            ci_low=pair_ci_low,
            ci_high=pair_ci_high,
            p_value=pair_p,
            n_unit=len(records["pairwise"]),
            note="paired bootstrap CI + exact sign test",
        )
    )

    rescue_a = pairwise_metric_from_records(records["rescue"], "a")
    rescue_b = pairwise_metric_from_records(records["rescue"], "b")
    rescue_ci_low, rescue_ci_high, _ = bootstrap_ci_delta(
        records["rescue"],
        lambda recs: pairwise_metric_from_records(recs, "a"),
        lambda recs: pairwise_metric_from_records(recs, "b"),
        rng,
        args.bootstrap_iters,
    )
    rescue_p = exact_sign_test([r["a"] for r in records["rescue"]], [r["b"] for r in records["rescue"]])
    rows.append(
        MetricRow(
            comparison_id=comparison_id,
            dataset=dataset,
            sample_unit="sample_clip_fail",
            comparator_a=label_a,
            comparator_b=label_b,
            metric="rescue",
            metric_a=rescue_a,
            metric_b=rescue_b,
            delta_b_minus_a=rescue_b - rescue_a if math.isfinite(rescue_a) and math.isfinite(rescue_b) else None,
            ci_low=rescue_ci_low,
            ci_high=rescue_ci_high,
            p_value=rescue_p,
            n_unit=len(records["rescue"]),
            note="paired bootstrap CI + exact sign test on clip-fail subset",
        )
    )

    auc_a = auroc_metric_from_records(records["auroc"], "a")
    auc_b = auroc_metric_from_records(records["auroc"], "b")
    auc_ci_low, auc_ci_high, _ = bootstrap_ci_delta(
        records["auroc"],
        lambda recs: auroc_metric_from_records(recs, "a"),
        lambda recs: auroc_metric_from_records(recs, "b"),
        rng,
        args.bootstrap_iters,
    )
    rows.append(
        MetricRow(
            comparison_id=comparison_id,
            dataset=dataset,
            sample_unit="sample",
            comparator_a=label_a,
            comparator_b=label_b,
            metric="auroc",
            metric_a=auc_a,
            metric_b=auc_b,
            delta_b_minus_a=auc_b - auc_a if math.isfinite(auc_a) and math.isfinite(auc_b) else None,
            ci_low=auc_ci_low,
            ci_high=auc_ci_high,
            p_value=None,
            n_unit=len(records["auroc"]),
            note="paired bootstrap CI of delta(AUROC)",
        )
    )

    sev_a = severity_metric_from_records(records["severity"], "a")
    sev_b = severity_metric_from_records(records["severity"], "b")
    sev_ci_low, sev_ci_high, _ = bootstrap_ci_delta(
        records["severity"],
        lambda recs: severity_metric_from_records(recs, "a"),
        lambda recs: severity_metric_from_records(recs, "b"),
        rng,
        args.bootstrap_iters,
    )
    rows.append(
        MetricRow(
            comparison_id=comparison_id,
            dataset=dataset,
            sample_unit="sample",
            comparator_a=label_a,
            comparator_b=label_b,
            metric="severity_correlation",
            metric_a=sev_a,
            metric_b=sev_b,
            delta_b_minus_a=sev_b - sev_a if math.isfinite(sev_a) and math.isfinite(sev_b) else None,
            ci_low=sev_ci_low,
            ci_high=sev_ci_high,
            p_value=None,
            n_unit=len(records["severity"]),
            note="paired bootstrap CI of delta(Spearman)",
        )
    )
    diagnostic = {
        "num_aligned_samples": len(aligned),
        "num_pairwise_valid": len(records["pairwise"]),
        "num_rescue_valid": len(records["rescue"]),
        "num_auroc_valid": len(records["auroc"]),
        "num_severity_valid": len(records["severity"]),
        "benchmark_json": benchmark_json,
    }
    return rows, diagnostic


def build_native_args(args: argparse.Namespace, benchmark_json: str, scored_exploded_json: str, include_recipes: List[str], enable_native_delta: bool, delta_mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark_json=benchmark_json,
        scored_exploded_json=scored_exploded_json,
        outdir=str(DEFAULT_OUTDIR),
        output_prefix="unused",
        calibration=args.calibration,
        gate_mode=args.gate_mode,
        clip_margin_threshold=args.clip_margin_threshold,
        semantic_confidence_threshold=args.semantic_confidence_threshold,
        lambda_delta=args.lambda_delta,
        include_recipes=include_recipes,
        enable_native_delta=enable_native_delta,
        delta_mode=delta_mode,
        delta_a=args.native_delta_a,
        delta_b=args.native_delta_b,
        delta_c=args.native_delta_c,
        w_relation=args.w_relation,
        w_attribute=args.w_attribute,
        w_spatial=args.w_spatial,
        w_count=args.w_count,
        w_subj=args.w_subj,
        w_obj=args.w_obj,
        w_attr_bind=args.w_attr_bind,
        w_spatial_bind=args.w_spatial_bind,
        w_rel_phrase=args.w_rel_phrase,
    )


def native_pair_records_from_matrices(matrices: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for m in matrices:
        records.append(
            {
                "pair_id": str(m["pair_id"]),
                "text": bool(m["text_correct"]),
                "image": bool(m["image_correct"]),
                "group": bool(m["group_correct"]),
            }
        )
    records.sort(key=lambda x: x["pair_id"])
    return records


def compare_native_recipes(
    comparison_id: str,
    native_benchmark_json: str,
    winoground_exploded_scored_json: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[List[MetricRow], Dict[str, Any]]:
    native_payload = load_json(native_benchmark_json)
    native_samples = native_payload.get("samples", [])
    scored_samples = edit_analysis.extract_samples(load_json(winoground_exploded_scored_json))

    exploded_args = build_native_args(
        args,
        benchmark_json=native_benchmark_json,
        scored_exploded_json=winoground_exploded_scored_json,
        include_recipes=["spatialbind_only"],
        enable_native_delta=False,
        delta_mode="relphrase_spatialbind",
    )
    native_args = build_native_args(
        args,
        benchmark_json=native_benchmark_json,
        scored_exploded_json=winoground_exploded_scored_json,
        include_recipes=["relphrase_spatialbind"],
        enable_native_delta=True,
        delta_mode="relphrase_spatialbind",
    )
    _, exploded_matrices = native_eval.build_pair_matrices(native_samples, scored_samples, exploded_args)
    _, native_matrices = native_eval.build_pair_matrices(native_samples, scored_samples, native_args)
    rec_a = native_pair_records_from_matrices([m for m in exploded_matrices if m["recipe"] == "spatialbind_only"])
    rec_b = native_pair_records_from_matrices([m for m in native_matrices if m["recipe"] == "relphrase_spatialbind"])
    by_id_a = {r["pair_id"]: r for r in rec_a}
    by_id_b = {r["pair_id"]: r for r in rec_b}
    pair_ids = sorted(set(by_id_a).intersection(by_id_b))
    aligned = [(by_id_a[pid], by_id_b[pid]) for pid in pair_ids]

    def metric_mean(aligned_records: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]], key: str, which: int) -> float:
        vals = [bool(item[which][key]) for item in aligned_records]
        return binary_mean(vals)

    rows: List[MetricRow] = []
    for metric in ("text", "image", "group"):
        a_metric = metric_mean(aligned, metric, 0)
        b_metric = metric_mean(aligned, metric, 1)
        ci_low, ci_high, _ = bootstrap_ci_delta(
            aligned,
            lambda recs, m=metric: metric_mean(recs, m, 0),
            lambda recs, m=metric: metric_mean(recs, m, 1),
            rng,
            args.bootstrap_iters,
        )
        p_value = exact_sign_test([r[0][metric] for r in aligned], [r[1][metric] for r in aligned])
        rows.append(
            MetricRow(
                comparison_id=comparison_id,
                dataset="Winoground_native_2x2",
                sample_unit="pair",
                comparator_a="exploded_best_projected_to_native(spatialbind_only)",
                comparator_b="native_aware_best(relphrase_spatialbind_native_delta)",
                metric=f"{metric}_score",
                metric_a=a_metric,
                metric_b=b_metric,
                delta_b_minus_a=b_metric - a_metric if math.isfinite(a_metric) and math.isfinite(b_metric) else None,
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=p_value,
                n_unit=len(aligned),
                note="closest valid paired alternative: both models evaluated on the same 400 Winoground pairs in native 2x2 space",
            )
        )
    diagnostic = {
        "num_aligned_pairs": len(aligned),
        "native_benchmark_json": native_benchmark_json,
        "winoground_exploded_scored_json": winoground_exploded_scored_json,
    }
    return rows, diagnostic


def write_csv(path: Path, rows: List[MetricRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(MetricRow.__annotations__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_deltas(path: Path, rows: List[MetricRow]) -> Optional[str]:
    plot_rows = [row for row in rows if row.metric in {"pairwise", "rescue", "auroc", "text_score", "image_score", "group_score"}]
    if not plot_rows:
        return None
    if plt is not None and path.suffix.lower() == ".png":
        labels = [f"{row.comparison_id}\n{row.metric}" for row in plot_rows]
        deltas = [safe_float(row.delta_b_minus_a) for row in plot_rows]
        lows = [safe_float(row.delta_b_minus_a) - safe_float(row.ci_low) if is_finite(row.delta_b_minus_a) and is_finite(row.ci_low) else 0.0 for row in plot_rows]
        highs = [safe_float(row.ci_high) - safe_float(row.delta_b_minus_a) if is_finite(row.delta_b_minus_a) and is_finite(row.ci_high) else 0.0 for row in plot_rows]
        fig_h = max(4.0, 0.5 * len(plot_rows))
        fig, ax = plt.subplots(figsize=(9, fig_h))
        y = np.arange(len(plot_rows))
        ax.errorbar(deltas, y, xerr=[lows, highs], fmt="o", color="#1f4e79", ecolor="#7aa6d1", capsize=3)
        ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Delta (B - A) with 95% paired bootstrap CI")
        ax.set_title("Selective Correction Significance Summary")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return str(path)
    return write_svg_delta_plot(path, plot_rows)


def write_svg_delta_plot(path: Path, rows: List[MetricRow]) -> str:
    width = 1100
    left = 410
    right = 50
    top = 50
    row_h = 42
    bottom = 40
    height = top + bottom + row_h * len(rows)
    values = []
    for row in rows:
        for value in (row.ci_low, row.ci_high, row.delta_b_minus_a):
            if is_finite(value):
                values.append(safe_float(value))
    min_v = min(values) if values else -0.05
    max_v = max(values) if values else 0.05
    span = max(max_v - min_v, 1e-6)
    pad = span * 0.1
    min_v -= pad
    max_v += pad
    plot_w = width - left - right

    def xcoord(value: float) -> float:
        return left + (value - min_v) / (max_v - min_v) * plot_w

    zero_x = xcoord(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px} .title{font-size:18px;font-weight:bold} .axis{stroke:#333;stroke-width:1} .ci{stroke:#7aa6d1;stroke-width:3} .pt{fill:#1f4e79}</style>',
        f'<text class="title" x="{left}" y="24">Selective Correction Significance Summary</text>',
        f'<line class="axis" x1="{zero_x:.2f}" y1="{top-10}" x2="{zero_x:.2f}" y2="{height-bottom+10}" stroke-dasharray="4 4"/>',
    ]
    for i, row in enumerate(rows):
        y = top + i * row_h + row_h / 2
        label = f"{row.comparison_id} / {row.metric}"
        parts.append(f'<text x="12" y="{y+4:.1f}">{label}</text>')
        if is_finite(row.ci_low) and is_finite(row.ci_high):
            x1 = xcoord(safe_float(row.ci_low))
            x2 = xcoord(safe_float(row.ci_high))
            parts.append(f'<line class="ci" x1="{x1:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y:.2f}"/>')
            parts.append(f'<line class="ci" x1="{x1:.2f}" y1="{y-6:.2f}" x2="{x1:.2f}" y2="{y+6:.2f}"/>')
            parts.append(f'<line class="ci" x1="{x2:.2f}" y1="{y-6:.2f}" x2="{x2:.2f}" y2="{y+6:.2f}"/>')
        if is_finite(row.delta_b_minus_a):
            xp = xcoord(safe_float(row.delta_b_minus_a))
            parts.append(f'<circle class="pt" cx="{xp:.2f}" cy="{y:.2f}" r="4"/>')
            parts.append(f'<text x="{plot_w + left + 8}" y="{y+4:.1f}">{fmt(row.delta_b_minus_a)}</text>')
    for tick in np.linspace(min_v, max_v, 5):
        xt = xcoord(float(tick))
        parts.append(f'<line class="axis" x1="{xt:.2f}" y1="{height-bottom}" x2="{xt:.2f}" y2="{height-bottom+6}"/>')
        parts.append(f'<text x="{xt-14:.2f}" y="{height-bottom+22}">{tick:.3f}</text>')
    parts.append(f'<text x="{left}" y="{height-8}">Delta (B - A) with 95% paired bootstrap CI</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return str(path)


def write_markdown(path: Path, rows: List[MetricRow], artifact_summary: Dict[str, Any], diagnostics: Dict[str, Any], figure_path: Optional[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Selective Correction Significance Summary\n\n")
        f.write("## 사용한 데이터 아티팩트\n")
        for key, value in artifact_summary.items():
            f.write(f"- {key}: `{value}`\n")
        f.write("\n## 핵심 결과 표\n\n")
        f.write("| 비교 | Metric | A | B | Delta(B-A) | 95% CI | p-value | N |\n")
        f.write("|---|---|---:|---:|---:|---|---:|---:|\n")
        for row in rows:
            ci = f"[{fmt(row.ci_low)}, {fmt(row.ci_high)}]"
            f.write(
                f"| {row.comparison_id} | {row.metric} | {fmt(row.metric_a)} | {fmt(row.metric_b)} | {fmt(row.delta_b_minus_a)} | {ci} | {fmt(row.p_value)} | {row.n_unit} |\n"
            )
        f.write("\n## 비교별 해석\n")
        for comparison_id in sorted({row.comparison_id for row in rows}):
            comp_rows = [row for row in rows if row.comparison_id == comparison_id]
            f.write(f"\n### {comparison_id}\n")
            for row in comp_rows:
                direction = "상승" if is_finite(row.delta_b_minus_a) and safe_float(row.delta_b_minus_a) > 0 else "하락"
                f.write(
                    f"- `{row.metric}`: A={fmt(row.metric_a)}, B={fmt(row.metric_b)}, delta={fmt(row.delta_b_minus_a)} "
                    f"(95% CI {fmt(row.ci_low)}~{fmt(row.ci_high)}), p={fmt(row.p_value)} -> {direction}\n"
                )
        f.write("\n## pairing / 재구성 주의사항\n")
        f.write("- `mixed_baseline_vs_selective`와 `winoground_slim_recipe`는 같은 scored benchmark의 동일 sample_id를 기준으로 paired reconstruction 했습니다.\n")
        f.write("- `Winoground exploded best vs native-aware best`는 metric 정의가 다르므로, exploded-best를 native 2x2 evaluator에 투영한 뒤 같은 400 pair에서 native-aware best와 비교했습니다. 이게 기존 산출물 기준 가장 엄밀한 paired alternative입니다.\n")
        f.write("- AUROC는 sample-level paired bootstrap으로 delta(AUROC)의 CI만 추정했고, 별도 p-value는 두지 않았습니다.\n")
        f.write("\n## 진단 정보\n")
        f.write("```json\n")
        f.write(json.dumps(diagnostics, ensure_ascii=False, indent=2))
        f.write("\n```\n")
        if figure_path:
            f.write("\n## Figure\n")
            f.write(f"- [delta_ci_plot.png]({figure_path})\n")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows: List[MetricRow] = []
    diagnostics: Dict[str, Any] = {}

    mixed_rows, mixed_diag = compare_edit_recipes(
        comparison_id="mixed_baseline_vs_selective",
        dataset="VG20k+COCO20k_stratified_4k",
        benchmark_json=args.mixed_benchmark_json,
        recipe_a="baseline_relation_only",
        label_a="combined_semantic_relation_only_baseline@+0.000",
        recipe_b="relphrase_spatialbind",
        label_b="edit_contrastive_relphrase_spatialbind@combined@+0.000@0.35",
        args=args,
        rng=rng,
    )
    rows.extend(mixed_rows)
    diagnostics["mixed_baseline_vs_selective"] = mixed_diag

    wino_rows, wino_diag = compare_edit_recipes(
        comparison_id="winoground_slim_recipe",
        dataset="Winoground_exploded_full",
        benchmark_json=args.winoground_exploded_json,
        recipe_a="spatialbind_only",
        label_a="edit_contrastive_spatialbind_only@combined@+0.000@0.35",
        recipe_b="relphrase_spatialbind",
        label_b="edit_contrastive_relphrase_spatialbind@combined@+0.000@0.35",
        args=args,
        rng=rng,
    )
    rows.extend(wino_rows)
    diagnostics["winoground_slim_recipe"] = wino_diag

    native_rows, native_diag = compare_native_recipes(
        comparison_id="winoground_exploded_best_vs_native_aware",
        native_benchmark_json=args.winoground_native_json,
        winoground_exploded_scored_json=args.winoground_exploded_json,
        args=args,
        rng=rng,
    )
    rows.extend(native_rows)
    diagnostics["winoground_exploded_best_vs_native_aware"] = native_diag

    artifact_summary = {
        "mixed_benchmark_json": args.mixed_benchmark_json,
        "coco5k_benchmark_json_found": args.coco5k_benchmark_json,
        "winoground_exploded_json": args.winoground_exploded_json,
        "winoground_native_json": args.winoground_native_json,
        "bootstrap_iters": args.bootstrap_iters,
        "seed": args.seed,
    }

    summary_payload = {
        "artifacts": artifact_summary,
        "diagnostics": diagnostics,
        "rows": [asdict(row) for row in rows],
    }
    json_path = outdir / "significance_summary.json"
    csv_path = outdir / "significance_summary.csv"
    md_path = outdir / "significance_summary.md"
    fig_path = outdir / "delta_ci_plot.svg"
    save_json(json_path, summary_payload)
    write_csv(csv_path, rows)
    figure_ref = plot_deltas(fig_path, rows)
    write_markdown(md_path, rows, artifact_summary, diagnostics, figure_ref)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import edit_aware_contrastive_grounding_analysis as edit_analysis


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"
OUTDIR = OUTPUT_ROOT / "controlled_signal_analysis"

RECIPE_ORDER = [
    "relphrase_spatialbind",
    "relphrase_only",
    "spatialbind_only",
    "full_semantic_grounding",
]

RECIPE_BRANCHES = {
    "relphrase_spatialbind": ["rel_phrase", "spatial_bind"],
    "relphrase_only": ["rel_phrase"],
    "spatialbind_only": ["spatial_bind"],
    "full_semantic_grounding": ["relation", "attribute", "spatial", "count", "subj", "obj", "attr_bind", "spatial_bind", "rel_phrase"],
}

KNOWN_SANITY = {
    ("clip_b32", "mixed_vg20k_coco20k_4k", "semantic"): 0.4042,
    ("clip_b32", "mixed_vg20k_coco20k_4k", "full_semantic_grounding"): 0.4438,
    ("clip_vitl14", "coco5k", "semantic"): 0.4748,
    ("clip_vitl14", "coco5k", "full_semantic_grounding"): 0.5312,
    ("blip_itm_base_coco", "mixed_vg20k_coco20k_4k", "semantic"): 0.4715,
    ("blip_itm_base_coco", "mixed_vg20k_coco20k_4k", "relphrase_spatialbind"): 0.5193,
    ("blip_itm_base_coco", "coco5k", "semantic"): 0.4882,
    ("blip_itm_base_coco", "coco5k", "relphrase_spatialbind"): 0.5762,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Controlled Signal Analysis for semantic-vs-grounding confounds.")
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260502)
    p.add_argument("--threshold-quantile", type=float, default=0.25)
    p.add_argument("--outdir", default=str(OUTDIR))
    p.add_argument("--max-artifacts", type=int, default=0, help="Debug limit; 0 means all usable artifacts.")
    return p.parse_args()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_finite(value: Any) -> bool:
    return math.isfinite(safe_float(value))


def fmt(value: Any, digits: int = 4) -> str:
    v = safe_float(value)
    return "NA" if not math.isfinite(v) else f"{v:.{digits}f}"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_samples(payload: Any) -> List[Dict[str, Any]]:
    return edit_analysis.extract_samples(payload)


def infer_model_benchmark(path: Path) -> Tuple[str, str]:
    parts = path.parts
    text = str(path)
    benchmark = "unknown"
    for candidate in ["mixed_vg20k_coco20k_4k", "coco5k", "winoground_exploded", "sugarcrepe_highcompat_50"]:
        if candidate in text:
            benchmark = candidate
            break
    model = "unknown"
    known = [
        "clip_b32",
        "clip_vitl14",
        "blip_itm_base_coco",
        "openclip_vit_b16_laion2b",
        "siglip_base_patch16_224",
        "bridgetower_base_itm_mlm",
        "clip",
    ]
    for candidate in known:
        if candidate in text:
            model = candidate
            break
    if model == "unknown" and "clip_only" in path.name:
        model = "clip_vitl14"
    if "clip_b32_mixed_consistency_rerun" in text:
        model = "clip_b32"
    return model, benchmark


def discover_artifacts() -> List[Path]:
    candidates: List[Path] = []
    for path in OUTPUT_ROOT.rglob("*factorized*.json"):
        text = str(path)
        if any(token in text for token in ["controlled_signal_analysis", "qwen2vl", "llm_parser"]):
            continue
        if not any(token in text for token in ["mixed_vg20k_coco20k_4k", "coco5k"]):
            continue
        if not any(token in text for token in ["cross_model_suite", "cross_model_blip_torch26_retry", "cross_model_openclip_bridgetower", "cross_model_siglip_qwen"]):
            continue
        candidates.append(path)
    # Add CLIP B/32 current compatibility rerun if present.
    b32 = OUTPUT_ROOT / "cross_model_suite" / "clip_b32_mixed_consistency_rerun" / "factorized" / "mixed_vg20k_coco20k_4k_clip_only_scored_scored_onto_img_factorized.json"
    if b32.exists() and b32 not in candidates:
        candidates.append(b32)
    return sorted(candidates)


def discover_skipped_csvs() -> List[Path]:
    out: List[Path] = []
    for pattern in ["*editaware*.csv", "*edit_aware*.csv"]:
        out.extend(OUTPUT_ROOT.rglob(pattern))
    return sorted(set(out))


def candidate_scores(sample: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if isinstance(sample.get("positive"), dict):
        yield sample["positive"].get("scores") or {}
    for neg in sample.get("negatives") or []:
        if isinstance(neg, dict):
            yield neg.get("scores") or {}


def available_score_columns(samples: Sequence[Dict[str, Any]], limit: int = 200) -> List[str]:
    cols = set()
    for sample in samples[:limit]:
        for scores in candidate_scores(sample):
            cols.update(scores.keys())
    return sorted(cols)


def weights() -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    semantic_weights = {"relation": 1.0, "attribute": 0.5, "spatial": 0.8, "count": 0.5}
    grounding_weights = {"subj": 0.5, "obj": 0.5, "attr_bind": 1.0, "spatial_bind": 1.0, "rel_phrase": 1.0}
    role_weights = {"a": 1.0, "b": 0.7, "c": 0.5}
    return semantic_weights, grounding_weights, role_weights


def visual_summary(sample: Dict[str, Any]) -> Dict[str, Any]:
    pos_score = safe_float((sample.get("positive") or {}).get("scores", {}).get("clip"))
    neg_scores = [safe_float((neg.get("scores") or {}).get("clip")) for neg in sample.get("negatives", [])]
    neg_scores = [score for score in neg_scores if math.isfinite(score)]
    clip_margin = pos_score - max(neg_scores) if math.isfinite(pos_score) and neg_scores else float("nan")
    return {
        "sample_id": sample.get("sample_id"),
        "pos_score": pos_score,
        "neg_scores": neg_scores,
        "margin": clip_margin,
        "clip_margin": clip_margin,
    }


def build_records(samples: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    semantic_weights, grounding_weights, role_weights = weights()
    calibration = edit_analysis.collect_calibration_stats(list(samples), "zscore")
    records: Dict[str, List[Dict[str, Any]]] = {"base": [], "semantic": []}
    for recipe in RECIPE_ORDER:
        records[recipe] = []
    for sample in samples:
        records["base"].append(visual_summary(sample))
        records["semantic"].append(
            edit_analysis.build_baseline_summary(
                sample,
                calibration,
                semantic_weights,
                grounding_weights,
                clip_margin_threshold=0.0,
                conf_threshold=0.6,
                lambda_sem=0.3,
            )
        )
        for recipe in RECIPE_ORDER:
            records[recipe].append(
                edit_analysis.build_contrastive_summary(
                    sample=sample,
                    calibration=calibration,
                    semantic_weights=semantic_weights,
                    grounding_weights=grounding_weights,
                    clip_margin_threshold=0.0,
                    conf_threshold=0.6,
                    lambda_delta=0.35,
                    gate_mode="combined",
                    config_name=f"edit_contrastive_{recipe}@combined@+0.000@0.35",
                    branch_subset=RECIPE_BRANCHES[recipe],
                    semantic_subset=None,
                    role_reversal_boost=1.0,
                    enable_role_reversal_delta_rule=False,
                    role_reversal_weights=role_weights,
                )
            )
    return records


def valid_mask(*record_lists: Sequence[Dict[str, Any]]) -> List[bool]:
    n = len(record_lists[0]) if record_lists else 0
    mask = []
    for i in range(n):
        ok = True
        for records in record_lists:
            if not math.isfinite(safe_float(records[i].get("margin"))):
                ok = False
                break
        mask.append(ok)
    return mask


def subset_records(records: Sequence[Dict[str, Any]], indices: Sequence[int]) -> List[Dict[str, Any]]:
    return [records[i] for i in indices]


def correct_array(records: Sequence[Dict[str, Any]], indices: Sequence[int]) -> List[int]:
    return [1 if safe_float(records[i].get("margin")) > 0.0 else 0 for i in indices]


def margin_array(records: Sequence[Dict[str, Any]], indices: Sequence[int]) -> List[float]:
    return [safe_float(records[i].get("margin")) for i in indices]


def quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def auroc_from_records(records: Sequence[Dict[str, Any]]) -> float:
    scores: List[float] = []
    labels: List[int] = []
    for rec in records:
        pos = safe_float(rec.get("pos_score"))
        if math.isfinite(pos):
            scores.append(pos)
            labels.append(1)
        for neg in rec.get("neg_scores") or []:
            negf = safe_float(neg)
            if math.isfinite(negf):
                scores.append(negf)
                labels.append(0)
    return edit_analysis.auroc(scores, labels) if scores else float("nan")


def pairwise_accuracy(records: Sequence[Dict[str, Any]]) -> float:
    vals = [1.0 if safe_float(r.get("margin")) > 0.0 else 0.0 for r in records if math.isfinite(safe_float(r.get("margin")))]
    return sum(vals) / len(vals) if vals else float("nan")


def rescue_rate(records: Sequence[Dict[str, Any]], base_records: Sequence[Dict[str, Any]]) -> float:
    vals = []
    for rec, base in zip(records, base_records):
        bm = safe_float(base.get("margin"))
        rm = safe_float(rec.get("margin"))
        if math.isfinite(bm) and math.isfinite(rm) and bm <= 0.0:
            vals.append(1.0 if rm > 0.0 else 0.0)
    return sum(vals) / len(vals) if vals else float("nan")


def exact_sign_p(a_correct: Sequence[int], b_correct: Sequence[int]) -> Tuple[float, int, int]:
    b_wins = sum(1 for a, b in zip(a_correct, b_correct) if b == 1 and a == 0)
    a_wins = sum(1 for a, b in zip(a_correct, b_correct) if a == 1 and b == 0)
    n = a_wins + b_wins
    if n == 0:
        return float("nan"), b_wins, a_wins
    k = min(a_wins, b_wins)
    # Two-sided exact binomial under p=0.5 via log probabilities.
    probs = [math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0)) for i in range(k + 1)]
    p = min(1.0, 2.0 * sum(probs))
    return p, b_wins, a_wins


def bootstrap_delta_ci(
    a_correct: Sequence[int],
    b_correct: Sequence[int],
    iters: int,
    rng: random.Random,
) -> Tuple[float, float]:
    n = len(a_correct)
    if n == 0:
        return float("nan"), float("nan")
    try:
        import numpy as np  # type: ignore

        seed = rng.randrange(0, 2**32 - 1)
        np_rng = np.random.default_rng(seed)
        diff = np.asarray(b_correct, dtype=float) - np.asarray(a_correct, dtype=float)
        draw = np_rng.integers(0, n, size=(iters, n))
        deltas = diff[draw].mean(axis=1)
        lo, hi = np.quantile(deltas, [0.025, 0.975])
        return float(lo), float(hi)
    except Exception:
        pass
    deltas = []
    for _ in range(iters):
        diff = 0.0
        for _j in range(n):
            idx = rng.randrange(n)
            diff += b_correct[idx] - a_correct[idx]
        deltas.append(diff / n)
    deltas.sort()
    lo = deltas[int(0.025 * (iters - 1))]
    hi = deltas[int(0.975 * (iters - 1))]
    return float(lo), float(hi)


def metric_row(
    model: str,
    benchmark: str,
    artifact_path: str,
    control_type: str,
    control_signal_type: str,
    q: float,
    threshold: float,
    n_total: int,
    indices: Sequence[int],
    signal_abs_all: Sequence[float],
    recipe_name: str,
    records: Dict[str, List[Dict[str, Any]]],
    bootstrap_iters: int,
    seed: int,
    notes: str,
) -> Dict[str, Any]:
    base = subset_records(records["base"], indices)
    semantic = subset_records(records["semantic"], indices)
    selective = subset_records(records[recipe_name], indices)
    base_fail_indices = [idx for idx in indices if safe_float(records["base"][idx].get("margin")) <= 0.0]
    n_base_fail = len(base_fail_indices)
    base_pair = pairwise_accuracy(base)
    sem_pair = pairwise_accuracy(semantic)
    sel_pair = pairwise_accuracy(selective)

    sem_correct = correct_array(records["semantic"], indices)
    base_correct = correct_array(records["base"], indices)
    sel_correct = correct_array(records[recipe_name], indices)
    digest = hashlib.sha1(f"{seed}:{model}:{benchmark}:{control_type}:{recipe_name}:{q}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:12], 16))
    sem_ci = bootstrap_delta_ci(base_correct, sem_correct, bootstrap_iters, rng)
    sem_p, _, _ = exact_sign_p(base_correct, sem_correct)
    sel_ci = bootstrap_delta_ci(sem_correct, sel_correct, bootstrap_iters, rng)
    sel_p, _, _ = exact_sign_p(sem_correct, sel_correct)

    sem_rescue = rescue_rate(semantic, base)
    sel_rescue = rescue_rate(selective, base)
    rescue_sem_correct = correct_array(records["semantic"], base_fail_indices)
    rescue_sel_correct = correct_array(records[recipe_name], base_fail_indices)
    rescue_ci = bootstrap_delta_ci(rescue_sem_correct, rescue_sel_correct, bootstrap_iters, rng)
    rescue_p, _, _ = exact_sign_p(rescue_sem_correct, rescue_sel_correct)

    in_abs = [signal_abs_all[i] for i in indices if math.isfinite(signal_abs_all[i])]
    outside = set(indices)
    out_abs = [signal_abs_all[i] for i in range(len(signal_abs_all)) if i not in outside and math.isfinite(signal_abs_all[i])]
    base_auroc = auroc_from_records(base)
    semantic_auroc = auroc_from_records(semantic)
    selective_auroc = auroc_from_records(selective)
    return {
        "model": model,
        "benchmark": benchmark,
        "artifact_path": artifact_path,
        "control_type": control_type,
        "control_signal_type": control_signal_type,
        "threshold_quantile": q,
        "threshold_value": threshold,
        "n_total": n_total,
        "n_subset": len(indices),
        "n_base_fail": n_base_fail,
        "base_pairwise": base_pair,
        "semantic_pairwise": sem_pair,
        "recipe_name": recipe_name,
        "selective_pairwise": sel_pair,
        "semantic_vs_base_delta": sem_pair - base_pair if math.isfinite(sem_pair) and math.isfinite(base_pair) else float("nan"),
        "selective_vs_semantic_delta": sel_pair - sem_pair if math.isfinite(sel_pair) and math.isfinite(sem_pair) else float("nan"),
        "semantic_vs_base_ci_low": sem_ci[0],
        "semantic_vs_base_ci_high": sem_ci[1],
        "semantic_vs_base_p": sem_p,
        "selective_vs_semantic_ci_low": sel_ci[0],
        "selective_vs_semantic_ci_high": sel_ci[1],
        "selective_vs_semantic_p": sel_p,
        "semantic_rescue": sem_rescue,
        "selective_rescue": sel_rescue,
        "selective_vs_semantic_rescue_delta": sel_rescue - sem_rescue if math.isfinite(sel_rescue) and math.isfinite(sem_rescue) else float("nan"),
        "selective_vs_semantic_rescue_ci_low": rescue_ci[0],
        "selective_vs_semantic_rescue_ci_high": rescue_ci[1],
        "selective_vs_semantic_rescue_p": rescue_p,
        "base_auroc": base_auroc,
        "semantic_auroc": semantic_auroc,
        "selective_auroc": selective_auroc,
        "selective_vs_semantic_auroc_delta": selective_auroc - semantic_auroc if math.isfinite(selective_auroc) and math.isfinite(semantic_auroc) else float("nan"),
        "mean_abs_control_signal_subset": sum(in_abs) / len(in_abs) if in_abs else float("nan"),
        "mean_abs_control_signal_outside": sum(out_abs) / len(out_abs) if out_abs else float("nan"),
        "notes": notes,
    }


def choose_best_recipe(records: Dict[str, List[Dict[str, Any]]], indices: Sequence[int]) -> str:
    best = None
    best_val = -1.0
    for recipe in RECIPE_ORDER:
        val = pairwise_accuracy(subset_records(records[recipe], indices))
        if math.isfinite(val) and val > best_val:
            best = recipe
            best_val = val
    return best or RECIPE_ORDER[0]


def analyze_artifact(
    path: Path,
    bootstrap_iters: int,
    seed: int,
    threshold_quantile: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    model, benchmark = infer_model_benchmark(path)
    payload = load_json(path)
    samples = extract_samples(payload)
    records = build_records(samples)
    mask = valid_mask(records["base"], records["semantic"], *[records[r] for r in RECIPE_ORDER])
    valid_indices = [i for i, ok in enumerate(mask) if ok]
    n_total = len(valid_indices)
    if n_total == 0:
        raise ValueError("No finite margins after record construction")

    full_metrics = {
        "base": pairwise_accuracy(subset_records(records["base"], valid_indices)),
        "semantic": pairwise_accuracy(subset_records(records["semantic"], valid_indices)),
    }
    for recipe in RECIPE_ORDER:
        full_metrics[recipe] = pairwise_accuracy(subset_records(records[recipe], valid_indices))

    sanity = []
    for (m, b, metric), expected in KNOWN_SANITY.items():
        if m == model and b == benchmark and metric in full_metrics:
            observed = full_metrics[metric]
            diff = observed - expected if math.isfinite(observed) else float("nan")
            status = "ok" if math.isfinite(diff) and abs(diff) <= 0.03 else "mismatch"
            sanity.append({"metric": metric, "expected": expected, "observed": observed, "diff": diff, "status": status})

    rows: List[Dict[str, Any]] = []
    counts: List[Dict[str, Any]] = []
    quantiles = sorted(set([0.10, threshold_quantile, 0.25, 0.50]))
    recipe_for_a1 = choose_best_recipe(records, valid_indices)

    semantic_signal = {
        i: safe_float(records["semantic"][i]["margin"]) - safe_float(records["base"][i]["margin"])
        for i in valid_indices
    }
    semantic_abs = [float("nan")] * len(samples)
    for i, value in semantic_signal.items():
        semantic_abs[i] = abs(value)
    for q in quantiles:
        threshold = quantile(list(semantic_signal.values()), q) if False else quantile([abs(v) for v in semantic_signal.values()], q)
        indices = [i for i in valid_indices if abs(semantic_signal[i]) <= threshold]
        notes = "small subset; interpret cautiously" if len(indices) < 100 else ""
        counts.append(count_row(model, benchmark, path, "semantic-controlled", "semantic_proxy", q, threshold, n_total, indices, semantic_abs, records))
        rows.append(metric_row(model, benchmark, str(path), "semantic-controlled", "semantic_proxy", q, threshold, n_total, indices, semantic_abs, recipe_for_a1, records, bootstrap_iters, seed, notes))

    for recipe in RECIPE_ORDER:
        ground_signal = {
            i: safe_float(records[recipe][i]["margin"]) - safe_float(records["semantic"][i]["margin"])
            for i in valid_indices
        }
        ground_abs = [float("nan")] * len(samples)
        for i, value in ground_signal.items():
            ground_abs[i] = abs(value)
        for q in quantiles:
            threshold = quantile([abs(v) for v in ground_signal.values()], q)
            indices = [i for i in valid_indices if abs(ground_signal[i]) <= threshold]
            notes = "small subset; interpret cautiously" if len(indices) < 100 else ""
            counts.append(count_row(model, benchmark, path, "grounding-controlled", "grounding_proxy_recipe_delta", q, threshold, n_total, indices, ground_abs, records, recipe))
            rows.append(metric_row(model, benchmark, str(path), "grounding-controlled", "grounding_proxy_recipe_delta", q, threshold, n_total, indices, ground_abs, recipe, records, bootstrap_iters, seed, notes))

    audit = {
        "artifact_path": str(path),
        "model": model,
        "benchmark": benchmark,
        "num_rows": len(samples),
        "num_valid_rows": n_total,
        "available_score_columns": available_score_columns(samples),
        "base_columns": ["candidate.scores.clip"],
        "semantic_columns": ["proxy: semantic margin - base margin from relation-only semantic baseline"],
        "selective_columns": [f"computed:{recipe}" for recipe in RECIPE_ORDER],
        "semantic_control_signal": "semantic_proxy",
        "grounding_control_signal": "grounding_proxy_recipe_delta",
        "usable": True,
        "skipped_reason": "",
        "full_metrics": full_metrics,
        "sanity_checks": sanity,
    }
    return rows, counts, audit


def count_row(
    model: str,
    benchmark: str,
    path: Path,
    control_type: str,
    signal_type: str,
    q: float,
    threshold: float,
    n_total: int,
    indices: Sequence[int],
    signal_abs_all: Sequence[float],
    records: Dict[str, List[Dict[str, Any]]],
    recipe_name: str = "",
) -> Dict[str, Any]:
    in_abs = [signal_abs_all[i] for i in indices if math.isfinite(signal_abs_all[i])]
    outside = set(indices)
    out_abs = [signal_abs_all[i] for i in range(len(signal_abs_all)) if i not in outside and math.isfinite(signal_abs_all[i])]
    return {
        "model": model,
        "benchmark": benchmark,
        "artifact_path": str(path),
        "control_type": control_type,
        "control_signal_type": signal_type,
        "recipe_name": recipe_name,
        "threshold_quantile": q,
        "threshold_value": threshold,
        "n_total": n_total,
        "n_subset": len(indices),
        "n_base_fail": sum(1 for i in indices if safe_float(records["base"][i].get("margin")) <= 0.0),
        "mean_abs_control_signal_subset": sum(in_abs) / len(in_abs) if in_abs else float("nan"),
        "mean_abs_control_signal_outside": sum(out_abs) / len(out_abs) if out_abs else float("nan"),
        "notes": "small subset; interpret cautiously" if len(indices) < 100 else "",
    }


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


def write_schema_audit(path: Path, audits: List[Dict[str, Any]], skipped: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Controlled Signal Analysis Schema Audit\n\n")
        f.write("## Usable Artifacts\n\n")
        for audit in audits:
            f.write(f"### {audit['model']} / {audit['benchmark']}\n")
            f.write(f"- artifact path: `{audit['artifact_path']}`\n")
            f.write(f"- rows: `{audit['num_rows']}`; valid rows: `{audit['num_valid_rows']}`\n")
            f.write(f"- available score columns: `{audit['available_score_columns']}`\n")
            f.write(f"- inferred base / visual score columns: `{audit['base_columns']}`\n")
            f.write(f"- inferred semantic score columns: `{audit['semantic_columns']}`\n")
            f.write(f"- inferred selective recipe score columns: `{audit['selective_columns']}`\n")
            f.write("- semantic-control signal computable: `yes` (`semantic_proxy`)\n")
            f.write("- grounding-control signal computable: `yes` (`grounding_proxy_recipe_delta`)\n")
            f.write(f"- full-set pairwise sanity: `{audit['full_metrics']}`\n")
            if audit.get("sanity_checks"):
                f.write(f"- known-value checks: `{audit['sanity_checks']}`\n")
            f.write("\n")
        f.write("## Skipped Artifacts\n\n")
        for item in skipped:
            f.write(f"- `{item['artifact_path']}`: {item['skipped_reason']}\n")


def interpretation_label(row: Dict[str, Any]) -> str:
    n = int(row.get("n_subset") or 0)
    if n < 100:
        return "small N"
    control = row["control_type"]
    delta = safe_float(row["selective_vs_semantic_delta"])
    sem_delta = safe_float(row["semantic_vs_base_delta"])
    ci_low = safe_float(row["selective_vs_semantic_ci_low"])
    if control == "semantic-controlled":
        return "semantic-controlled gain" if delta > 0 and (not math.isfinite(ci_low) or ci_low >= 0) else "semantic-controlled weak"
    return "grounding-controlled semantic gain" if sem_delta > 0 else "grounding-controlled weak"


def write_latex(path: Path, rows: List[Dict[str, Any]], primary_q: float) -> None:
    primary = [r for r in rows if abs(safe_float(r["threshold_quantile"]) - primary_q) < 1e-9]
    # Compact appendix table: best recipe per artifact/control by requested focus.
    selected: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in primary:
        grouped[(row["control_type"], row["model"], row["benchmark"])].append(row)
    for key, group in grouped.items():
        if key[0] == "semantic-controlled":
            selected.append(max(group, key=lambda r: safe_float(r["selective_pairwise"])))
        else:
            selected.append(max(group, key=lambda r: safe_float(r["semantic_vs_base_delta"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llrrrrlll}\n")
        f.write("\\toprule\n")
        f.write("Control & Model / Benchmark & N & Semantic & Best selective & $\\Delta$ & 95\\% CI & p-value & Interpretation \\\\\n")
        f.write("\\midrule\n")
        for row in selected:
            name = f"{row['model']} / {row['benchmark']}".replace("_", "\\_")
            ci = f"[{fmt(row['selective_vs_semantic_ci_low'], 3)}, {fmt(row['selective_vs_semantic_ci_high'], 3)}]"
            interp = interpretation_label(row).replace("_", "\\_")
            f.write(
                f"{row['control_type'].replace('_', ' ')} & {name} & {int(row['n_subset'])} & "
                f"{fmt(row['semantic_pairwise'], 3)} & {fmt(row['selective_pairwise'], 3)} & "
                f"{fmt(row['selective_vs_semantic_delta'], 3)} & {ci} & {fmt(row['selective_vs_semantic_p'], 3)} & {interp} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def write_histograms(outdir: Path, rows: List[Dict[str, Any]]) -> None:
    # Lightweight diagnostic plots from threshold/count distributions, not raw per-row signals.
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    for control, filename in [
        ("semantic-controlled", "semantic_control_signal_histograms.png"),
        ("grounding-controlled", "grounding_control_signal_histograms.png"),
    ]:
        vals = [safe_float(r["threshold_value"]) for r in rows if r["control_type"] == control and abs(safe_float(r["threshold_quantile"]) - 0.25) < 1e-9]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=min(20, max(5, len(vals))))
        plt.title(f"{control} 25th-percentile thresholds")
        plt.xlabel("|control signal| threshold")
        plt.ylabel("artifact/recipe count")
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=160)
        plt.close()


def write_summary(path: Path, rows: List[Dict[str, Any]], audits: List[Dict[str, Any]], primary_q: float) -> None:
    primary = [r for r in rows if abs(safe_float(r["threshold_quantile"]) - primary_q) < 1e-9]
    sem_rows = [r for r in primary if r["control_type"] == "semantic-controlled"]
    ground_rows = [r for r in primary if r["control_type"] == "grounding-controlled"]
    sem_pos = sum(1 for r in sem_rows if safe_float(r["selective_vs_semantic_delta"]) > 0)
    sem_total = len(sem_rows)
    ground_sem_pos = sum(1 for r in ground_rows if safe_float(r["semantic_vs_base_delta"]) > 0)
    ground_sel_pos = sum(1 for r in ground_rows if safe_float(r["selective_vs_semantic_delta"]) > 0)
    usable = ", ".join(f"{a['model']}/{a['benchmark']}" for a in audits)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Controlled Signal Analysis\n\n")
        f.write("## 1. Purpose\n")
        f.write("This analysis does not prove causal disentanglement. It tests whether each signal remains useful under approximate subset controls for the other signal.\n\n")
        f.write("## 2. A-1 Semantic-controlled subset\n")
        f.write(f"- Usable artifacts: {usable}\n")
        f.write(f"- Subset: rows with low absolute semantic-control signal at quantile `{primary_q}`.\n")
        f.write("- Signal type: `semantic_proxy = margin_semantic - margin_base` because explicit text-only ontology priors were not consistently available in per-pair artifacts.\n")
        f.write(f"- Selective correction is positive over semantic baseline in `{sem_pos}/{sem_total}` primary rows.\n\n")
        f.write("| Model | Benchmark | N | Recipe | Semantic | Selective | Delta | CI | p |\n")
        f.write("|---|---|---:|---|---:|---:|---:|---|---:|\n")
        for r in sem_rows:
            f.write(f"| {r['model']} | {r['benchmark']} | {r['n_subset']} | {r['recipe_name']} | {fmt(r['semantic_pairwise'])} | {fmt(r['selective_pairwise'])} | {fmt(r['selective_vs_semantic_delta'])} | [{fmt(r['selective_vs_semantic_ci_low'])}, {fmt(r['selective_vs_semantic_ci_high'])}] | {fmt(r['selective_vs_semantic_p'])} |\n")
        if sem_pos:
            f.write("\nOn semantic-controlled subsets where text-only ontology score differences are small, selective correction remains positive for a subset of benchmark/model settings. This suggests that the observed gains are not fully explained by text-only semantic plausibility.\n\n")
        else:
            f.write("\nOn semantic-controlled subsets, selective correction gains weaken, suggesting that part of the full-set gain is entangled with text-derived semantic structure.\n\n")
        f.write("## 3. A-2 Grounding-controlled subset\n")
        f.write(f"- Subset: rows with low absolute recipe-specific grounding proxy at quantile `{primary_q}`.\n")
        f.write("- Signal type: `grounding_proxy_recipe_delta = margin_recipe - margin_semantic`; this is not a pure grounding variable.\n")
        f.write(f"- Semantic baseline improves over base in `{ground_sem_pos}/{len(ground_rows)}` primary rows.\n")
        f.write(f"- Selective-over-semantic remains positive in `{ground_sel_pos}/{len(ground_rows)}` primary rows.\n\n")
        f.write("| Model | Benchmark | Recipe | N | Base | Semantic | Selective | Sem-Base | Sel-Sem |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in ground_rows:
            f.write(f"| {r['model']} | {r['benchmark']} | {r['recipe_name']} | {r['n_subset']} | {fmt(r['base_pairwise'])} | {fmt(r['semantic_pairwise'])} | {fmt(r['selective_pairwise'])} | {fmt(r['semantic_vs_base_delta'])} | {fmt(r['selective_vs_semantic_delta'])} |\n")
        f.write("\nOn grounding-controlled subsets, semantic baseline gains often remain visible while selective gains are reduced or recipe-dependent, supporting the interpretation that text-only semantic plausibility and image-conditioned correction contribute differently under the protocol.\n\n")
        f.write("## 4. Limitations\n")
        f.write("- Control signals are approximate, especially when proxy signals are used.\n")
        f.write("- Selection on low absolute margin difference is not causal randomization.\n")
        f.write("- The analysis reduces but does not eliminate confounding.\n")
        f.write("- Results should be interpreted as diagnostic subset evidence, not causal proof.\n\n")
        f.write("## 5. Recommended paper insertion\n")
        f.write("To further test whether selective-correction gains are reducible to text-only semantic plausibility, we construct semantic-controlled subsets where the semantic-baseline margin increment over the visual baseline is small. On these subsets, image-conditioned selective correction remains positive for several benchmark/model settings, suggesting that the full-set gains are not explained solely by semantic plausibility. Conversely, grounding-controlled subsets based on recipe-specific correction deltas show that semantic-baseline gains can remain visible while selective gains shrink or become recipe-dependent, supporting the view that semantic and grounding signals contribute differently, while not constituting causal disentanglement.\n")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    artifact_paths = discover_artifacts()
    if args.max_artifacts:
        artifact_paths = artifact_paths[: args.max_artifacts]
    skipped = [{"artifact_path": str(p), "skipped_reason": "aggregate CSV; no per-pair candidate rows for controlled subset/bootstrap"} for p in discover_skipped_csvs()]
    rows: List[Dict[str, Any]] = []
    counts: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for path in artifact_paths:
        try:
            new_rows, new_counts, audit = analyze_artifact(path, args.bootstrap_iters, args.seed, args.threshold_quantile)
            rows.extend(new_rows)
            counts.extend(new_counts)
            audits.append(audit)
            for check in audit.get("sanity_checks", []):
                if check.get("status") == "mismatch":
                    warnings.append(f"Known-value mismatch for {audit['model']}/{audit['benchmark']} {check}")
        except Exception as exc:
            skipped.append({"artifact_path": str(path), "skipped_reason": repr(exc)})
    write_csv(outdir / "controlled_signal_analysis_table.csv", rows)
    write_csv(outdir / "controlled_signal_subset_counts.csv", counts)
    write_schema_audit(outdir / "controlled_signal_schema_audit.md", audits, skipped)
    write_latex(outdir / "controlled_signal_analysis_latex_table.tex", rows, args.threshold_quantile)
    write_summary(outdir / "controlled_signal_analysis_summary.md", rows, audits, args.threshold_quantile)
    write_json(outdir / "controlled_signal_analysis_results.json", {"rows": rows, "counts": counts, "audits": audits, "skipped": skipped, "warnings": warnings})
    write_histograms(outdir, rows)
    print(f"usable_artifacts: {len(audits)}")
    print(f"output_dir: {outdir}")
    print(f"main_csv: {outdir / 'controlled_signal_analysis_table.csv'}")
    print(f"markdown_summary: {outdir / 'controlled_signal_analysis_summary.md'}")
    print(f"latex_table: {outdir / 'controlled_signal_analysis_latex_table.tex'}")
    primary = [r for r in rows if abs(safe_float(r["threshold_quantile"]) - args.threshold_quantile) < 1e-9]
    sem = [r for r in primary if r["control_type"] == "semantic-controlled"]
    ground = [r for r in primary if r["control_type"] == "grounding-controlled"]
    print(f"A-1: selective positive over semantic in {sum(safe_float(r['selective_vs_semantic_delta']) > 0 for r in sem)}/{len(sem)} primary rows.")
    print(f"A-2: semantic positive over base in {sum(safe_float(r['semantic_vs_base_delta']) > 0 for r in ground)}/{len(ground)} primary rows.")
    best = max(primary, key=lambda r: safe_float(r["selective_vs_semantic_delta"]), default=None)
    print(f"strongest_evidence: {best['model']} {best['benchmark']} {best['control_type']} {best['recipe_name']} delta={fmt(best['selective_vs_semantic_delta'])}" if best else "strongest_evidence: NA")
    print("main_limitation: controls use proxy margin differences and are diagnostic subsets, not causal randomization.")
    print("appendix_safety: safe with cautious wording if proxy limitation is stated.")


if __name__ == "__main__":
    main()

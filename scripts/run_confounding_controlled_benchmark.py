#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

import run_confounding_controlled_benchmark as v2


ROOT = Path(__file__).resolve().parent
V2_DIR = ROOT / "output" / "confounding_controlled_benchmark"
V3_DIR = V2_DIR / "v3"

V2_AUDIT_CSV = V2_DIR / "candidate_filter_audit.csv"
V2_SHORTLIST_BENCHMARK = V2_DIR / "confounding_controlled_benchmark_v2_pairwise_shortlist.json"
V2_CLIP_L14_SHORTLIST_FACTOR = (
    V2_DIR
    / "clip_vitl14"
    / "factorized"
    / "confounding_controlled_benchmark_v2_pairwise_shortlist_clip_vitl14_scored_scored_onto_img_factorized.json"
)

V3_EXPANDED_SHORTLIST_JSON = V3_DIR / "confounding_controlled_benchmark_v3_expanded_shortlist.json"
V3_RELAXED_BENCHMARK_JSON = V3_DIR / "confounding_controlled_benchmark_v3_relaxed_matched.json"
V3_STRICT_SAMPLE_IDS_JSON = V3_DIR / "confounding_controlled_benchmark_v3_strict_sample_ids.json"
V3_RELAXED_SAMPLE_IDS_JSON = V3_DIR / "confounding_controlled_benchmark_v3_relaxed_sample_ids.json"
V3_RESULTS_CSV = V3_DIR / "confounding_controlled_benchmark_v3_results.csv"
V3_SUMMARY_MD = V3_DIR / "confounding_controlled_benchmark_v3_summary.md"
V3_LATEX_TEX = V3_DIR / "confounding_controlled_benchmark_v3_latex_table.tex"
V3_AUDIT_CSV = V3_DIR / "v3_candidate_filter_audit.csv"
V3_SELECTION_AUDIT_MD = V3_DIR / "v3_selection_audit.md"

STRICT_NAME = "strict_matched"
RELAXED_NAME = "relaxed_matched"
NEGATIVE_TYPES = list(v2.NEGATIVE_TYPES)
ALL_TYPES = list(v2.ALL_TYPES)
AUDIT_RECONSTRUCTION_IMAGE_COUNT = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Construct confounding-controlled benchmark v3 from v2 artifacts.")
    p.add_argument("--max-images", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260502)
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--expanded-plausible-k", type=int, default=20)
    p.add_argument("--expanded-implausible-k", type=int, default=12)
    return p.parse_args()


def safe_float(value: Any) -> float:
    return v2.safe_float(value)


def is_finite(value: Any) -> bool:
    return v2.is_finite(value)


def fmt(value: Any, digits: int = 4) -> str:
    return v2.fmt(value, digits)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_env(gpu: int) -> Dict[str, str]:
    return v2.model_env(gpu)


def benchmark_stem(path: Path) -> str:
    return v2.slugify(path.stem)


def scored_json_path(model_cfg: Dict[str, Any], benchmark_json: Path) -> Path:
    return V3_DIR / model_cfg["model"] / f"{benchmark_stem(benchmark_json)}_{model_cfg['model']}_scored.json"


def factorized_json_path(model_cfg: Dict[str, Any], benchmark_json: Path) -> Path:
    return V3_DIR / model_cfg["model"] / "factorized" / f"{scored_json_path(model_cfg, benchmark_json).stem}_scored_onto_img_factorized.json"


def run_command(cmd: List[str], env: Dict[str, str]) -> None:
    print("RUN:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def score_with_model(benchmark_json: Path, model_cfg: Dict[str, Any], gpu: int, device: str) -> Path:
    env = model_env(gpu)
    scored_json = scored_json_path(model_cfg, benchmark_json)
    factorized_json = factorized_json_path(model_cfg, benchmark_json)
    scored_json.parent.mkdir(parents=True, exist_ok=True)
    factorized_json.parent.mkdir(parents=True, exist_ok=True)
    if factorized_json.exists():
        print(f"REUSE: {factorized_json}", flush=True)
        return factorized_json
    run_command(
        [
            sys.executable,
            str(ROOT / "score_semantic_violation_benchmark.py"),
            "--benchmark-json",
            str(benchmark_json),
            "--output-json",
            str(scored_json),
            "--scorers",
            "clip",
            "--clip-checkpoint",
            str(model_cfg["checkpoint"]),
            "--device",
            device,
            "--images-root",
            str(v2.PROJECT_ROOT),
        ],
        env=env,
    )
    run_command(
        [
            sys.executable,
            str(ROOT / "score_image_conditioned_semantic_plausibility.py"),
            "--benchmark-json",
            str(scored_json),
            "--ontology-json",
            str(v2.ONTOLOGY_JSON),
            "--predicate-frames-json",
            str(v2.PREDICATE_FRAMES_JSON),
            "--compatibility-npz",
            str(v2.COMPAT_NPZ),
            "--outdir",
            str(factorized_json.parent),
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
            str(model_cfg["checkpoint"]),
            "--images-root",
            str(v2.PROJECT_ROOT),
            "--device",
            device,
        ],
        env=env,
    )
    return factorized_json


def load_audit_rows() -> List[Dict[str, Any]]:
    return list(csv.DictReader(V2_AUDIT_CSV.open(encoding="utf-8")))


def assign_candidate_ids(image_rows: Sequence[Dict[str, Any]]) -> None:
    for row in image_rows:
        for neg_type, pool in row["candidate_pools"].items():
            for idx, candidate in enumerate(pool, 1):
                candidate["candidate_id"] = f"{row['sample_id']}__{neg_type}__cand{idx:03d}"
                candidate["candidate_index"] = idx


def build_candidate_row_index(image_rows: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[Dict[str, Any], str, Dict[str, Any]]]:
    index: Dict[str, Tuple[Dict[str, Any], str, Dict[str, Any]]] = {}
    for row in image_rows:
        for neg_type, pool in row["candidate_pools"].items():
            for candidate in pool:
                index[candidate["candidate_id"]] = (row, neg_type, candidate)
    return index


def verify_reconstruction(candidate_index: Dict[str, Tuple[Dict[str, Any], str, Dict[str, Any]]], audit_rows: Sequence[Dict[str, Any]]) -> None:
    checked = 0
    mismatched = 0
    for row in audit_rows[:500]:
        candidate_id = row["candidate_id"]
        triple = candidate_index.get(candidate_id)
        if triple is None:
            mismatched += 1
            continue
        _, _, candidate = triple
        if v2.normalize_text(candidate["candidate_caption"]) != v2.normalize_text(row["candidate_caption"]):
            mismatched += 1
        checked += 1
    if checked == 0 or mismatched > max(3, checked // 20):
        raise RuntimeError(f"candidate reconstruction mismatch is too high: checked={checked} mismatched={mismatched}")


def merge_audit_metrics(candidate_index: Dict[str, Tuple[Dict[str, Any], str, Dict[str, Any]]], audit_rows: Sequence[Dict[str, Any]]) -> None:
    for row in audit_rows:
        triple = candidate_index.get(row["candidate_id"])
        if triple is None:
            continue
        _, _, candidate = triple
        for key in [
            "base_margin",
            "semantic_gap",
            "grounding_delta_proxy",
            "lexical_overlap",
            "object_overlap",
            "changed_slots",
            "grammar_proxy",
            "selected_for_shortlist",
            "selected_final",
        ]:
            candidate[key] = row.get(key, "")


def choose_expanded_shortlist(
    image_rows: Sequence[Dict[str, Any]],
    plausible_k: int,
    implausible_k: int,
) -> Dict[str, Any]:
    counts = Counter()
    for row in image_rows:
        shortlist: Dict[str, List[Dict[str, Any]]] = {}
        for neg_type, pool in row["candidate_pools"].items():
            ranked: List[Dict[str, Any]]
            if neg_type == "text_plausible_image_wrong":
                ranked = sorted(
                    pool,
                    key=lambda candidate: (
                        0 if safe_float(candidate.get("object_overlap")) >= 1.0 else 1,
                        0 if safe_float(candidate.get("base_margin")) <= 0.10 else 1,
                        0 if safe_float(candidate.get("base_margin")) >= -0.05 else 1,
                        abs(safe_float(candidate.get("base_margin"))),
                        -safe_float(candidate.get("lexical_overlap")),
                        abs(safe_float(candidate.get("semantic_gap"))),
                    ),
                )[:plausible_k]
            elif neg_type == "text_implausible_image_wrong":
                ranked = sorted(
                    pool,
                    key=lambda candidate: (
                        0 if ("relation" not in str(candidate.get("changed_slots") or "") and "spatial" not in str(candidate.get("changed_slots") or "")) else 1,
                        -safe_float(candidate.get("semantic_gap")),
                        safe_float(candidate.get("grounding_delta_proxy")),
                        abs(safe_float(candidate.get("base_margin"))),
                        -safe_float(candidate.get("grammar_proxy")),
                    ),
                )[:implausible_k]
            else:
                ranked = list(pool)
            shortlist[neg_type] = ranked
            counts[neg_type] += len(ranked)
        row["screening_shortlist"] = shortlist
    return {"expanded_shortlist_counts": dict(counts)}


def method_summary_index(payload: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    samples = v2.edit_analysis.extract_samples(payload)
    return {
        method: {record["sample_id"]: record for record in records}
        for method, records in v2.build_method_summaries(samples).items()
    }


def enrich_shortlist_metrics(
    image_rows: Sequence[Dict[str, Any]],
    factorized_payload: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    method_index = method_summary_index(factorized_payload)
    for row in image_rows:
        for neg_type, pool in row["screening_shortlist"].items():
            for candidate in pool:
                sample_id = candidate["candidate_id"]
                candidate["margin_base"] = safe_float(method_index["visual_only"][sample_id]["margin"])
                candidate["margin_semantic"] = safe_float(method_index["semantic_baseline"][sample_id]["margin"])
                candidate["margin_relphrase_spatialbind"] = safe_float(method_index["relphrase_spatialbind"][sample_id]["margin"])
                candidate["margin_full"] = safe_float(method_index["full_semantic_grounding"][sample_id]["margin"])
                candidate["margin_full_minus_relphrase"] = safe_float(method_index["full_minus_relphrase"][sample_id]["margin"])
                candidate["margin_full_minus_spatialbind"] = safe_float(method_index["full_minus_spatialbind"][sample_id]["margin"])
                candidate["margin_full_minus_relphrase_spatialbind"] = safe_float(method_index["full_minus_relphrase_spatialbind"][sample_id]["margin"])
                candidate["semantic_increment"] = candidate["margin_semantic"] - candidate["margin_base"]
                candidate["selective_increment"] = candidate["margin_full"] - candidate["margin_semantic"]
                candidate["leaveout_delta"] = candidate["margin_full"] - candidate["margin_full_minus_relphrase_spatialbind"]
    return method_index


def quantile(values: Sequence[float], q: float, default: float) -> float:
    vals = [safe_float(value) for value in values if is_finite(value)]
    if not vals:
        return default
    return float(np.quantile(np.array(vals, dtype=float), q))


def compute_thresholds(image_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    plausible = [candidate for row in image_rows for candidate in row["screening_shortlist"]["text_plausible_image_wrong"]]
    implausible = [candidate for row in image_rows for candidate in row["screening_shortlist"]["text_implausible_image_wrong"]]
    relation = [candidate for row in image_rows for candidate in row["screening_shortlist"]["relation_spatial_wrong"]]
    positive_impl_sem = [safe_float(candidate.get("semantic_increment")) for candidate in implausible if safe_float(candidate.get("semantic_increment")) > 0.0]
    positive_impl_gap = [safe_float(candidate.get("semantic_gap")) for candidate in implausible if safe_float(candidate.get("semantic_gap")) > 0.0]
    return {
        "plausible_semantic_abs_q25": quantile([abs(safe_float(candidate.get("semantic_increment"))) for candidate in plausible], 0.25, 0.02),
        "plausible_lex_q75": quantile([safe_float(candidate.get("lexical_overlap")) for candidate in plausible], 0.75, 0.40),
        "implausible_semantic_inc_q75": quantile(positive_impl_sem, 0.75, 0.02),
        "implausible_semantic_gap_q75": quantile(positive_impl_gap, 0.75, 0.20),
        "implausible_ground_q50": quantile([safe_float(candidate.get("grounding_delta_proxy")) for candidate in implausible], 0.50, 0.02),
        "relation_leaveout_q75": quantile([safe_float(candidate.get("leaveout_delta")) for candidate in relation], 0.75, 0.05),
    }


def plausible_strict(candidate: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    return (
        -0.05 <= safe_float(candidate.get("margin_base")) <= 0.10
        and abs(safe_float(candidate.get("semantic_increment"))) <= min(0.02, thresholds["plausible_semantic_abs_q25"])
        and safe_float(candidate.get("lexical_overlap")) >= max(0.40, thresholds["plausible_lex_q75"] - 0.15)
        and safe_float(candidate.get("object_overlap")) >= 1.0
    )


def plausible_relaxed(candidate: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    return (
        safe_float(candidate.get("margin_base")) <= 0.10
        and abs(safe_float(candidate.get("semantic_increment"))) <= 0.02
        and safe_float(candidate.get("lexical_overlap")) >= 0.30
        and safe_float(candidate.get("object_overlap")) >= 1.0
    )


def implausible_no_relation_spatial(candidate: Dict[str, Any]) -> bool:
    changed = str(candidate.get("changed_slots") or "")
    return "relation" not in changed and "spatial" not in changed


def implausible_strict(candidate: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    sem_inc = safe_float(candidate.get("semantic_increment"))
    sem_gap = safe_float(candidate.get("semantic_gap"))
    ground = safe_float(candidate.get("grounding_delta_proxy"))
    return (
        implausible_no_relation_spatial(candidate)
        and ground <= min(0.02, max(0.0, thresholds["implausible_ground_q50"]))
        and (
            sem_inc >= max(0.02, thresholds["implausible_semantic_inc_q75"])
            or sem_gap >= max(0.20, thresholds["implausible_semantic_gap_q75"])
        )
        and safe_float(candidate.get("selective_increment")) <= max(0.02, sem_inc)
        and safe_float(candidate.get("margin_base")) <= 0.10
    )


def implausible_relaxed(candidate: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    sem_inc = safe_float(candidate.get("semantic_increment"))
    sem_gap = safe_float(candidate.get("semantic_gap"))
    return (
        (sem_inc > 0.0 or sem_gap > 0.0)
        and safe_float(candidate.get("grounding_delta_proxy")) <= 0.03
        and safe_float(candidate.get("selective_increment")) <= max(0.03, sem_inc + 0.01)
    )


def relation_allowed_slots(candidate: Dict[str, Any]) -> bool:
    changed = str(candidate.get("changed_slots") or "")
    has_target = any(token in changed for token in ["relation", "relation_direction", "spatial", "role_reversal"])
    untouched = not any(token in changed for token in ["subject", "object", "count", "attribute"])
    return has_target and untouched


def relation_strict(candidate: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    return (
        relation_allowed_slots(candidate)
        and safe_float(candidate.get("lexical_overlap")) >= 0.80
        and safe_float(candidate.get("leaveout_delta")) >= max(0.05, thresholds["relation_leaveout_q75"] * 0.5)
    )


def relation_relaxed(candidate: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    return relation_allowed_slots(candidate) and safe_float(candidate.get("lexical_overlap")) >= 0.80


def candidate_sort_key(candidate: Dict[str, Any], negative_type: str) -> Tuple[Any, ...]:
    if negative_type == "text_plausible_image_wrong":
        return (
            abs(safe_float(candidate.get("semantic_increment"))),
            abs(safe_float(candidate.get("margin_base"))),
            -safe_float(candidate.get("selective_increment")),
            -safe_float(candidate.get("lexical_overlap")),
        )
    if negative_type == "text_implausible_image_wrong":
        return (
            -safe_float(candidate.get("semantic_increment")),
            -safe_float(candidate.get("semantic_gap")),
            safe_float(candidate.get("selective_increment")),
            abs(safe_float(candidate.get("margin_base"))),
        )
    return (
        -safe_float(candidate.get("leaveout_delta")),
        -safe_float(candidate.get("selective_increment")),
        abs(safe_float(candidate.get("margin_base"))),
        -safe_float(candidate.get("lexical_overlap")),
    )


def choose_final_candidates(image_rows: Sequence[Dict[str, Any]], thresholds: Dict[str, float]) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    selected_by_type: Dict[str, List[Dict[str, Any]]] = {neg_type: [] for neg_type in NEGATIVE_TYPES}
    audit_rows: List[Dict[str, Any]] = []
    for row in image_rows:
        for neg_type, pool in row["screening_shortlist"].items():
            if neg_type == "text_plausible_image_wrong":
                strict_fn = plausible_strict
                relaxed_fn = plausible_relaxed
            elif neg_type == "text_implausible_image_wrong":
                strict_fn = implausible_strict
                relaxed_fn = implausible_relaxed
            else:
                strict_fn = relation_strict
                relaxed_fn = relation_relaxed
            ranked = sorted(
                pool,
                key=lambda candidate: (
                    0 if strict_fn(candidate, thresholds) else 1 if relaxed_fn(candidate, thresholds) else 2,
                    *candidate_sort_key(candidate, neg_type),
                ),
            )
            chosen = ranked[0]
            chosen["strict_eligible"] = strict_fn(chosen, thresholds)
            chosen["relaxed_eligible"] = relaxed_fn(chosen, thresholds)
            selected_by_type[neg_type].append(chosen)
            for candidate in pool:
                audit_rows.append(
                    {
                        "image_sample_id": row["sample_id"],
                        "negative_type": neg_type,
                        "candidate_id": candidate["candidate_id"],
                        "selected_v3_candidate": candidate["candidate_id"] == chosen["candidate_id"],
                        "strict_eligible": strict_fn(candidate, thresholds),
                        "relaxed_eligible": relaxed_fn(candidate, thresholds),
                        "margin_base": candidate.get("margin_base"),
                        "margin_semantic": candidate.get("margin_semantic"),
                        "margin_full": candidate.get("margin_full"),
                        "semantic_increment": candidate.get("semantic_increment"),
                        "selective_increment": candidate.get("selective_increment"),
                        "leaveout_delta": candidate.get("leaveout_delta"),
                        "lexical_overlap": candidate.get("lexical_overlap"),
                        "object_overlap": candidate.get("object_overlap"),
                        "changed_slots": candidate.get("changed_slots"),
                    }
                )
    return selected_by_type, audit_rows


def largest_matching_window(candidates: Sequence[Dict[str, Any]], target_low: float = 0.55, target_high: float = 0.65) -> Tuple[List[Dict[str, Any]], float]:
    items = sorted(candidates, key=lambda candidate: safe_float(candidate.get("margin_base")))
    if not items:
        return [], float("nan")
    positives = np.array([1 if safe_float(candidate.get("margin_base")) > 0.0 else 0 for candidate in items], dtype=int)
    prefix = np.concatenate([[0], np.cumsum(positives)])
    best: Tuple[int, int] = (0, len(items))
    best_score = (-1, float("inf"))
    for left in range(len(items)):
        for right in range(left + 1, len(items) + 1):
            n = right - left
            pos = int(prefix[right] - prefix[left])
            acc = pos / n
            if target_low <= acc <= target_high:
                score = (n, abs(acc - 0.60))
                if score > best_score:
                    best_score = score
                    best = (left, right)
    if best_score[0] < 0:
        return items, float(sum(positives) / len(positives))
    left, right = best
    kept = items[left:right]
    pos = int(prefix[right] - prefix[left])
    return kept, float(pos / len(kept))


def build_subset_maps(selected_by_type: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    strict_map: Dict[str, List[Dict[str, Any]]] = {}
    relaxed_map: Dict[str, List[Dict[str, Any]]] = {}
    for neg_type, candidates in selected_by_type.items():
        strict_candidates = [candidate for candidate in candidates if candidate.get("strict_eligible")]
        relaxed_candidates = [candidate for candidate in candidates if candidate.get("relaxed_eligible")]
        strict_map[neg_type], _ = largest_matching_window(strict_candidates)
        relaxed_map[neg_type], _ = largest_matching_window(relaxed_candidates)
    return {STRICT_NAME: strict_map, RELAXED_NAME: relaxed_map}


def sample_ids_from_subset(subset_map: Dict[str, List[Dict[str, Any]]]) -> Set[str]:
    return {candidate["candidate_id"] for candidates in subset_map.values() for candidate in candidates}


def subset_accuracy(candidates: Sequence[Dict[str, Any]]) -> float:
    if not candidates:
        return float("nan")
    return float(sum(1 for candidate in candidates if safe_float(candidate.get("margin_base")) > 0.0) / len(candidates))


def build_subset_report(subsets: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for subset_name, subset_map in subsets.items():
        subset_info: Dict[str, Any] = {"negative_types": {}}
        for neg_type, candidates in subset_map.items():
            subset_info["negative_types"][neg_type] = {
                "n": len(candidates),
                "base_pairwise": subset_accuracy(candidates),
            }
        report[subset_name] = subset_info
    return report


def build_benchmark_from_sample_ids(sample_map: Dict[str, Dict[str, Any]], sample_ids: Set[str]) -> Dict[str, Any]:
    samples = [sample_map[sample_id] for sample_id in sample_ids if sample_id in sample_map]
    samples.sort(key=lambda sample: sample["sample_id"])
    return {
        "metadata": {
            "version": "3.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": "pairwise_semantic_violation",
            "notes": "V3 confounding-controlled diagnostic benchmark built from v2 candidate artifacts and expanded CLIP-L/14 screening.",
            "num_samples": len(samples),
        },
        "samples": samples,
    }


def evaluate_subset_payload(
    model_cfg: Dict[str, Any],
    payload: Dict[str, Any],
    bootstrap_iters: int,
    seed: int,
    subset_name: str,
) -> List[Dict[str, Any]]:
    rows, _ = v2.evaluate_model_payload(model_cfg, payload, bootstrap_iters, seed)
    for row in rows:
        row["subset"] = subset_name
        row["best_selective_pairwise"] = float("nan")
        row["full_minus_relphrase_spatialbind_delta"] = float("nan")
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["negative_type"]].append(row)
    for negative_type, group_rows in grouped.items():
        by_method = {row["method"]: row for row in group_rows}
        best_selective = max(
            [row for row in group_rows if row["method"] in v2.PRIMARY_SELECTIVE_METHODS],
            key=lambda row: (safe_float(row["pairwise"]), safe_float(row["mean_margin"])),
        )
        label = interpret_condition(negative_type, by_method, best_selective)
        for row in group_rows:
            row["best_selective_pairwise"] = best_selective["pairwise"]
            row["full_minus_relphrase_spatialbind_delta"] = safe_float(by_method.get("full_semantic_grounding", {}).get("full_vs_leaveout_delta"))
            row["interpretation_label"] = label
    return rows


def interpret_condition(negative_type: str, by_method: Dict[str, Dict[str, Any]], best_selective: Dict[str, Any]) -> str:
    n = int(by_method.get("visual_only", {}).get("n") or 0)
    if n < 50:
        return "small N"
    base = safe_float(by_method.get("visual_only", {}).get("pairwise"))
    sem = safe_float(by_method.get("semantic_baseline", {}).get("pairwise"))
    sem_delta = safe_float(by_method.get("semantic_baseline", {}).get("semantic_vs_base_delta"))
    sel_delta = safe_float(best_selective.get("selective_vs_semantic_delta"))
    leave_delta = safe_float(by_method.get("full_semantic_grounding", {}).get("full_vs_leaveout_delta"))
    full_delta = safe_float(by_method.get("full_semantic_grounding", {}).get("selective_vs_semantic_delta"))
    if negative_type == "text_plausible_image_wrong":
        if 0.55 <= base <= 0.65 and sem_delta <= 0.02 and sel_delta > 0.0:
            return "clean image-conditioned condition"
        return "partial / entangled"
    if negative_type == "text_implausible_image_wrong":
        if sem_delta >= 0.05 and sel_delta < sem_delta:
            return "clean semantic condition"
        return "partial / entangled"
    if negative_type == "relation_spatial_wrong":
        if full_delta >= 0.10 and leave_delta >= 0.05:
            return "clean relation/spatial condition"
        return "partial / entangled"
    if base < sem < safe_float(best_selective.get("pairwise")):
        return "partial / entangled"
    return "partial / entangled"


def model_configs_for_v3() -> List[Dict[str, Any]]:
    return [
        cfg
        for cfg in v2.build_model_configs()
        if cfg["model"] in {"clip_vitl14", "clip_b32", "siglip_base_patch16_224", "bridgetower_base_itm_mlm"}
    ]


def write_selection_audit(
    thresholds: Dict[str, float],
    shortlist_info: Dict[str, Any],
    subset_report: Dict[str, Any],
    results_rows: Sequence[Dict[str, Any]],
) -> None:
    lines = [
        "# V3 Selection Audit",
        "",
        "## Expanded Shortlist",
        "",
        f"- expanded shortlist counts: `{shortlist_info['expanded_shortlist_counts']}`",
        "",
        "## Thresholds",
        "",
        f"- thresholds: `{thresholds}`",
        "",
        "## Matched Subsets",
        "",
        f"- subset report: `{subset_report}`",
        "",
        "## CLIP L/14 Snapshot",
        "",
    ]
    primary = [row for row in results_rows if row["model"] == "clip_vitl14"]
    for subset_name in [STRICT_NAME, RELAXED_NAME]:
        lines.append(f"### `{subset_name}`")
        for negative_type in ["text_implausible_image_wrong", "text_plausible_image_wrong", "relation_spatial_wrong", "combined"]:
            rows = [row for row in primary if row["subset"] == subset_name and row["negative_type"] == negative_type]
            if not rows:
                continue
            by_method = {row["method"]: row for row in rows}
            best_selective = max(
                [row for row in rows if row["method"] in v2.PRIMARY_SELECTIVE_METHODS],
                key=lambda row: (safe_float(row["pairwise"]), safe_float(row["mean_margin"])),
            )
            lines.extend(
                [
                    f"- {negative_type}: base `{fmt(by_method['visual_only']['pairwise'])}`, semantic `{fmt(by_method['semantic_baseline']['pairwise'])}`, best selective `{best_selective['method']}={fmt(best_selective['pairwise'])}`, interpretation `{best_selective['interpretation_label']}`",
                ]
            )
        lines.append("")
    V3_SELECTION_AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_latex_table(results_rows: Sequence[Dict[str, Any]]) -> None:
    primary = [row for row in results_rows if row["model"] == "clip_vitl14"]
    lines = [
        "\\begin{tabular}{llllllll}",
        "\\toprule",
        "Subset & Negative type & Base & Semantic & Best selective & $\\Delta$(sem-base) & $\\Delta$(sel-sem) & Interpretation \\\\",
        "\\midrule",
    ]
    for subset_name in [STRICT_NAME, RELAXED_NAME]:
        subset_rows = [row for row in primary if row["subset"] == subset_name]
        for negative_type in ["text_implausible_image_wrong", "text_plausible_image_wrong", "relation_spatial_wrong", "combined"]:
            rows = [row for row in subset_rows if row["negative_type"] == negative_type]
            if not rows:
                continue
            by_method = {row["method"]: row for row in rows}
            best_selective = max(
                [row for row in rows if row["method"] in v2.PRIMARY_SELECTIVE_METHODS],
                key=lambda row: (safe_float(row["pairwise"]), safe_float(row["mean_margin"])),
            )
            lines.append(
                " & ".join(
                    [
                        v2.latex_escape(subset_name),
                        v2.latex_escape(negative_type),
                        fmt(by_method["visual_only"]["pairwise"], 3),
                        fmt(by_method["semantic_baseline"]["pairwise"], 3),
                        v2.latex_escape(f"{best_selective['method']} ({fmt(best_selective['pairwise'], 3)})"),
                        fmt(by_method["semantic_baseline"]["semantic_vs_base_delta"], 3),
                        fmt(best_selective["selective_vs_semantic_delta"], 3),
                        v2.latex_escape(best_selective["interpretation_label"]),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    V3_LATEX_TEX.write_text("\n".join(lines), encoding="utf-8")


def benchmark_status(results_rows: Sequence[Dict[str, Any]]) -> str:
    primary = [row for row in results_rows if row["model"] == "clip_vitl14" and row["subset"] == RELAXED_NAME]
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in primary:
        grouped[row["negative_type"]][row["method"]] = row
    sem_ok = grouped.get("text_implausible_image_wrong", {}).get("full_semantic_grounding", {}).get("interpretation_label") == "clean semantic condition"
    img_ok = grouped.get("text_plausible_image_wrong", {}).get("full_semantic_grounding", {}).get("interpretation_label") == "clean image-conditioned condition"
    rel_ok = grouped.get("relation_spatial_wrong", {}).get("full_semantic_grounding", {}).get("interpretation_label") == "clean relation/spatial condition"
    if sem_ok and img_ok and rel_ok:
        return "strong enough for main text"
    if rel_ok or sem_ok or img_ok:
        return "appendix-only diagnostic"
    return "too noisy to include"


def write_summary(
    thresholds: Dict[str, float],
    shortlist_info: Dict[str, Any],
    subset_report: Dict[str, Any],
    results_rows: Sequence[Dict[str, Any]],
    models: Sequence[Dict[str, Any]],
) -> None:
    status = benchmark_status(results_rows)
    primary = [row for row in results_rows if row["model"] == "clip_vitl14"]
    lines = [
        "# Confounding-Controlled Diagnostic Benchmark V3 Summary",
        "",
        "## Purpose",
        "",
        "V3 reuses the v2 candidate artifacts, expands the CLIP L/14 shortlist where needed, and tightens candidate selection to better separate semantic-prior-heavy negatives, plausible cross-image negatives, and relation/spatial negatives. This remains an automatically derived diagnostic benchmark without human validation, and it does not justify any causal disentanglement claim.",
        "",
        "## Construction",
        "",
        f"- expanded shortlist counts: `{shortlist_info['expanded_shortlist_counts']}`",
        f"- thresholds: `{thresholds}`",
        f"- matched subset report: `{subset_report}`",
        "",
        "## CLIP L/14 Results",
        "",
    ]
    for subset_name in [STRICT_NAME, RELAXED_NAME]:
        lines.append(f"### `{subset_name}`")
        for negative_type in ["text_implausible_image_wrong", "text_plausible_image_wrong", "relation_spatial_wrong", "combined"]:
            rows = [row for row in primary if row["subset"] == subset_name and row["negative_type"] == negative_type]
            if not rows:
                continue
            by_method = {row["method"]: row for row in rows}
            best_selective = max(
                [row for row in rows if row["method"] in v2.PRIMARY_SELECTIVE_METHODS],
                key=lambda row: (safe_float(row["pairwise"]), safe_float(row["mean_margin"])),
            )
            lines.extend(
                [
                    f"- {negative_type}: N `{by_method['visual_only']['n']}`, base `{fmt(by_method['visual_only']['pairwise'])}`, semantic `{fmt(by_method['semantic_baseline']['pairwise'])}`, best selective `{best_selective['method']}={fmt(best_selective['pairwise'])}`, semantic-base `{fmt(by_method['semantic_baseline']['semantic_vs_base_delta'])}`, selective-semantic `{fmt(best_selective['selective_vs_semantic_delta'])}`, full-minus-relphrase-spatialbind `{fmt(by_method['full_semantic_grounding']['full_minus_relphrase_spatialbind_delta'])}`, interpretation `{best_selective['interpretation_label']}`",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Cross-Model Coverage",
            "",
            f"- models evaluated on relaxed subset: `{[cfg['model'] for cfg in models]}`",
            "",
            "## Final Judgment",
            "",
            f"- inclusion status: `{status}`",
            "- automatic heuristic construction",
            "- no human validation",
            "- expected text plausibility labels are not gold annotations",
            "- noisy negatives are still possible",
            "- controlled diagnostic set, not universal benchmark",
            "- no causal disentanglement claim",
            "",
        ]
    )
    V3_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    V3_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/7] rebuilding v2 candidate pools", flush=True)
    image_rows, _ = v2.build_image_candidate_rows_v2(AUDIT_RECONSTRUCTION_IMAGE_COUNT, args.seed)
    assign_candidate_ids(image_rows)
    candidate_index = build_candidate_row_index(image_rows)
    audit_rows = load_audit_rows()
    verify_reconstruction(candidate_index, audit_rows)
    merge_audit_metrics(candidate_index, audit_rows)

    print("[2/7] selecting expanded v3 shortlist", flush=True)
    shortlist_info = choose_expanded_shortlist(
        image_rows,
        plausible_k=args.expanded_plausible_k,
        implausible_k=args.expanded_implausible_k,
    )
    expanded_shortlist = v2.build_pairwise_benchmark_from_candidates(image_rows, "shortlist")
    save_json(V3_EXPANDED_SHORTLIST_JSON, expanded_shortlist)

    print("[3/7] scoring expanded shortlist with CLIP L/14", flush=True)
    clip_l14_cfg = next(cfg for cfg in model_configs_for_v3() if cfg["model"] == "clip_vitl14")
    expanded_factorized_path = score_with_model(V3_EXPANDED_SHORTLIST_JSON, clip_l14_cfg, args.gpu, args.device)
    expanded_payload = load_json(expanded_factorized_path)
    enrich_shortlist_metrics(image_rows, expanded_payload)
    thresholds = compute_thresholds(image_rows)

    print("[4/7] choosing v3 final candidates and matched subsets", flush=True)
    selected_by_type, audit_selection_rows = choose_final_candidates(image_rows, thresholds)
    subsets = build_subset_maps(selected_by_type)
    subset_report = build_subset_report(subsets)
    write_csv(V3_AUDIT_CSV, audit_selection_rows)

    shortlist_benchmark_map = {
        sample["sample_id"]: sample
        for sample in expanded_shortlist["samples"]
    }
    strict_sample_ids = sample_ids_from_subset(subsets[STRICT_NAME])
    relaxed_sample_ids = sample_ids_from_subset(subsets[RELAXED_NAME])
    relaxed_benchmark = build_benchmark_from_sample_ids(shortlist_benchmark_map, relaxed_sample_ids)
    save_json(V3_RELAXED_BENCHMARK_JSON, relaxed_benchmark)
    save_json(V3_STRICT_SAMPLE_IDS_JSON, sorted(strict_sample_ids))
    save_json(V3_RELAXED_SAMPLE_IDS_JSON, sorted(relaxed_sample_ids))

    print("[5/7] evaluating CLIP L/14 from cached shortlist payload", flush=True)
    results_rows: List[Dict[str, Any]] = []
    relaxed_payload = v2.subset_payload(expanded_payload, relaxed_sample_ids)
    strict_payload = v2.subset_payload(expanded_payload, strict_sample_ids)
    results_rows.extend(evaluate_subset_payload(clip_l14_cfg, strict_payload, args.bootstrap_iters, args.seed, STRICT_NAME))
    results_rows.extend(evaluate_subset_payload(clip_l14_cfg, relaxed_payload, args.bootstrap_iters, args.seed, RELAXED_NAME))

    print("[6/7] scoring relaxed v3 benchmark for additional models", flush=True)
    successful_models = [clip_l14_cfg]
    for model_cfg in model_configs_for_v3():
        if model_cfg["model"] == "clip_vitl14":
            continue
        try:
            factorized_path = score_with_model(V3_RELAXED_BENCHMARK_JSON, model_cfg, args.gpu, args.device)
            payload = load_json(factorized_path)
            results_rows.extend(evaluate_subset_payload(model_cfg, v2.subset_payload(payload, strict_sample_ids), args.bootstrap_iters, args.seed, STRICT_NAME))
            results_rows.extend(evaluate_subset_payload(model_cfg, payload, args.bootstrap_iters, args.seed, RELAXED_NAME))
            successful_models.append(model_cfg)
        except Exception as exc:
            print(f"SKIP MODEL {model_cfg['model']}: {exc}", flush=True)

    print("[7/7] writing outputs", flush=True)
    write_csv(V3_RESULTS_CSV, results_rows)
    write_latex_table(results_rows)
    write_selection_audit(thresholds, shortlist_info, subset_report, results_rows)
    write_summary(thresholds, shortlist_info, subset_report, results_rows, successful_models)

    print(f"strict sample count: {len(strict_sample_ids)}")
    print(f"relaxed sample count: {len(relaxed_sample_ids)}")
    print(f"models evaluated: {[cfg['model'] for cfg in successful_models]}")
    print(f"inclusion status: {benchmark_status(results_rows)}")


if __name__ == "__main__":
    main()

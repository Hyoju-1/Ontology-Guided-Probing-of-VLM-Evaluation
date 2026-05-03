# From Text Plausibility to Image-Conditioned Correctness: Ontology-Guided Probing of VLM Evaluation

Anonymous double-blind NeurIPS E&D submission package.

## Purpose

This release packages the research code and selected derived summaries used to reproduce or inspect the main experiments for ontology-guided selective correction, semantic baselines, edit-aware contrastive deltas, coverage-controlled diagnostics, held-out operating-point validation, and cross-model evaluation.

## Included Components

- `Knowledge/`: core experiment scripts, configs, and selected derived outputs
- `scripts/`: paper-facing figure scripts
- `artifacts/`: small figure/table input artifacts
- `figures/figure2_case_images/`: qualitative case assets used by Figure 2
- `anonymization_report.md`: release construction and anonymization notes
- `anonymization_scan_results.txt`: recursive post-build leak scan report

## Core Experiment Groups

- Benchmark construction and ontology resources:
  - `Knowledge/build_semantic_violation_benchmark.py`
  - `Knowledge/build_winoground_materialized_benchmark.py`
  - `Knowledge/build_large_pool_extension_benchmarks.py`
  - `Knowledge/build_vg_coverage_ontology_json.py`
  - `Knowledge/build_ontology_prototypes.py`
  - `Knowledge/extract_semantic_components.py`
  - `Knowledge/export_generalized_semantic_compat.py`

- Scoring and edit-aware correction:
  - `Knowledge/score_semantic_violation_benchmark.py`
  - `Knowledge/score_image_conditioned_semantic_plausibility.py`
  - `Knowledge/edit_aware_contrastive_grounding_analysis.py`
  - `Knowledge/prepare_component_fusion_scores.py`

- Cross-model evaluation:
  - `Knowledge/run_cross_model_suite.py`
  - `Knowledge/run_openclip_retrieval_eval.py`
  - `Knowledge/run_bridgetower_retrieval_eval.py`
  - `Knowledge/run_qwen2vl_forced_choice_eval.py`
  - `Knowledge/run_qwen2vl_likelihood_grounding_eval.py`
  - `Knowledge/summarize_cross_model_suite.py`
  - `Knowledge/summarize_blip_cross_model.py`
  - `Knowledge/summarize_siglip_qwen_cross_model.py`

- Statistical tests and diagnostics:
  - `Knowledge/run_editaware_significance.py`
  - `Knowledge/bootstrap_editaware_significance.py`
  - `Knowledge/analyze_significance_selective_correction.py`
  - `Knowledge/analyze_coco_winoground_significance_extension.py`
  - `Knowledge/analyze_blip_openclip_significance.py`
  - `Knowledge/build_significance_master_table.py`
  - `Knowledge/run_gate_rescue_harm_audit.py`
  - `Knowledge/run_heldout_operating_point_validation.py`
  - `Knowledge/run_coverage_controlled_evaluation.py`
  - `Knowledge/run_controlled_signal_analysis.py`
  - `Knowledge/run_backbone_calibration_test.py`
  - `Knowledge/run_confounding_controlled_benchmark.py`
  - `Knowledge/run_confounding_controlled_benchmark_v3.py`

- SugarCrepe / controlled pilot diagnostics:
  - `Knowledge/run_sugarcrepe_pilot.py`
  - `Knowledge/summarize_sugarcrepe_pilot.py`
  - `Knowledge/analyze_sugarcrepe_significance.py`
  - `Knowledge/analyze_sugarcrepe_subtypes.py`
  - `Knowledge/run_sugarcrepe_gate_relaxation.py`

- Figure generation:
  - `scripts/make_figure1_ed_overview.py`
  - `scripts/make_figure2_failure_cases.py`
  - `scripts/make_figure3_cross_model_evaluator_dependence.py`
  - `scripts/make_figure4_score_geometry.py`

## Expected Inputs

Raw datasets and large model checkpoints are not included. Most scripts expect benchmark JSONs, scored benchmark files, or upstream model outputs under relative `Knowledge/output/` paths. Where the original workspace used absolute local paths, this release rewrites them to relative placeholders or generic local-path markers.

## Reproducing Main Tables

Typical entry points:

- Main significance summaries:
  - `Knowledge/build_significance_master_table.py`
  - `Knowledge/analyze_coco_winoground_significance_extension.py`
  - `Knowledge/analyze_blip_openclip_significance.py`

- Cross-model tables:
  - `Knowledge/run_cross_model_suite.py`
  - `Knowledge/summarize_cross_model_suite.py`
  - `Knowledge/summarize_blip_cross_model.py`
  - `Knowledge/summarize_siglip_qwen_cross_model.py`

- Backbone-aware and calibration analyses:
  - `Knowledge/analyze_backbone_aware_two_mode.py`
  - `Knowledge/run_backbone_calibration_test.py`

Selected paper-facing CSV/JSON/MD/TEX outputs are already included under `Knowledge/output/` for inspection.

## Appendix Diagnostics

Representative appendix diagnostics can be reproduced or inspected with:

- `Knowledge/run_gate_rescue_harm_audit.py`
- `Knowledge/run_heldout_operating_point_validation.py`
- `Knowledge/run_coverage_controlled_evaluation.py`
- `Knowledge/run_controlled_signal_analysis.py`
- `Knowledge/run_confounding_controlled_benchmark_v3.py`
- `Knowledge/run_qwen2vl_likelihood_grounding_eval.py`
- `Knowledge/analyze_gate_sensitivity.py`

## Notes

- Raw datasets, proprietary checkpoints, and local environment caches are intentionally omitted.
- Some scripts assume external benchmark files or model outputs that must be regenerated from original public sources.
- Relative paths may need editing if users place benchmark inputs in different locations.
- This package prioritizes double-blind safety over including every large intermediate artifact.

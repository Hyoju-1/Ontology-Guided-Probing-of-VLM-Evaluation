# Ontology-Guided Probing of VLM Evaluation

This repository contains the anonymous supplementary code and selected result artifacts for the paper:

**From Text Plausibility to Image-Conditioned Correctness: Ontology-Guided Probing of VLM Evaluation**

This release is prepared for double-blind review. Author names, affiliations, personal paths, and private repository information have been removed.

## Main Scripts

scripts/run_main_cross_model_eval.py
scripts/run_significance_tests.py
scripts/run_backbone_calibration_test.py
scripts/run_confounding_controlled_benchmark.py
scripts/run_controlled_signal_analysis.py
scripts/run_gate_rescue_harm_audit.py
scripts/run_heldout_operating_point_validation.py
scripts/run_coverage_controlled_evaluation.py
scripts/run_eligibility_shuffle_control.py
scripts/run_qwen2vl_likelihood_analysis.py

These scripts correspond to the main paper results and appendix diagnostics.

## Contents

The release includes only files needed to inspect or reproduce the paper-facing results:

core ontology/parsing/scoring/evaluation code
scripts for main and appendix experiments
selected summary tables and LaTeX tables
compact CSV/Markdown result summaries
anonymization report and scan results

The release does not include raw datasets, model checkpoints, embedding caches, temporary outputs, failed reruns, or unrelated experiment files.


## Selected Outputs
output/tables/
output/summaries/
output/latex_tables/

These directories contain compact final artifacts used to support the reported paper tables. Full raw output dumps are intentionally excluded.

## Repository Structure

```text
anonymous_release_ontology_guided_vlm_eval/
├── README.md
├── requirements.txt or environment.yml
├── config/
├── src/
│   ├── ontology/
│   ├── scoring/
│   ├── evaluation/
│   └── utils/
├── scripts/
│   ├── run_main_cross_model_eval.py
│   ├── run_significance_tests.py
│   ├── run_backbone_calibration_test.py
│   ├── run_confounding_controlled_benchmark.py
│   ├── run_controlled_signal_analysis.py
│   ├── run_gate_rescue_harm_audit.py
│   ├── run_heldout_operating_point_validation.py
│   ├── run_coverage_controlled_evaluation.py
│   ├── run_eligibility_shuffle_control.py
│   └── run_qwen2vl_likelihood_analysis.py
├── output/
│   ├── tables/
│   ├── summaries/
│   └── latex_tables/
├── figures/
└── anonymization_report.md

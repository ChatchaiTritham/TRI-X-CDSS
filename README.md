# TRI-X-CDSS: Integration Package for the TRI-X Clinical Triage Framework (TRI-X-CDSS)

> An installable Python package that wires the TRI-X decision components — triage, TiTrATE, governance, dynamic risk staging, and routing — into one import surface, with a worked patient-level example a reader can run to see the pieces operate together.

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Reproducible](https://img.shields.io/badge/reproducible-seed--42-success)

## Overview

The TRI-X manuscript argues for governing a triage decision under diagnostic uncertainty rather than predicting a single label. That argument is carried by the [TRI-X](../TRI-X) repository, which holds the formal guideline logic, the routing schema, and the synthetic-cohort experiments the paper reports. This repository plays a different and narrower part. It packages the same family of components — emergency screening, time-bound bedside reasoning, dynamic risk staging, and care-pathway routing — behind a single installable module, so the framework can be imported, exercised, and extended as ordinary Python rather than read as study code.

There is no standalone manuscript tied to this package, and there are no headline metrics to defend. What it offers is an engineering view of the framework: a `trix_cdss` namespace whose public functions cover dizziness triage, ESI-to-DRAS mapping, red-flag screening, time-bounded assessment, and pathway routing, plus a Tier-1 patient-level layer with simple digital-twin disease models. We keep the scope deliberately modest. The package validates that the components import cleanly and run together on a worked scenario; it does not reproduce the manuscript's safety-property experiments, which remain the job of the TRI-X repository.

A reader who wants to understand what the framework computes can read the source directly, run the bundled BPPV example, and inspect the figures it writes. A reader who wants the paper's reproducibility evidence should start from TRI-X instead. We state that division here so neither repository is mistaken for the other.

## Key results

This package produces an implementation demonstration, not effectiveness numbers. The points below describe what re-running the code actually yields.

- The smoke test confirms that the public package surface stays importable: `FRAMEWORK_NAME`, `__version__`, and `perform_dizziness_triage` resolve from a fresh `import trix_cdss`.
- The Tier-1 BPPV example runs end to end from a seeded disease model (`random_seed=42`), simulating natural history against immediate and delayed Epley maneuvers and printing per-scenario symptom trajectories and DRAS-5 urgency levels.
- The example writes three implementation figures to `examples/output/` and the figure manifest lists exactly those three entries.
- No accuracy, sensitivity, or calibration claim is made anywhere in this repository; any such evidence lives with the source studies, not here.

## Repository structure

```text
src/trix_cdss/            package source: core/ (triage, titrate, srgl, dras5, orasr),
                          tier1_patient/ (digital_twin disease models, xai), visualization/
examples/tier1_examples/  bppv_simulation_demo.py — worked end-to-end scenario
examples/output/          figures written by the demo (fig1–fig3) + visual QA sheet
scripts/                  generate_manuscript_manifest.py — rebuilds FIGURE_MANIFEST.csv
notebooks/                quickstart and case-comparison walkthroughs
tests/                    package smoke test
FIGURE_MANIFEST.csv       curated inventory of the implementation figures
data/, models/, results/, figures/, outputs/, evaluation/   placeholders for generated assets
pyproject.toml, setup.py, requirements.txt, pytest.ini       packaging and test config
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```

## Reproducing the results

```bash
python examples/tier1_examples/bppv_simulation_demo.py   # seed 42; writes figs to examples/output/
python scripts/generate_manuscript_manifest.py           # rebuild FIGURE_MANIFEST.csv + QA sheet
python -m pytest -q                                       # package smoke test
```

The disease simulation is seeded, so the printed trajectories and the rendered figures 1 and 2 redraw consistently across runs on the same machine. The demo is a single-scenario walkthrough rather than an experiment harness — it carries no `run_all.py`, and the `results/` and `evaluation/` directories ship as placeholders because this package does not generate study-level result files. Treat the console output and the three figures as the reproducible artefacts here.

## Results and figures

The figures are illustrative outputs from one patient-level scenario, kept under `examples/output/` and indexed in `FIGURE_MANIFEST.csv`. They show how the framework behaves on a worked case; they are not summaries of a cohort.

- `examples/output/fig1_bppv_trajectory_epley_immediate.png` — vertigo severity over time for a single BPPV patient who receives the Epley maneuver at presentation. Read it as the symptom curve the seeded disease model produces under immediate treatment.
- `examples/output/fig2_bppv_trajectory_comparison.png` — the same patient under three plans (no intervention, immediate Epley, delayed Epley) on shared axes, so the trajectories can be compared directly.
- `examples/output/fig3_shap_feature_importance.png` — a feature-importance bar chart for DRAS-5 urgency interpretation. The importance values are computed by the rule-based DRAS-5 classifier (`classify_urgency_level` in `src/trix_cdss/core/dras5.py`): the demo builds a `DRAS5Features` vector from the synthetic patient and reads back the per-feature contributions the classifier attributed to the decision. These are additive rule-based attributions, not Shapley (SHAP) values; the bars show how each feature pushed the urgency score up or down for this one synthetic case.

All three figures are computed from the seeded simulation and the classifier; none of the plotted values are hardcoded. Figure 3 reflects the rule-based feature attributions for the single demo patient, not a cohort or a fitted SHAP explainer.

## Data

The package ships no patient data. The Tier-1 example builds a single synthetic patient archetype in code and drives it through guideline-motivated disease-model parameters; the disease model is seeded for repeatability. No real records, no human subjects, and therefore no institutional review board approval are involved. The synthetic parameters encode clinically plausible priors for demonstration and are not calibrated to epidemiological frequencies.

## Citation

```bibtex
@article{tritham_trix,
  title   = {TRI-X: A Safety-First Explainable Framework for Decision-Centric
             Clinical Triage under Diagnostic Uncertainty},
  author  = {Tritham, Chatchai and Snae Namahoot, Chakkrit},
  journal = {Journal of Intelligent Information Systems},
  note    = {to appear},
  year    = {2026}
}
```

This integration package supports the TRI-X study; cite the article above when referring to the framework.

## License

Released under the MIT License (see `LICENSE`).

## Contact

**Chatchai Tritham** — Department of Computer Science and Information Technology, Faculty of Science, Naresuan University, Phitsanulok 65000, Thailand. Email: chatchait66@nu.ac.th · ORCID: 0000-0001-7899-228X
**Chakkrit Snae Namahoot** — same affiliation. Email: chakkrits@nu.ac.th · ORCID: 0000-0003-4660-4590

## Portfolio relationship

| Repository | Role |
|---|---|
| BASICS-CDSS | Beyond-accuracy evaluation methodology |
| TRI-X | Framework-level package |
| ORASR | Routing and safety-action component |
| DRAS-5 | Dynamic risk-state component |
| SAFE-Gate | Safety-gated ensemble framework |
| SynDX | Synthetic validation and explainability evidence |
| SURgul | SRGL/governance reproducibility component |
| TRI-X-CDSS | Integration and implementation package |
| Selective-CDSS | Risk-controlled selective-prediction (abstention) component |
| Causal-CDSS | Causal-inference evaluation component |
| Beyond-Accuracy | Simulation-based safety/calibration evaluation framework |

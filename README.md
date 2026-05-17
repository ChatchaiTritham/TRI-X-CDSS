# TRI-X-CDSS

## Overview

Three-tier evaluation framework for dizziness clinical decision support.

## Installation

```bash
pip install -e .
```

## Repository Structure

- `src/trix_cdss/`: source package
- `tests/`: automated tests
- `examples/`: example usage
- `notebooks/`: interactive walkthroughs
- `docs/`: supporting documentation

## Tutorials And Demos

- Example script:
  - `examples/tier1_examples/bppv_simulation_demo.py`: end-to-end Tier 1 BPPV simulation, figure generation, and treatment comparison
  - `scripts/generate_manuscript_manifest.py`: implementation figure manifest and visual QA sheet
- Notebooks:
  - `notebooks/01_trix_cdss_quickstart.ipynb`: interactive quickstart across Triage, SRGL, TiTrATE, DRAS-5, and ORASR
  - `notebooks/02_trix_cdss_case_comparison.ipynb`: richer multi-case comparison with urgency and routing visual outputs

Run the example from the repository root:

```bash
python examples/tier1_examples/bppv_simulation_demo.py
```

## Implementation Figure Status

TRI-X-CDSS is currently an implementation/integration repository. Its figures
are implementation evidence, not a standalone article figure set. Do not claim
TRI-X-CDSS as a standalone article package unless a separate integration or
benchmark manuscript is written and verified.

Regenerate example figures:

```bash
python examples/tier1_examples/bppv_simulation_demo.py
```

Regenerate the manifest and visual QA sheet:

```bash
python scripts/generate_manuscript_manifest.py
```

Outputs:

- `examples/output/`: PNG implementation figures
- `FIGURE_MANIFEST.csv`: figure role, source script, source artifact, caption,
  and implementation-evidence section
- `examples/output/visual_qa_contact_sheet.png`: visual QA sheet

## Cross-Repository Tutorial Charts

- `../tutorial_surface_comparison.png`: scripts vs examples vs notebooks across all repositories
- `../tutorial_asset_density.png`: interactive/tutorial asset density normalized by repository size

## Source Layout

This repository uses the recommended `src/<package_name>` layout.
Importable code lives in `src/trix_cdss/`.

## Testing

```bash
pytest tests -v
```

## Manuscript Alignment

TRI-X-CDSS is currently an implementation/integration repository. No standalone
TRI-X-CDSS manuscript is active at this time. If a future integration or
benchmark manuscript is written, it should be treated as in preparation until it
is independently verified and documented.

The repository currently supports:

- Tier 1 BPPV simulation evidence
- Epley-immediate and comparative trajectory figures
- SHAP feature-importance implementation evidence
- integration across Triage, SRGL, TiTrATE, DRAS-5, and ORASR concepts

Use this repository to support reproducibility and integration claims, not to
increase the standalone article count.

## Methodological References

The implementation package connects:

- TRI-X as the framework owner
- SURgul/SRGL as governance logic
- DRAS-5 as stateful risk-action behavior
- ORASR as pathway routing
- SynDX as synthetic validation/XAI evidence

## Citation

TRI-X-CDSS is an implementation/integration repository. Cite this software
repository using `CITATION.cff`; do not cite it as a standalone article unless a
future manuscript is written, verified, and documented.

## Contact

### Contact Author

**Chatchai Tritham** (Author)

- Email: [chatchait66@nu.ac.th](mailto:chatchait66@nu.ac.th)
- ORCID: [0000-0001-7899-228X](https://orcid.org/0000-0001-7899-228X)
- Department of Computer Science and Information Technology
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand

### Supervisor

**Chakkrit Snae Namahoot**

- E-mail: [chakkrits@nu.ac.th](mailto:chakkrits@nu.ac.th)
- ORCID: [0000-0003-4660-4590](https://orcid.org/0000-0003-4660-4590)
- Department of Computer Science and Information Technology
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand

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
- Notebooks:
  - `notebooks/01_trix_cdss_quickstart.ipynb`: interactive quickstart across Triage, SRGL, TiTrATE, DRAS-5, and ORASR
  - `notebooks/02_trix_cdss_case_comparison.ipynb`: richer multi-case comparison with urgency and routing visual outputs

Run the example from the repository root:

```bash
python examples/tier1_examples/bppv_simulation_demo.py
```

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

## Contact

### Contact Author

**Chatchai Tritham** (PhD Candidate)

- Email: [chatchait66@nu.ac.th](mailto:chatchait66@nu.ac.th)
- Department of Computer Science and Information Technology
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand

### Supervisor

**Chakkrit Snae Namahoot**

- Email: [chakkrits@nu.ac.th](mailto:chakkrits@nu.ac.th)
- Department of Computer Science
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand

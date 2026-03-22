# TRI-X-CDSS: Three-Tier Evaluation Framework for Dizziness Clinical Decision Support

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**TRI-X** (Triage-TiTrATE-eXplainable AI) is a comprehensive framework for evaluating clinical decision support systems in the high-uncertainty domain of acute dizziness and vertigo diagnosis.

## 🎯 Overview

TRI-X addresses the critical gap between CDSS algorithm development and clinical deployment through a novel **3-tier evaluation framework**:

### **Three Core Components**
- **Triage**: ESI-based urgency assessment for dizziness presentations
- **TiTrATE**: Time-bound Risk-stratified Action with Treatment Escalation
- **XAI**: Explainable AI throughout (SHAP, LIME, NMF, Counterfactual Reasoning)

### **Three Evaluation Tiers**
1. **Tier 1 (Patient-Level)**: Digital Twin temporal simulation + Causal SCM for treatment effects
2. **Tier 2 (System-Level)**: Multi-agent workflow simulation for deployment readiness
3. **Tier 3 (Integration)**: Cross-tier validation + clinical governance framework

### **DRAS-5 Urgency Classification**
1. **Level 5**: Immediate emergency (stroke within thrombolysis window)
2. **Level 4**: Urgent specialist (same-day evaluation)
3. **Level 3**: Scheduled specialist OPD (reduces ED crowding)
4. **Level 2**: Lower urgency (routine GP appointment)
5. **Level 1**: Safe/no danger (self-care, GP follow-up as needed)

## 🌟 Why Dizziness/Vertigo?

Dizziness represents a **high-uncertainty diagnostic domain**:
- Patients struggle to describe symptoms accurately ("dizzy" vs "vertigo")
- Multiple specialties involved (Emergency Medicine, Neurology, ENT, Primary Care)
- Broad differential diagnosis (BPPV to posterior circulation stroke)
- Critical time-sensitive decisions (2-hour thrombolysis window for stroke)
- Low stroke prevalence (3-5%) but catastrophic if missed

## 🚀 Novel Contributions

### 1. **XAI-Based Synthetic Data (SynDX)**
- **Novel methodology**: Use XAI tools to generate synthetic patient scenarios
- **Privacy-preserving**: No real patient data required
- **Bias-free**: No medical expert bias in generation
- **Transparent**: Fully documented generation process

### 2. **Digital Twin for CDSS Evaluation**
- Novel application of digital twins (typically used for treatment planning)
- Simulate disease progression for 5 vestibular disorders:
  - BPPV (Benign Paroxysmal Positional Vertigo)
  - Vestibular Neuritis
  - Posterior Circulation Stroke
  - Meniere's Disease
  - Vestibular Migraine

### 3. **Causal Inference for Treatment Effects**
- Do-calculus interventions: `do(Epley)`, `do(Thrombolysis)`
- Counterfactual reasoning: "What if patient presented 30 minutes earlier?"
- ATE, CATE, ITE estimation for personalized decisions

### 4. **Multi-Agent Workflow Simulation**
- Pre-deployment prediction of alert fatigue and override rates
- Uncertainty propagation: Patient reporting → CDSS → Clinician decision
- Resource constraint modeling (imaging availability, specialist after-hours)

### 5. **Pre-Clinical Governance Framework**
- Deployment readiness scorecard (Safety, Efficiency, Transparency, Governance)
- Clear pathway: Synthetic → Retrospective → Prospective → RCT
- Regulatory alignment (FDA AI/ML SaMD, WHO AI Ethics, NICE Evidence Standards)

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/ChatchaiTritham/TRI-X-CDSS.git
cd TRI-X-CDSS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## 🎬 Quick Start

### Example 1: Patient-Level DRAS-5 Classification

```python
from trix_cdss.core import perform_dizziness_triage, classify_urgency_level
from trix_cdss.core.triage import VitalSigns, DizzinessSymptoms, PatientRiskFactors
from trix_cdss.core.dras5 import DRAS5Features

# Patient presentation
vitals = VitalSigns(
    heart_rate=85, systolic_bp=140, diastolic_bp=85,
    respiratory_rate=16, temperature=37.0, oxygen_saturation=98
)

symptoms = DizzinessSymptoms(
    symptom_description="spinning sensation",
    duration_hours=2.0,
    diplopia=True,  # Double vision - stroke warning sign
    ataxia=True,    # Cannot walk - stroke warning sign
    positional_trigger=False
)

risk_factors = PatientRiskFactors(
    age=72,
    atrial_fibrillation=True,  # High stroke risk
    hypertension=True
)

# Perform triage
triage = perform_dizziness_triage(vitals, symptoms, risk_factors)
print(f"ESI Level: {triage.esi_level}")  # ESI Level 2 (High Risk)

# DRAS-5 classification
features = DRAS5Features(
    age=72,
    atrial_fibrillation=True,
    symptom_duration_hours=2.0,  # Within thrombolysis window
    hints_central=True,  # HINTS suggests central cause
    stroke_probability=0.85
)

classification = classify_urgency_level(features)
print(f"DRAS Level: {classification.level}")  # DRAS-5 (Immediate Emergency)
print(f"Confidence: {classification.confidence:.2f}")
print(f"Pathway: {classification.recommended_pathway}")
# Output: ED → Stroke Protocol → CT/MRI → Neurology Stat → Thrombolysis Evaluation
```

### Example 2: Digital Twin Disease Simulation

```python
from trix_cdss.tier1_patient.digital_twin import BPPVModel, PatientArchetype
from trix_cdss.tier1_patient.xai import discover_temporal_patterns

# Create BPPV patient
patient = PatientArchetype(
    age=65,
    gender="female",
    disease_type="bppv",
    canal_affected="posterior"
)

# Simulate disease progression
model = BPPVModel()
trajectory = model.simulate_progression(
    patient=patient,
    time_points=[0, 1, 4, 24],  # T0, T1, T2, T3 (hours)
    intervention="epley_maneuver",
    intervention_time=4.0  # Epley at T2 (4 hours)
)

# Visualize trajectory
from trix_cdss.visualization import plot_disease_trajectory
plot_disease_trajectory(trajectory, save_path="bppv_trajectory.png")

# Discover patterns with NMF
patterns = discover_temporal_patterns(
    patient_trajectories=[trajectory1, trajectory2, ...],
    n_patterns=5
)
```

### Example 3: Multi-Agent Workflow Simulation

```python
from trix_cdss.tier2_system import EmergencyDepartment, PatientAgent, CDSSAgent
from trix_cdss.tier2_system.workflows import StrokeImagingWorkflow

# Create environment
ed = EmergencyDepartment(
    num_beds=30,
    ct_available=True,
    neurology_available=True
)

# Create agents
patient = PatientAgent(archetype=stroke_patient_archetype)
cdss = CDSSAgent(trix_model=trix_model)

# Run workflow
workflow = StrokeImagingWorkflow(ed, patient, cdss)
result = workflow.execute()

print(f"Time to imaging: {result.time_to_imaging_minutes} minutes")
print(f"Alert override: {result.clinician_override}")
print(f"Thrombolysis eligible: {result.thrombolysis_eligible}")
```

## 📊 Visualization Examples

TRI-X-CDSS includes comprehensive visualization tools for all tiers:

### Patient-Level Visualizations
- Disease trajectory plots (temporal symptom evolution)
- SHAP feature importance bar charts
- NMF pattern heatmaps
- Causal DAGs with do-calculus interventions
- ROC curves for DRAS-5 classification
- Confusion matrices

### System-Level Visualizations
- Multi-agent interaction networks
- Alert fatigue over time (line plots)
- Workflow disruption distributions (histograms)
- Uncertainty propagation cascade diagrams
- Resource utilization heatmaps

### Integration Visualizations
- Cross-tier comparison plots
- Deployment readiness radar charts
- Sankey diagrams for patient flow

## 📂 Repository Structure

```
TRI-X-CDSS/
├── src/trix_cdss/
│   ├── core/                   # Core TRI-X components (Week 1-2) ✅
│   │   ├── triage.py          # ESI triage
│   │   ├── titrate.py         # Time-bound risk assessment
│   │   ├── srgl.py            # Safety governance
│   │   ├── dras5.py           # 5-level urgency classification
│   │   └── orasr.py           # Operational routing
│   │
│   ├── tier1_patient/         # Patient-level evaluation (Week 3-5)
│   │   ├── digital_twin/      # Disease models
│   │   │   ├── disease_models/
│   │   │   │   ├── bppv.py
│   │   │   │   ├── vestibular_neuritis.py
│   │   │   │   ├── posterior_stroke.py
│   │   │   │   ├── menieres.py
│   │   │   │   └── vestibular_migraine.py
│   │   │   ├── temporal_simulation.py
│   │   │   └── patient_archetype.py
│   │   ├── causal/            # Causal inference
│   │   │   ├── causal_graphs/
│   │   │   ├── interventions.py
│   │   │   └── treatment_effects.py
│   │   └── xai/               # Explainable AI
│   │       ├── shap_lime.py
│   │       ├── nmf_patterns.py
│   │       └── counterfactual.py
│   │
│   ├── tier2_system/          # System-level evaluation (Week 6-8)
│   │   ├── agents/            # Multi-agent simulation
│   │   ├── environment/       # ED, Neurology Clinic, GP Clinic
│   │   ├── workflows/         # HINTS, Epley, Stroke Imaging, Referral
│   │   └── systemic_metrics/  # Alert fatigue, override rates
│   │
│   ├── tier3_integration/     # Integration + governance
│   │   ├── cross_tier_validation.py
│   │   └── deployment_readiness.py
│   │
│   ├── data/
│   │   └── syndx/             # Synthetic dizziness dataset
│   │       ├── generation/    # XAI-based generation
│   │       ├── archetypes/    # Patient archetypes
│   │       └── validation/    # Fidelity, utility, discriminability
│   │
│   ├── visualization/         # Plotting and visualization
│   │   ├── patient_level_viz.py
│   │   ├── system_level_viz.py
│   │   └── integration_viz.py
│   │
│   └── evaluation/            # Performance metrics
│       ├── metrics.py
│       └── benchmarks.py
│
├── tests/                     # Unit and integration tests
│   ├── core/
│   ├── tier1/
│   ├── tier2/
│   ├── tier3/
│   └── integration/
│
├── examples/                  # Example scripts
│   ├── tier1_examples/
│   ├── tier2_examples/
│   └── full_pipeline/
│
├── docs/                      # Documentation
│   ├── figures/
│   ├── api/
│   └── tutorials/
│
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest --cov=trix_cdss tests/

# Run specific tier tests
pytest tests/tier1/
pytest tests/tier2/
pytest tests/tier3/

# Run integration tests
pytest tests/integration/
```

## 📈 Performance Benchmarks

TRI-X-CDSS includes comprehensive performance testing:

```bash
# Run performance benchmarks
python -m trix_cdss.evaluation.benchmarks

# Expected performance (on i7-10th gen, 16GB RAM):
# - Patient-level classification: <100ms per patient
# - Disease trajectory simulation: <500ms per patient
# - Multi-agent simulation (30 days, 1440 patients): <30 minutes
# - Full 3-tier evaluation (5000 patients): <2 hours
```

## 📚 Documentation

- **Framework Overview**: [docs/framework_overview.md](docs/framework_overview.md)
- **Disease Models**: [docs/disease_models.md](docs/disease_models.md)
- **Clinical Workflows**: [docs/workflows.md](docs/workflows.md)
- **API Reference**: [docs/api/](docs/api/)
- **Tutorials**: [docs/tutorials/](docs/tutorials/)

## 🔬 Research Papers

This repository supports three research papers:

1. **Paper 1 (Tier 1)**: "Patient-Level Evaluation of Clinical Decision Support for Dizziness: Integrating Digital Twin Simulation and Causal Inference"
   - Target: Nature Machine Intelligence
   - Status: In preparation

2. **Paper 2 (Tier 2)**: "System-Level Evaluation of Clinical Decision Support for Dizziness: Multi-Agent Simulation of Workflow Integration and Uncertainty Propagation"
   - Target: Journal of the American Medical Informatics Association (JAMIA)
   - Status: In preparation

3. **Paper 3 (Tier 3)**: "From Synthetic Validation to Clinical Deployment: A Pre-Clinical Governance Framework for Clinical Decision Support Systems"
   - Target: Nature Medicine or The Lancet Digital Health
   - Status: In preparation

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by FDA AI/ML SaMD Framework and WHO AI Ethics Guidelines
- Built on principles from DECIDE-AI and CONSORT-AI frameworks
- Clinical guidelines from ESI, HINTS exam protocols, and stroke pathway standards

## 📖 Citation

If you use TRI-X-CDSS in your research, please cite:

```bibtex
@software{trix_cdss_2026,
  title = {TRI-X-CDSS: Three-Tier Evaluation Framework for Dizziness Clinical Decision Support},
  author = {Chatchai Tritham},
  year = {2026},
  url = {https://github.com/ChatchaiTritham/TRI-X-CDSS},
  version = {1.0.0}
}
```

---

**Status**: 🟡 Active Development (Week 1 of 30)
**Last Updated**: 2026-01-28

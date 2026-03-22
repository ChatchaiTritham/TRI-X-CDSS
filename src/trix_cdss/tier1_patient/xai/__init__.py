"""Explainable AI (XAI) Components for TRI-X-CDSS.

This module provides three complementary XAI approaches for transparent
clinical decision support:

1. **SHAP/LIME**: Feature importance for DRAS-5 classification
   - Why was this patient classified as DRAS-5?
   - Which features increase/decrease urgency?
   - Validation that model uses clinically appropriate features

2. **NMF Patterns**: Temporal pattern discovery in disease trajectories
   - Identify common symptom progression patterns
   - Discover disease subtypes (rapid vs slow resolution)
   - Predict outcomes based on early trajectory matching

3. **Counterfactual Reasoning**: Causal "what-if" analysis
   - What if Epley performed immediately vs delayed?
   - What if patient arrived within golden hour?
   - Optimal treatment timing recommendations

These XAI methods address different aspects of explainability:
- SHAP/LIME: Decision-level explanation (why this classification?)
- NMF: Pattern-level explanation (what trajectory type?)
- Counterfactual: Causal explanation (what would happen if...?)

Usage Example:
    >>> from trix_cdss.tier1_patient.xai import DRAS5Explainer, CounterfactualAnalyzer
    >>>
    >>> # SHAP explanation
    >>> explainer = DRAS5Explainer(dras_classifier)
    >>> explanation = explainer.explain_patient_shap(patient_features, predicted_dras=5)
    >>> print(explainer.generate_textual_explanation(explanation))
    >>>
    >>> # Counterfactual analysis
    >>> analyzer = CounterfactualAnalyzer(disease_model)
    >>> effect = analyzer.estimate_treatment_effect(patient, treatment, control)
    >>> print(effect.clinical_interpretation)
"""

from trix_cdss.tier1_patient.xai.counterfactual import (
    CounterfactualAnalyzer,
    CounterfactualOutcome,
    CounterfactualScenario,
    InterventionType,
    TreatmentEffect,
)
from trix_cdss.tier1_patient.xai.nmf_patterns import (
    NMFTemporalAnalyzer,
    PatternAssignment,
    TemporalPattern,
)
from trix_cdss.tier1_patient.xai.shap_lime import (
    DRAS5Explainer,
    FeatureImportance,
    SHAPExplanation,
    create_shap_summary_plot,
)

__all__ = [
    # SHAP/LIME
    "DRAS5Explainer",
    "SHAPExplanation",
    "FeatureImportance",
    "create_shap_summary_plot",
    # NMF Patterns
    "NMFTemporalAnalyzer",
    "TemporalPattern",
    "PatternAssignment",
    # Counterfactual
    "CounterfactualAnalyzer",
    "CounterfactualScenario",
    "CounterfactualOutcome",
    "TreatmentEffect",
    "InterventionType",
]

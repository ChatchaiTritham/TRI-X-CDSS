"""TRI-X-CDSS: Three-Tier Evaluation Framework for Dizziness Clinical Decision Support.

TRI-X (Triage-TiTrATE-eXplainable AI) Framework Components:
- Triage: ESI-based urgency assessment
- TiTrATE: Time-bound Risk-stratified Action with Treatment Escalation
- XAI: Explainable AI (SHAP, LIME, NMF, Counterfactual)

Three Evaluation Tiers:
- Tier 1 (Patient-Level): Digital Twin + Causal SCM
- Tier 2 (System-Level): Multi-Agent Workflow Simulation
- Tier 3 (Integration): Cross-Tier Validation + Deployment Readiness

DRAS-5 Urgency Classification:
- Level 5: Immediate emergency (stroke <2hr thrombolysis window)
- Level 4: Urgent specialist (same-day)
- Level 3: Scheduled specialist OPD (reduces ED crowding)
- Level 2: Lower urgency (routine GP)
- Level 1: Safe/no danger (self-care)
"""

from trix_cdss.constants import FRAMEWORK_NAME, PACKAGE_VERSION

__version__ = PACKAGE_VERSION
__author__ = "Clinical AI Research Team"
__email__ = "research@example.org"

# Core imports
from trix_cdss.core import (  # Triage; TiTrATE; SRGL; DRAS-5; ORASR
    DRAS5Classification,
    ESITriageAssessment,
    ORASRRouting,
    SRGLScreening,
    TiTrATEAssessment,
    calculate_risk_score,
    calibrate_thresholds,
    check_safety_protocols,
    classify_urgency_level,
    map_esi_to_dras,
    perform_dizziness_triage,
    perform_red_flag_screening,
    perform_safety_checks,
    perform_time_bounded_assessment,
    route_to_care_pathway,
)

__all__ = [
    # Version
    "__version__",
    # Triage
    "ESITriageAssessment",
    "perform_dizziness_triage",
    "map_esi_to_dras",
    # TiTrATE
    "TiTrATEAssessment",
    "perform_time_bounded_assessment",
    "calculate_risk_score",
    # SRGL
    "SRGLScreening",
    "perform_red_flag_screening",
    "check_safety_protocols",
    # DRAS-5
    "DRAS5Classification",
    "classify_urgency_level",
    "calibrate_thresholds",
    # ORASR
    "ORASRRouting",
    "route_to_care_pathway",
    "perform_safety_checks",
    "FRAMEWORK_NAME",
]

"""Visualization modules for TRI-X-CDSS.

This module provides comprehensive visualization tools for all three tiers:
- Tier 1 (Patient-Level): Disease trajectories, SHAP, NMF, Causal DAGs
- Tier 2 (System-Level): Multi-agent networks, workflow disruption, alert fatigue
- Tier 3 (Integration): Cross-tier comparison, deployment readiness
"""

from trix_cdss.visualization.patient_level_viz import (
    plot_causal_dag,
    plot_confusion_matrix,
    plot_disease_trajectory,
    plot_multiple_trajectories,
    plot_nmf_patterns,
    plot_roc_curves,
    plot_shap_importance,
)

__all__ = [
    # Tier 1 visualizations
    "plot_disease_trajectory",
    "plot_multiple_trajectories",
    "plot_shap_importance",
    "plot_nmf_patterns",
    "plot_causal_dag",
    "plot_roc_curves",
    "plot_confusion_matrix",
]

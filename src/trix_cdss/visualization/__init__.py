"""Visualization modules for TRI-X-CDSS.

This module provides comprehensive visualization tools for all three tiers:
- Tier 1 (Patient-Level): Disease trajectories, rule-based feature attribution,
  NMF, Causal DAGs
- Tier 2 (System-Level): Multi-agent networks, workflow disruption, alert fatigue
- Tier 3 (Integration): Cross-tier comparison, deployment readiness

The shared publication style (apply_pub_style / PALETTE / save_fig) is vendored
byte-identical from GitHub/_management/pubviz.py.
"""

from trix_cdss.visualization.patient_level_viz import (
    PALETTE,
    apply_pub_style,
    plot_causal_dag,
    plot_confusion_matrix,
    plot_disease_trajectory,
    plot_feature_attribution,
    plot_multiple_trajectories,
    plot_nmf_patterns,
    plot_roc_curves,
    plot_shap_importance,  # backward-compatible alias of plot_feature_attribution
    save_fig,
)

__all__ = [
    # Shared publication style (vendored pubviz)
    "apply_pub_style",
    "PALETTE",
    "save_fig",
    # Tier 1 visualizations
    "plot_disease_trajectory",
    "plot_multiple_trajectories",
    "plot_feature_attribution",
    "plot_shap_importance",
    "plot_nmf_patterns",
    "plot_causal_dag",
    "plot_roc_curves",
    "plot_confusion_matrix",
]

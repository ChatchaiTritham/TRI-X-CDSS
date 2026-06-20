"""Patient-Level Visualization (Tier 1).

This module provides visualization functions for:
- Disease trajectory plots
- Rule-based feature attribution (DRAS-5 urgency)
- NMF temporal patterns
- Causal DAGs
- Performance metrics (ROC, confusion matrix)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from trix_cdss.constants import DEFAULT_DPI, DEFAULT_FIGURE_SIZE

# Canonical Top-Tier figure style, vendored byte-identical from
# GitHub/_management/pubviz.py (see FIGURE_STYLE.md). Single source of truth for
# the Okabe-Ito palette, publication rcParams, and the matched pdf+png saver.
from trix_cdss.visualization.pubviz import PALETTE, apply_pub_style, save_fig


def _save_fig(fig: Figure, save_path: Union[str, Path]) -> None:
    """Save vector .pdf + 300-dpi .png via the canonical pubviz.save_fig.

    Bridges the legacy ``save_path`` (a path with extension) signature used by
    these plotters onto ``save_fig(fig, basename, out_dir)``.
    """
    path = Path(save_path)
    save_fig(fig, path.stem, out_dir=path.parent if str(path.parent) else ".")


# Double-column ≈ 7.2 in, single-column ≈ 3.5 in (FIGURE_STYLE.md rule 2).
DEFAULT_TRAJECTORY_FIGSIZE: Tuple[float, float] = (7.2, 8.4)
DEFAULT_COMPARISON_FIGSIZE: Tuple[float, float] = (7.2, 6.0)
DEFAULT_BAR_PLOT_FIGSIZE: Tuple[float, float] = (7.2, 4.2)
DEFAULT_GRID_ALPHA = 0.3
DEFAULT_BAR_WIDTH = 0.2
DEFAULT_SEVERITY_SCALE_MAX = 10.5
DEFAULT_BINARY_SCALE_MAX = 1.2
DEFAULT_DRAS_SCALE_MAX = 5.5

# Faint horizontal severity bands (0-10 scale) drawn behind trajectory panels so
# the same vertical reference holds across every scenario panel. Light enough
# that the symptom lines stay dominant.
SEVERITY_ZONES: List[Tuple[float, float, str, str]] = [
    (0.0, 3.0, "#2ca02c", "Mild (0-3)"),
    (3.0, 6.0, "#fee08b", "Moderate (3-6)"),
    (6.0, 10.0, "#d73027", "Severe (6-10)"),
]


def _draw_severity_zones(ax, with_labels: bool = False) -> None:
    """Shade faint mild/moderate/severe severity bands for temporal alignment."""
    for lo, hi, color, label in SEVERITY_ZONES:
        ax.axhspan(lo, hi, alpha=0.07, color=color, zorder=0,
                   label=label if with_labels else None)


def _mark_intervention(ax, traj, label: bool = True) -> None:
    """Replicate the intervention time (e.g. Epley maneuver) as a vline.

    Reads the timing from the trajectory's interventions dict (skips the
    boolean success/flag entries), so every panel shows the same event marker.
    """
    for name, time in traj.interventions.items():
        # Skip flag entries (success/window booleans) -- only real timings.
        if name.endswith("_successful") or isinstance(time, bool):
            continue
        ax.axvline(x=time, color=PALETTE[2], linestyle="--", linewidth=1.2,
                   zorder=1, label=f"Intervention: {name}" if label else None)


def plot_disease_trajectory(
    trajectory: "DiseaseTrajectory",  # type: ignore
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[float, float] = DEFAULT_TRAJECTORY_FIGSIZE,
) -> Figure:
    """Plot disease trajectory over time.

    Creates a comprehensive visualization showing:
    - Vertigo severity over time
    - Nausea severity over time
    - Clinical signs (ataxia, nystagmus, hearing loss, tinnitus)
    - Intervention markers

    Args:
        trajectory: DiseaseTrajectory object
        save_path: Path to save figure (optional)
        show: Whether to display figure
        figsize: Figure size (width, height)

    Returns:
        Matplotlib Figure object

    Example:
        >>> from trix_cdss.tier1_patient.digital_twin import BPPVModel, generate_archetype
        >>> patient = generate_archetype("P001", DiseaseType.BPPV)
        >>> model = BPPVModel()
        >>> trajectory = model.simulate_progression(patient, [0, 1, 4, 24])
        >>> fig = plot_disease_trajectory(trajectory, save_path="bppv_trajectory.png")
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    time_points = np.array(trajectory.time_points)

    # Extract symptoms
    vertigo = [s.vertigo_severity for s in trajectory.symptom_states]
    nausea = [s.nausea_severity for s in trajectory.symptom_states]
    ataxia = [float(s.ataxia_present) for s in trajectory.symptom_states]
    nystagmus = [float(s.nystagmus_present) for s in trajectory.symptom_states]
    hearing_loss = [float(s.hearing_loss_present) for s in trajectory.symptom_states]
    tinnitus = [float(s.tinnitus_present) for s in trajectory.symptom_states]

    # Plot 1: Symptom severity (continuous). Distinct color + marker per series
    # so the panel stays legible in grayscale / for color-blind readers.
    ax1 = axes[0]
    # Faint severity zones first so they sit behind the curves (temporal anchor).
    _draw_severity_zones(ax1)
    ax1.plot(time_points, vertigo, marker="o", label="Vertigo", color=PALETTE[0])
    ax1.plot(time_points, nausea, marker="s", label="Nausea", color=PALETTE[1])
    ax1.fill_between(time_points, 0, vertigo, alpha=0.15, color=PALETTE[0])
    ax1.fill_between(time_points, 0, nausea, alpha=0.15, color=PALETTE[1])

    # Mark interventions (Epley maneuver time etc.) via the shared helper.
    _mark_intervention(ax1, trajectory)

    ax1.set_ylabel("Symptom severity (0–10)")
    ax1.set_ylim(0, DEFAULT_SEVERITY_SCALE_MAX)
    ax1.legend(loc="upper right", ncol=2)
    ax1.set_title(
        f"Disease trajectory: {trajectory.disease_type} (patient {trajectory.patient_id})"
    )

    # Plot 2: Clinical signs (binary). Hatching backs up color for accessibility.
    ax2 = axes[1]
    width = DEFAULT_BAR_WIDTH
    x_pos = np.arange(len(time_points))

    ax2.bar(x_pos - 1.5 * width, ataxia, width, label="Ataxia", color=PALETTE[0])
    ax2.bar(
        x_pos - 0.5 * width, nystagmus, width, label="Nystagmus", color=PALETTE[1], hatch="//"
    )
    ax2.bar(
        x_pos + 0.5 * width, hearing_loss, width, label="Hearing loss", color=PALETTE[2]
    )
    ax2.bar(
        x_pos + 1.5 * width, tinnitus, width, label="Tinnitus", color=PALETTE[3], hatch="\\\\"
    )

    ax2.set_ylabel("Sign present (0/1)")
    ax2.set_ylim(0, DEFAULT_BINARY_SCALE_MAX)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{t:.0f}" for t in time_points])
    ax2.legend(loc="upper right", ncol=4)
    ax2.grid(True, axis="y")

    # Plot 3: DRAS level progression (if available)
    ax3 = axes[2]
    if trajectory.predicted_dras_levels:
        dras_levels = trajectory.predicted_dras_levels

        # Color zones (severity gradient, intentionally light so the line reads).
        zone_specs = [
            (0, 1, "#2ca02c", "DRAS-1 (safe)"),
            (1, 2, "#a6d96a", "DRAS-2 (low)"),
            (2, 3, "#fee08b", "DRAS-3 (moderate)"),
            (3, 4, "#fdae61", "DRAS-4 (urgent)"),
            (4, 5, "#d73027", "DRAS-5 (emergency)"),
        ]
        for lo, hi, color, label in zone_specs:
            ax3.axhspan(lo, hi, alpha=0.12, color=color, label=label)

        ax3.plot(
            time_points,
            dras_levels,
            marker="o",
            linewidth=2.0,
            markersize=7,
            color=PALETTE[6],
            label="Predicted DRAS level",
        )

        ax3.set_ylabel("DRAS level")
        ax3.set_ylim(0, DEFAULT_DRAS_SCALE_MAX)
        ax3.set_yticks([1, 2, 3, 4, 5])
        ax3.legend(loc="upper right", ncol=3)
    else:
        ax3.text(
            0.5,
            0.5,
            "DRAS levels not available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            color="gray",
        )

    ax3.set_xlabel("Time since onset (hours)")

    # Save both vector PDF + 300-dpi PNG if requested.
    if save_path:
        _save_fig(fig, save_path)

    if show:
        plt.show()

    return fig


def plot_multiple_trajectories(
    trajectories: List["DiseaseTrajectory"],  # type: ignore
    comparison_group: str = "intervention",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[float, float] = DEFAULT_COMPARISON_FIGSIZE,
) -> Figure:
    """Plot multiple disease trajectories for comparison.

    Args:
        trajectories: List of DiseaseTrajectory objects
        comparison_group: "intervention", "disease_type", or "severity"
        save_path: Path to save figure
        show: Whether to display figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Example:
        >>> # Compare Epley immediate vs delayed
        >>> traj_immediate = model.simulate_progression(patient, [0,1,4,24], "epley", 0)
        >>> traj_delayed = model.simulate_progression(patient, [0,1,4,24], "epley", 24)
        >>> fig = plot_multiple_trajectories([traj_immediate, traj_delayed])
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Color-blind-safe series colors + distinct markers/linestyles so the three
    # scenarios separate by shape, not color alone.
    markers = ["o", "s", "^", "D", "v", "P"]
    linestyles = ["-", "--", "-.", ":"]

    # Faint severity zones behind BOTH panels so the mild/moderate/severe
    # reference lines up across the comparison.
    ax1, ax2 = axes[0], axes[1]
    _draw_severity_zones(ax1, with_labels=True)
    _draw_severity_zones(ax2)

    def _intervention_time(traj):
        """Earliest real intervention timing for this scenario (or None)."""
        for name, time in traj.interventions.items():
            if name.endswith("_successful") or isinstance(time, bool):
                continue
            return name, time
        return None

    # Plot vertigo severity
    for i, traj in enumerate(trajectories):
        time_points = np.array(traj.time_points)
        vertigo = [s.vertigo_severity for s in traj.symptom_states]
        color = PALETTE[i % len(PALETTE)]

        label = f"{traj.patient_id}"
        ev = _intervention_time(traj)
        if ev is not None:
            label += f" ({ev[0]})"

        ax1.plot(
            time_points,
            vertigo,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            label=label,
            color=color,
        )

        # Replicate this scenario's intervention marker on EVERY panel, color-
        # matched to its trajectory so the reader sees when Epley was applied.
        if ev is not None:
            for ax in (ax1, ax2):
                ax.axvline(x=ev[1], color=color, linestyle=":", linewidth=1.1,
                           alpha=0.8, zorder=1)

    ax1.set_ylabel("Vertigo severity (0–10)")
    ax1.set_ylim(0, DEFAULT_SEVERITY_SCALE_MAX)
    ax1.legend(loc="upper right", ncol=2, fontsize=8)
    ax1.set_title(f"Disease trajectory comparison ({len(trajectories)} scenarios)")

    # Plot nausea severity (same color/marker mapping as vertigo panel)
    for i, traj in enumerate(trajectories):
        time_points = np.array(traj.time_points)
        nausea = [s.nausea_severity for s in traj.symptom_states]

        ax2.plot(
            time_points,
            nausea,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            color=PALETTE[i % len(PALETTE)],
        )

    ax2.set_ylabel("Nausea severity (0–10)")
    ax2.set_xlabel("Time since onset (hours)")
    ax2.set_ylim(0, DEFAULT_SEVERITY_SCALE_MAX)

    if save_path:
        _save_fig(fig, save_path)

    if show:
        plt.show()

    return fig


def plot_feature_attribution(
    feature_importance: Dict[str, float],
    top_n: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[float, float] = DEFAULT_BAR_PLOT_FIGSIZE,
) -> Figure:
    """Plot rule-based feature attribution for DRAS-5 urgency.

    Bars show the additive per-feature contributions that the rule-based
    ``classify_urgency_level`` assigned when scoring one case. These are NOT
    Shapley (SHAP) values and no fitted explainer is involved; the input is the
    classifier's own attribution dictionary.

    Args:
        feature_importance: Dictionary {feature_name: rule_attribution_points}
        top_n: Number of top features to display
        save_path: Path to save figure
        show: Whether to display figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Example:
        >>> feature_importance = {
        ...     "hints_central": 3.0,
        ...     "atrial_fibrillation": 2.0,
        ...     "age_75": 1.5,
        ...     "symptom_duration": 1.0,
        ... }
        >>> fig = plot_feature_attribution(feature_importance)
    """
    # Sort by absolute value, then draw largest at the top (descending downward).
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[
        :top_n
    ]
    sorted_features = sorted_features[::-1]

    features = [f[0].replace("_", " ") for f in sorted_features]
    values = [f[1] for f in sorted_features]

    # Color by sign (Okabe-Ito orange/blue); bars carry sign too, so this is
    # redundant-coded rather than color-only.
    colors = [PALETTE[1] if v > 0 else PALETTE[0] for v in values]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(features, values, color=colors, edgecolor="black", linewidth=0.6)

    # Add value labels, offset to the open side of each bar to avoid overlap.
    span = max(abs(min(values)), abs(max(values))) or 1.0
    pad = 0.02 * span
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ha = "left" if value >= 0 else "right"
        ax.text(
            width + (pad if value >= 0 else -pad),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}",
            va="center",
            ha=ha,
            fontsize=9,
        )

    ax.axvline(x=0, color="black", linewidth=1.0)
    # Headroom so value labels are not clipped by the axes box.
    ax.set_xlim(min(0, min(values)) - 0.18 * span, max(0, max(values)) + 0.18 * span)
    ax.set_xlabel("Rule-based attribution to urgency score (points)")
    ax.set_title("Rule-based feature attribution (DRAS-5 urgency)")
    ax.grid(True, axis="x")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=PALETTE[1], label="Increases urgency"),
        Patch(facecolor=PALETTE[0], label="Decreases urgency"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    if save_path:
        _save_fig(fig, save_path)

    if show:
        plt.show()

    return fig


# Backward-compatible alias: the old name implied SHAP, which this is not.
# Kept so existing imports do not break; new code should use the rule-based name.
plot_shap_importance = plot_feature_attribution


def plot_nmf_patterns(
    patterns: Dict[str, np.ndarray],
    symptom_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (14, 8),
) -> Figure:
    """Plot NMF temporal patterns as heatmap.

    Args:
        patterns: Dictionary {pattern_name: array of shape [n_timepoints, n_symptoms]}
        symptom_names: List of symptom names
        save_path: Path to save figure
        show: Whether to display figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Example:
        >>> patterns = {
        ...     "Pattern_1_BPPV": np.array([[8, 5, 0, 1, 0, 0], [5, 3, 0, 1, 0, 0], ...]),
        ...     "Pattern_2_Stroke": np.array([[9, 8, 1, 1, 0, 0], [9, 8, 1, 1, 0, 0], ...]),
        ... }
        >>> symptom_names = ["Vertigo", "Nausea", "Ataxia", "Nystagmus", "Hearing Loss", "Tinnitus"]
        >>> fig = plot_nmf_patterns(patterns, symptom_names)
    """
    n_patterns = len(patterns)
    fig, axes = plt.subplots(1, n_patterns, figsize=figsize, sharey=True)

    if n_patterns == 1:
        axes = [axes]

    for i, (pattern_name, pattern_array) in enumerate(patterns.items()):
        ax = axes[i]

        # Transpose for heatmap: rows=symptoms, cols=timepoints
        data = pattern_array.T

        sns.heatmap(
            data,
            ax=ax,
            cmap="YlOrRd",
            cbar=True,
            yticklabels=symptom_names if i == 0 else False,
            xticklabels=[f"T{j}" for j in range(pattern_array.shape[0])],
            annot=True,
            fmt=".1f",
            linewidths=0.5,
            vmin=0,
            vmax=10,
        )

        ax.set_title(pattern_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time Point", fontsize=10)

        if i == 0:
            ax.set_ylabel("Symptoms", fontsize=10)

    plt.suptitle("NMF Temporal Patterns Discovery", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[OK] Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_causal_dag(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    intervention_node: Optional[str] = None,
    outcome_node: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
) -> Figure:
    """Plot causal DAG (Directed Acyclic Graph).

    Args:
        nodes: List of node names
        edges: List of edges (source, target)
        intervention_node: Node where do-operator applied
        outcome_node: Outcome node
        save_path: Path to save figure
        show: Whether to display figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Example:
        >>> nodes = ["Age", "AFib", "Stroke", "Thrombolysis", "mRS"]
        >>> edges = [("Age", "AFib"), ("AFib", "Stroke"), ("Stroke", "mRS"), ("Thrombolysis", "mRS")]
        >>> fig = plot_causal_dag(nodes, edges, intervention_node="Thrombolysis", outcome_node="mRS")
    """
    try:
        import networkx as nx
    except ImportError:
        print("[WARN] networkx not installed. Install with: pip install networkx")
        return None

    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    fig, ax = plt.subplots(figsize=figsize)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

    # Node colors
    node_colors = []
    for node in G.nodes():
        if node == intervention_node:
            node_colors.append("#2ecc71")  # Green for intervention
        elif node == outcome_node:
            node_colors.append("#e74c3c")  # Red for outcome
        else:
            node_colors.append("#3498db")  # Blue for other nodes

    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", arrows=True, arrowsize=20, arrowstyle="->", width=2, ax=ax
    )

    ax.set_title("Causal DAG with Do-Calculus Intervention", fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label=f"Intervention: do({intervention_node})", alpha=0.9),
        Patch(facecolor="#e74c3c", label=f"Outcome: {outcome_node}", alpha=0.9),
        Patch(facecolor="#3498db", label="Other Variables", alpha=0.9),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[OK] Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_roc_curves(
    y_true: List[int],
    y_scores: List[float],
    labels: List[str],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
) -> Figure:
    """Plot ROC curves for binary classification.

    Args:
        y_true: List of true labels (0/1)
        y_scores: List of predicted scores (0-1)
        labels: List of class labels
        save_path: Path to save figure
        show: Whether to display figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=figsize)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color="#e74c3c", lw=2.5, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, 0, tpr, alpha=0.2, color="#e74c3c")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curve for DRAS-5 Stroke Detection", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[OK] Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (8, 6),
) -> Figure:
    """Plot confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        save_path: Path to save figure
        show: Whether to display figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("DRAS-5 Classification Confusion Matrix", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[OK] Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig

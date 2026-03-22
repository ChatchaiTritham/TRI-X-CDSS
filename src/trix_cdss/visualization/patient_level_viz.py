"""Patient-Level Visualization (Tier 1).

This module provides visualization functions for:
- Disease trajectory plots
- SHAP feature importance
- NMF temporal patterns
- Causal DAGs
- Performance metrics (ROC, confusion matrix)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from trix_cdss.constants import DEFAULT_DPI, DEFAULT_FIGURE_SIZE

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = DEFAULT_FIGURE_SIZE
plt.rcParams["font.size"] = 11

DEFAULT_TRAJECTORY_FIGSIZE: Tuple[float, float] = (14.0, 10.0)
DEFAULT_COMPARISON_FIGSIZE: Tuple[float, float] = (14.0, 8.0)
DEFAULT_BAR_PLOT_FIGSIZE: Tuple[float, float] = (10.0, 6.0)
DEFAULT_GRID_ALPHA = 0.3
DEFAULT_BAR_WIDTH = 0.2
DEFAULT_SEVERITY_SCALE_MAX = 10.5
DEFAULT_BINARY_SCALE_MAX = 1.2
DEFAULT_DRAS_SCALE_MAX = 5.5


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

    # Plot 1: Symptom severity (continuous)
    ax1 = axes[0]
    ax1.plot(time_points, vertigo, marker="o", linewidth=2, label="Vertigo", color="#e74c3c")
    ax1.plot(time_points, nausea, marker="s", linewidth=2, label="Nausea", color="#3498db")
    ax1.fill_between(time_points, 0, vertigo, alpha=0.2, color="#e74c3c")
    ax1.fill_between(time_points, 0, nausea, alpha=0.2, color="#3498db")

    # Mark interventions
    for intervention_name, intervention_time in trajectory.interventions.items():
        if intervention_name != "epley_successful":
            ax1.axvline(
                x=intervention_time,
                color="#2ecc71",
                linestyle="--",
                linewidth=2,
                label=f"Intervention: {intervention_name}",
            )

    ax1.set_ylabel("Severity (0-10)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, DEFAULT_SEVERITY_SCALE_MAX)
    ax1.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    ax1.set_title(
        f"Disease Trajectory: {trajectory.disease_type} (Patient ID: {trajectory.patient_id})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Plot 2: Clinical signs (binary)
    ax2 = axes[1]
    width = DEFAULT_BAR_WIDTH
    x_pos = np.arange(len(time_points))

    ax2.bar(x_pos - 1.5 * width, ataxia, width, label="Ataxia", color="#e67e22", alpha=0.8)
    ax2.bar(x_pos - 0.5 * width, nystagmus, width, label="Nystagmus", color="#9b59b6", alpha=0.8)
    ax2.bar(
        x_pos + 0.5 * width, hearing_loss, width, label="Hearing Loss", color="#1abc9c", alpha=0.8
    )
    ax2.bar(x_pos + 1.5 * width, tinnitus, width, label="Tinnitus", color="#f39c12", alpha=0.8)

    ax2.set_ylabel("Present (0/1)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, DEFAULT_BINARY_SCALE_MAX)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{t:.1f}h" for t in time_points])
    ax2.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=DEFAULT_GRID_ALPHA, axis="y")

    # Plot 3: DRAS level progression (if available)
    ax3 = axes[2]
    if trajectory.predicted_dras_levels:
        dras_levels = trajectory.predicted_dras_levels
        ax3.plot(
            time_points,
            dras_levels,
            marker="o",
            linewidth=2.5,
            markersize=10,
            color="#34495e",
            label="Predicted DRAS Level",
        )
        ax3.fill_between(time_points, 0, dras_levels, alpha=0.3, color="#34495e")

        # Color zones
        ax3.axhspan(0, 1, alpha=0.1, color="green", label="Safe (DRAS-1)")
        ax3.axhspan(1, 2, alpha=0.1, color="lightgreen", label="Low (DRAS-2)")
        ax3.axhspan(2, 3, alpha=0.1, color="yellow", label="Moderate (DRAS-3)")
        ax3.axhspan(3, 4, alpha=0.1, color="orange", label="Urgent (DRAS-4)")
        ax3.axhspan(4, 5, alpha=0.1, color="red", label="Emergency (DRAS-5)")

        ax3.set_ylabel("DRAS Level", fontsize=12, fontweight="bold")
        ax3.set_ylim(0, DEFAULT_DRAS_SCALE_MAX)
        ax3.set_yticks([1, 2, 3, 4, 5])
        ax3.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, ncol=2)
    else:
        ax3.text(
            0.5,
            0.5,
            "DRAS Levels Not Available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=14,
            color="gray",
        )

    ax3.set_xlabel("Time Since Onset (hours)", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=DEFAULT_GRID_ALPHA)

    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
        print(f"[OK] Figure saved to: {save_path}")

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

    # Color palette
    colors = sns.color_palette("husl", n_colors=len(trajectories))

    # Plot vertigo severity
    ax1 = axes[0]
    for i, traj in enumerate(trajectories):
        time_points = np.array(traj.time_points)
        vertigo = [s.vertigo_severity for s in traj.symptom_states]

        label = f"{traj.patient_id}"
        if traj.interventions:
            intervention_names = [k for k in traj.interventions.keys() if k != "epley_successful"]
            if intervention_names:
                label += f" ({intervention_names[0]})"

        ax1.plot(
            time_points, vertigo, marker="o", linewidth=2, label=label, color=colors[i], alpha=0.8
        )

    ax1.set_ylabel("Vertigo Severity (0-10)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, DEFAULT_SEVERITY_SCALE_MAX)
    ax1.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=DEFAULT_GRID_ALPHA)
    ax1.set_title(
        f"Disease Trajectory Comparison ({len(trajectories)} patients)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Plot nausea severity
    ax2 = axes[1]
    for i, traj in enumerate(trajectories):
        time_points = np.array(traj.time_points)
        nausea = [s.nausea_severity for s in traj.symptom_states]

        ax2.plot(time_points, nausea, marker="s", linewidth=2, color=colors[i], alpha=0.8)

    ax2.set_ylabel("Nausea Severity (0-10)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time Since Onset (hours)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, DEFAULT_SEVERITY_SCALE_MAX)
    ax2.grid(True, alpha=DEFAULT_GRID_ALPHA)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
        print(f"[OK] Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_shap_importance(
    feature_importance: Dict[str, float],
    top_n: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[float, float] = DEFAULT_BAR_PLOT_FIGSIZE,
) -> Figure:
    """Plot SHAP feature importance bar chart.

    Args:
        feature_importance: Dictionary {feature_name: shap_value}
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
        >>> fig = plot_shap_importance(feature_importance)
    """
    # Sort by absolute value
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[
        :top_n
    ]

    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]

    # Color by positive/negative
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(features, values, color=colors, alpha=0.8, edgecolor="black")

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(
            width + 0.05 * max(abs(min(values)), abs(max(values))),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

    ax.axvline(x=0, color="black", linewidth=1.5, linestyle="-")
    ax.set_xlabel("SHAP Value (Feature Impact)", fontsize=12, fontweight="bold")
    ax.set_title(
        "SHAP Feature Importance for DRAS-5 Classification", fontsize=14, fontweight="bold", pad=15
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#e74c3c", label="Increases Urgency", alpha=0.8),
        Patch(facecolor="#3498db", label="Decreases Urgency", alpha=0.8),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[OK] Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


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

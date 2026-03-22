"""Non-negative Matrix Factorization (NMF) for Temporal Pattern Discovery.

NMF discovers latent patterns in disease trajectories by decomposing symptom
time-series data into interpretable components.

Use Cases:
1. Identify common symptom progression patterns across patients
2. Discover disease subtypes (e.g., rapid vs slow resolution BPPV)
3. Predict trajectory outcomes based on early pattern matching
4. Cluster similar patients for personalized treatment

NMF Decomposition:
    X ≈ WH
    - X: (n_patients, n_timepoints × n_symptoms) symptom matrix
    - W: (n_patients, n_patterns) patient pattern weights
    - H: (n_patterns, n_timepoints × n_symptoms) pattern templates

Clinical Interpretation:
- Pattern 1: "Rapid Resolution" (BPPV with immediate Epley)
- Pattern 2: "Gradual Compensation" (VN with VRT)
- Pattern 3: "Treatment-Resistant" (Refractory cases)
- Pattern 4: "Episodic Fluctuation" (Meniere's/VM attacks)

Advantages over PCA:
- Non-negativity constraint → interpretable parts-based representation
- Aligns with clinical thinking (additive symptoms, not subtractive)
- Patterns correspond to real symptom combinations

References:
- Lee DD, Seung HS. Nature. 1999;401:788-791. (Original NMF)
- Devarajan K. Brief Bioinform. 2008;9:184-194. (NMF for biomedical data)
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

from ..digital_twin.disease_model import DiseaseTrajectory


@dataclass
class TemporalPattern:
    """Discovered temporal pattern from NMF.

    Attributes:
        pattern_id: Pattern identifier
        pattern_name: Human-readable name
        pattern_template: (n_timepoints, n_symptoms) pattern matrix
        prevalence: Fraction of patients with this pattern
        clinical_interpretation: Clinical meaning
    """

    pattern_id: int
    pattern_name: str
    pattern_template: np.ndarray
    prevalence: float
    clinical_interpretation: str


@dataclass
class PatternAssignment:
    """Pattern assignment for single patient.

    Attributes:
        patient_id: Patient identifier
        pattern_weights: Dictionary of pattern_id -> weight
        dominant_pattern: Pattern with highest weight
        trajectory_reconstruction_error: Reconstruction error (MSE)
    """

    patient_id: str
    pattern_weights: Dict[int, float]
    dominant_pattern: int
    trajectory_reconstruction_error: float


class NMFTemporalAnalyzer:
    """NMF-based temporal pattern discovery for disease trajectories.

    Discovers latent patterns in symptom progression across patient cohorts.
    """

    def __init__(
        self,
        n_patterns: int = 5,
        symptom_features: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        """Initialize NMF temporal analyzer.

        Args:
            n_patterns: Number of patterns to discover
            symptom_features: List of symptom features to analyze
            random_state: Random seed for reproducibility
        """
        self.n_patterns = n_patterns
        self.symptom_features = symptom_features or self._get_default_symptoms()
        self.random_state = random_state

        # NMF model (initialized when fit)
        self.nmf_model = None

        # Discovered patterns
        self.patterns: List[TemporalPattern] = []

        # Pattern names (assigned after interpretation)
        self.pattern_names = {}

    def _get_default_symptoms(self) -> List[str]:
        """Get default symptom features for analysis.

        Returns:
            List of symptom feature names
        """
        return [
            "vertigo_severity",
            "nausea_severity",
            "ataxia_present",
            "nystagmus_present",
            "hearing_loss_present",
            "tinnitus_present",
        ]

    def trajectories_to_matrix(
        self,
        trajectories: List[DiseaseTrajectory],
    ) -> Tuple[np.ndarray, List[str], List[float]]:
        """Convert disease trajectories to matrix for NMF.

        Args:
            trajectories: List of disease trajectories

        Returns:
            (X, patient_ids, time_points)
            - X: (n_patients, n_timepoints × n_symptoms) matrix
            - patient_ids: List of patient IDs
            - time_points: List of time points
        """
        if not trajectories:
            raise ValueError("No trajectories provided")

        # Assume all trajectories have same time points
        time_points = trajectories[0].time_points
        n_timepoints = len(time_points)
        n_symptoms = len(self.symptom_features)

        # Initialize matrix
        n_patients = len(trajectories)
        X = np.zeros((n_patients, n_timepoints * n_symptoms))
        patient_ids = []

        for i, trajectory in enumerate(trajectories):
            patient_ids.append(trajectory.patient_id)

            # Extract symptom values at each time point
            for t_idx, state in enumerate(trajectory.symptom_states):
                for s_idx, symptom in enumerate(self.symptom_features):
                    # Get symptom value
                    if symptom == "vertigo_severity":
                        value = state.vertigo_severity
                    elif symptom == "nausea_severity":
                        value = state.nausea_severity
                    elif symptom == "ataxia_present":
                        value = 1.0 if state.ataxia_present else 0.0
                    elif symptom == "nystagmus_present":
                        value = 1.0 if state.nystagmus_present else 0.0
                    elif symptom == "hearing_loss_present":
                        value = 1.0 if state.hearing_loss_present else 0.0
                    elif symptom == "tinnitus_present":
                        value = 1.0 if state.tinnitus_present else 0.0
                    else:
                        value = 0.0

                    # Flatten: [t0_s0, t0_s1, ..., t0_sN, t1_s0, ..., tM_sN]
                    idx = t_idx * n_symptoms + s_idx
                    X[i, idx] = value

        return X, patient_ids, time_points

    def fit_patterns(
        self,
        trajectories: List[DiseaseTrajectory],
        normalize: bool = True,
    ) -> None:
        """Discover temporal patterns using NMF.

        Args:
            trajectories: List of disease trajectories
            normalize: Whether to normalize features (recommended)
        """
        # Convert trajectories to matrix
        X, patient_ids, time_points = self.trajectories_to_matrix(trajectories)

        # Normalize features (0-1 scale)
        if normalize:
            X_min = X.min(axis=0, keepdims=True)
            X_max = X.max(axis=0, keepdims=True)
            X_range = X_max - X_min
            X_range[X_range == 0] = 1.0  # Avoid division by zero
            X_normalized = (X - X_min) / X_range
        else:
            X_normalized = X

        # Fit NMF model
        self.nmf_model = NMF(
            n_components=self.n_patterns,
            init='nndsvda',  # Non-negative double SVD (deterministic)
            random_state=self.random_state,
            max_iter=500,
            alpha_W=0.1,  # L1 regularization for sparsity
            alpha_H=0.1,
        )

        W = self.nmf_model.fit_transform(X_normalized)  # Patient weights
        H = self.nmf_model.components_  # Pattern templates

        # Calculate pattern prevalence (how many patients have this as dominant)
        dominant_patterns = W.argmax(axis=1)
        pattern_prevalence = np.bincount(dominant_patterns, minlength=self.n_patterns) / len(
            trajectories
        )

        # Store discovered patterns
        n_timepoints = len(time_points)
        n_symptoms = len(self.symptom_features)

        self.patterns = []
        for p in range(self.n_patterns):
            # Reshape pattern template to (n_timepoints, n_symptoms)
            pattern_template = H[p, :].reshape(n_timepoints, n_symptoms)

            # Auto-interpret pattern
            interpretation = self._interpret_pattern(pattern_template, time_points, p)

            self.patterns.append(
                TemporalPattern(
                    pattern_id=p,
                    pattern_name=interpretation["name"],
                    pattern_template=pattern_template,
                    prevalence=float(pattern_prevalence[p]),
                    clinical_interpretation=interpretation["description"],
                )
            )

    def _interpret_pattern(
        self,
        pattern_template: np.ndarray,
        time_points: List[float],
        pattern_id: int,
    ) -> Dict[str, str]:
        """Automatically interpret pattern based on shape.

        Args:
            pattern_template: (n_timepoints, n_symptoms) pattern matrix
            time_points: Time points
            pattern_id: Pattern ID

        Returns:
            Dictionary with pattern name and description
        """
        # Analyze vertigo trajectory (index 0)
        vertigo_trajectory = pattern_template[:, 0]

        # Classify pattern shape
        initial_severity = vertigo_trajectory[0]
        final_severity = vertigo_trajectory[-1]
        peak_severity = vertigo_trajectory.max()
        peak_time_idx = vertigo_trajectory.argmax()

        # Calculate trajectory characteristics
        rapid_improvement = (initial_severity - final_severity) > 0.6 and peak_time_idx == 0
        gradual_improvement = (initial_severity - final_severity) > 0.3 and (
            initial_severity - final_severity
        ) < 0.6
        stable_severe = initial_severity > 0.7 and final_severity > 0.6
        episodic = peak_time_idx > 0 and peak_time_idx < len(vertigo_trajectory) - 1

        # Check for hearing loss (index 4)
        has_hearing_loss = pattern_template[:, 4].mean() > 0.3

        # Check for tinnitus (index 5)
        has_tinnitus = pattern_template[:, 5].mean() > 0.3

        # Pattern classification
        if rapid_improvement:
            name = "Rapid Resolution"
            description = "Rapid symptom improvement within hours (typical of BPPV with immediate Epley maneuver)"
        elif gradual_improvement and not has_hearing_loss:
            name = "Gradual Compensation"
            description = (
                "Gradual improvement over days-weeks (typical of Vestibular Neuritis with VRT)"
            )
        elif episodic and has_hearing_loss and has_tinnitus:
            name = "Episodic with Hearing Loss"
            description = "Episodic vertigo attacks with hearing loss and tinnitus (typical of Meniere's disease)"
        elif episodic and not has_hearing_loss:
            name = "Episodic Migraine"
            description = "Episodic vertigo without hearing loss (typical of Vestibular Migraine)"
        elif stable_severe:
            name = "Treatment-Resistant"
            description = "Persistent severe symptoms despite treatment (consider alternative diagnosis or complications)"
        else:
            name = f"Pattern {pattern_id + 1}"
            description = "Undetermined pattern requiring clinical review"

        return {
            "name": name,
            "description": description,
        }

    def assign_patterns(
        self,
        trajectories: List[DiseaseTrajectory],
    ) -> List[PatternAssignment]:
        """Assign patterns to patient trajectories.

        Args:
            trajectories: List of disease trajectories

        Returns:
            List of PatternAssignment objects
        """
        if self.nmf_model is None:
            raise ValueError("Must fit patterns first (call fit_patterns)")

        # Convert trajectories to matrix
        X, patient_ids, _ = self.trajectories_to_matrix(trajectories)

        # Normalize (same as training)
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1.0
        X_normalized = (X - X_min) / X_range

        # Transform to get pattern weights
        W = self.nmf_model.transform(X_normalized)

        # Reconstruct and calculate error
        X_reconstructed = W @ self.nmf_model.components_
        reconstruction_errors = ((X_normalized - X_reconstructed) ** 2).mean(axis=1)

        # Create pattern assignments
        assignments = []
        for i, patient_id in enumerate(patient_ids):
            # Normalize weights to sum to 1
            weights_normalized = W[i, :] / W[i, :].sum()

            pattern_weights = {p: float(weight) for p, weight in enumerate(weights_normalized)}

            dominant_pattern = int(W[i, :].argmax())

            assignments.append(
                PatternAssignment(
                    patient_id=patient_id,
                    pattern_weights=pattern_weights,
                    dominant_pattern=dominant_pattern,
                    trajectory_reconstruction_error=float(reconstruction_errors[i]),
                )
            )

        return assignments

    def predict_trajectory_outcome(
        self,
        trajectory: DiseaseTrajectory,
        early_timepoints: int = 3,
    ) -> Dict[str, any]:
        """Predict trajectory outcome based on early pattern matching.

        Args:
            trajectory: Disease trajectory (only early timepoints used)
            early_timepoints: Number of early timepoints to use

        Returns:
            Dictionary with predicted outcome
        """
        if self.nmf_model is None:
            raise ValueError("Must fit patterns first (call fit_patterns)")

        # Use only early timepoints
        early_trajectory = DiseaseTrajectory(
            patient_id=trajectory.patient_id,
            disease_type=trajectory.disease_type,
            time_points=trajectory.time_points[:early_timepoints],
            symptom_states=trajectory.symptom_states[:early_timepoints],
            interventions=trajectory.interventions,
        )

        # Get pattern assignment
        assignments = self.assign_patterns([early_trajectory])
        assignment = assignments[0]

        # Get dominant pattern
        pattern = self.patterns[assignment.dominant_pattern]

        # Predict final severity based on pattern template
        final_time_idx = -1  # Last timepoint in pattern
        predicted_final_severity = float(
            pattern.pattern_template[final_time_idx, 0]
        )  # Vertigo severity

        return {
            "patient_id": trajectory.patient_id,
            "dominant_pattern": pattern.pattern_name,
            "pattern_weight": assignment.pattern_weights[assignment.dominant_pattern],
            "predicted_final_vertigo_severity": predicted_final_severity,
            "clinical_interpretation": pattern.clinical_interpretation,
            "reconstruction_error": assignment.trajectory_reconstruction_error,
        }

    def get_pattern_summary(self) -> pd.DataFrame:
        """Get summary of discovered patterns.

        Returns:
            DataFrame with pattern information
        """
        if not self.patterns:
            raise ValueError("No patterns discovered. Call fit_patterns first.")

        summary_data = []
        for pattern in self.patterns:
            summary_data.append(
                {
                    "Pattern ID": pattern.pattern_id,
                    "Pattern Name": pattern.pattern_name,
                    "Prevalence (%)": f"{pattern.prevalence * 100:.1f}%",
                    "Clinical Interpretation": pattern.clinical_interpretation,
                }
            )

        return pd.DataFrame(summary_data)

    def get_pattern_template(self, pattern_id: int) -> np.ndarray:
        """Get pattern template for visualization.

        Args:
            pattern_id: Pattern ID

        Returns:
            (n_timepoints, n_symptoms) pattern matrix
        """
        if pattern_id >= len(self.patterns):
            raise ValueError(f"Pattern {pattern_id} not found")

        return self.patterns[pattern_id].pattern_template

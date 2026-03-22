"""Base Disease Model for Digital Twin Simulation.

This module provides abstract base classes for disease models and
data structures for disease trajectories.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from trix_cdss.tier1_patient.digital_twin.patient_archetype import PatientArchetype


@dataclass
class SymptomState:
    """State of symptoms at a given time point.

    Attributes:
        time_hours: Time since onset in hours
        vertigo_severity: Vertigo severity (0-10 scale)
        nausea_severity: Nausea severity (0-10 scale)
        ataxia_present: Inability to walk (boolean)
        nystagmus_present: Nystagmus observed (boolean)
        hearing_loss_present: Hearing loss (boolean)
        tinnitus_present: Tinnitus (boolean)

    Additional symptoms (disease-specific):
        extra_symptoms: Dictionary of additional symptoms
    """

    time_hours: float
    vertigo_severity: float  # 0-10
    nausea_severity: float  # 0-10
    ataxia_present: bool = False
    nystagmus_present: bool = False
    hearing_loss_present: bool = False
    tinnitus_present: bool = False

    # Disease-specific symptoms
    extra_symptoms: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate symptom values."""
        if not 0 <= self.vertigo_severity <= 10:
            raise ValueError(f"vertigo_severity must be 0-10: {self.vertigo_severity}")
        if not 0 <= self.nausea_severity <= 10:
            raise ValueError(f"nausea_severity must be 0-10: {self.nausea_severity}")

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        return {
            "time_hours": self.time_hours,
            "vertigo_severity": self.vertigo_severity,
            "nausea_severity": self.nausea_severity,
            "ataxia_present": self.ataxia_present,
            "nystagmus_present": self.nystagmus_present,
            "hearing_loss_present": self.hearing_loss_present,
            "tinnitus_present": self.tinnitus_present,
            **self.extra_symptoms,
        }


@dataclass
class DiseaseTrajectory:
    """Complete disease trajectory over time.

    Attributes:
        patient_id: Patient identifier
        disease_type: Type of disease
        time_points: List of time points (hours since onset)
        symptom_states: List of SymptomState at each time point
        interventions: Dictionary of interventions and timing
        predicted_dras_levels: Predicted DRAS levels at each time point
        actual_outcome: Actual clinical outcome (if available)
    """

    patient_id: str
    disease_type: str
    time_points: List[float]
    symptom_states: List[SymptomState]
    interventions: Dict[str, float] = field(default_factory=dict)  # intervention_name -> time
    predicted_dras_levels: List[int] = field(default_factory=list)
    actual_outcome: Optional[str] = None

    def __post_init__(self):
        """Validate trajectory."""
        if len(self.time_points) != len(self.symptom_states):
            raise ValueError("time_points and symptom_states must have same length")

    def get_symptom_at_time(self, time_hours: float) -> Optional[SymptomState]:
        """Get symptom state at specific time (interpolated if needed).

        Args:
            time_hours: Time in hours since onset

        Returns:
            SymptomState at requested time, or None if time out of range
        """
        if not self.time_points:
            return None

        # Find closest time point
        idx = np.argmin(np.abs(np.array(self.time_points) - time_hours))
        return self.symptom_states[idx]

    def to_array(self) -> np.ndarray:
        """Convert trajectory to array format for ML/NMF.

        Returns:
            Array of shape [n_timepoints, n_features]
        """
        features = []
        for state in self.symptom_states:
            features.append(
                [
                    state.vertigo_severity,
                    state.nausea_severity,
                    float(state.ataxia_present),
                    float(state.nystagmus_present),
                    float(state.hearing_loss_present),
                    float(state.tinnitus_present),
                ]
            )
        return np.array(features)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        return {
            "patient_id": self.patient_id,
            "disease_type": self.disease_type,
            "time_points": self.time_points,
            "symptom_states": [s.to_dict() for s in self.symptom_states],
            "interventions": self.interventions,
            "predicted_dras_levels": self.predicted_dras_levels,
            "actual_outcome": self.actual_outcome,
        }


class DiseaseModel(ABC):
    """Abstract base class for disease models.

    Disease models simulate the temporal progression of vestibular disorders
    under different intervention scenarios.
    """

    def __init__(self, name: str, random_seed: Optional[int] = None):
        """Initialize disease model.

        Args:
            name: Name of the disease
            random_seed: Random seed for reproducibility
        """
        self.name = name
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    @abstractmethod
    def simulate_progression(
        self,
        patient: PatientArchetype,
        time_points: List[float],
        intervention: Optional[str] = None,
        intervention_time: Optional[float] = None,
    ) -> DiseaseTrajectory:
        """Simulate disease progression over time.

        Args:
            patient: Patient archetype
            time_points: Time points to simulate (hours since onset)
            intervention: Name of intervention (if any)
            intervention_time: Time of intervention (hours since onset)

        Returns:
            DiseaseTrajectory with symptom evolution
        """
        pass

    @abstractmethod
    def get_natural_history(self, patient: PatientArchetype) -> Dict[str, float]:
        """Get natural history parameters for this patient.

        Args:
            patient: Patient archetype

        Returns:
            Dictionary with natural history parameters (resolution rate, etc.)
        """
        pass

    @abstractmethod
    def calculate_intervention_effect(
        self,
        patient: PatientArchetype,
        intervention: str,
        time_since_onset: float,
    ) -> float:
        """Calculate effect of intervention.

        Args:
            patient: Patient archetype
            intervention: Name of intervention
            time_since_onset: Time when intervention applied

        Returns:
            Effect size (e.g., symptom reduction, 0-1 scale)
        """
        pass

    def add_measurement_noise(
        self,
        true_value: float,
        patient: PatientArchetype,
    ) -> float:
        """Add measurement noise based on patient reporting accuracy.

        Args:
            true_value: True symptom value
            patient: Patient archetype

        Returns:
            Noisy symptom value
        """
        reporting_error = patient.get_symptom_reporting_error()
        noise = np.random.normal(0, reporting_error * true_value)
        return np.clip(true_value + noise, 0, 10)

    def simulate_multiple_patients(
        self,
        patients: List[PatientArchetype],
        time_points: List[float],
        intervention: Optional[str] = None,
        intervention_time: Optional[float] = None,
    ) -> List[DiseaseTrajectory]:
        """Simulate progression for multiple patients.

        Args:
            patients: List of patient archetypes
            time_points: Time points to simulate
            intervention: Intervention name
            intervention_time: Intervention timing

        Returns:
            List of disease trajectories
        """
        trajectories = []
        for patient in patients:
            trajectory = self.simulate_progression(
                patient=patient,
                time_points=time_points,
                intervention=intervention,
                intervention_time=intervention_time,
            )
            trajectories.append(trajectory)
        return trajectories

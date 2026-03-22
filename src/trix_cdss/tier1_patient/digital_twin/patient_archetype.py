"""Patient Archetype Definition for Digital Twin Simulation.

Patient archetypes represent typical patient profiles with demographics,
risk factors, and baseline characteristics used for disease simulation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class DiseaseType(Enum):
    """Types of vestibular disorders."""

    BPPV = "bppv"
    VESTIBULAR_NEURITIS = "vestibular_neuritis"
    POSTERIOR_STROKE = "posterior_stroke"
    MENIERES = "menieres"
    VESTIBULAR_MIGRAINE = "vestibular_migraine"


class Gender(Enum):
    """Patient gender."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


@dataclass
class PatientArchetype:
    """Patient archetype for digital twin simulation.

    Attributes:
        patient_id: Unique patient identifier
        age: Patient age in years
        gender: Patient gender
        disease_type: Type of vestibular disorder

    Risk Factors:
        atrial_fibrillation: AFib (stroke risk)
        hypertension: High blood pressure
        diabetes: Diabetes mellitus
        prior_stroke: Prior stroke/TIA history
        prior_bppv: Prior BPPV episodes (recurrence risk)

    Clinical Characteristics:
        health_literacy: 0-1 scale (affects symptom reporting)
        symptom_reporting_accuracy: 0-1 scale (noise in reporting)
        baseline_vestibular_function: 0-1 scale (1 = normal)

    Disease-Specific Parameters:
        disease_params: Dictionary of disease-specific parameters
    """

    # Demographics
    patient_id: str
    age: int
    gender: Gender
    disease_type: DiseaseType

    # Risk factors
    atrial_fibrillation: bool = False
    hypertension: bool = False
    diabetes: bool = False
    prior_stroke: bool = False
    prior_bppv: bool = False

    # Baseline characteristics
    health_literacy: float = 0.7  # 0-1 scale
    symptom_reporting_accuracy: float = 0.8  # 0-1 scale
    baseline_vestibular_function: float = 1.0  # 0-1 scale

    # Disease-specific parameters
    disease_params: Dict[str, any] = field(default_factory=dict)

    # Metadata
    presentation_time: Optional[datetime] = None
    symptom_onset_time: Optional[datetime] = None

    def __post_init__(self):
        """Validate archetype parameters."""
        if self.age < 0 or self.age > 120:
            raise ValueError(f"Invalid age: {self.age}")

        if not 0 <= self.health_literacy <= 1:
            raise ValueError(f"health_literacy must be 0-1: {self.health_literacy}")

        if not 0 <= self.symptom_reporting_accuracy <= 1:
            raise ValueError(
                f"symptom_reporting_accuracy must be 0-1: {self.symptom_reporting_accuracy}"
            )

        if not 0 <= self.baseline_vestibular_function <= 1:
            raise ValueError(
                f"baseline_vestibular_function must be 0-1: {self.baseline_vestibular_function}"
            )

    def calculate_stroke_risk(self) -> float:
        """Calculate stroke risk score (0-10 scale).

        Returns:
            Stroke risk score based on age and risk factors
        """
        score = 0.0

        # Age
        if self.age >= 75:
            score += 2.0
        elif self.age >= 60:
            score += 1.0

        # Risk factors
        if self.atrial_fibrillation:
            score += 3.0
        if self.prior_stroke:
            score += 3.0
        if self.hypertension:
            score += 1.0
        if self.diabetes:
            score += 1.0

        return min(score, 10.0)

    def get_symptom_reporting_error(self) -> float:
        """Get symptom reporting error standard deviation.

        Returns:
            Standard deviation for noise in symptom reporting
        """
        # Lower accuracy and health literacy → higher error
        base_error = 1.0 - self.symptom_reporting_accuracy
        literacy_factor = 1.0 - self.health_literacy

        return base_error * 0.3 + literacy_factor * 0.2

    def to_dict(self) -> Dict[str, any]:
        """Convert archetype to dictionary.

        Returns:
            Dictionary representation of archetype
        """
        return {
            "patient_id": self.patient_id,
            "age": self.age,
            "gender": self.gender.value,
            "disease_type": self.disease_type.value,
            "atrial_fibrillation": self.atrial_fibrillation,
            "hypertension": self.hypertension,
            "diabetes": self.diabetes,
            "prior_stroke": self.prior_stroke,
            "prior_bppv": self.prior_bppv,
            "health_literacy": self.health_literacy,
            "symptom_reporting_accuracy": self.symptom_reporting_accuracy,
            "baseline_vestibular_function": self.baseline_vestibular_function,
            "disease_params": self.disease_params,
            "stroke_risk_score": self.calculate_stroke_risk(),
        }


def generate_archetype(
    patient_id: str,
    disease_type: DiseaseType,
    age_range: tuple = (40, 80),
    random_seed: Optional[int] = None,
) -> PatientArchetype:
    """Generate random patient archetype.

    Args:
        patient_id: Patient identifier
        disease_type: Type of vestibular disorder
        age_range: Min and max age
        random_seed: Random seed for reproducibility

    Returns:
        Randomly generated patient archetype

    Example:
        >>> archetype = generate_archetype("P001", DiseaseType.BPPV, age_range=(50, 70))
        >>> print(archetype.age)
        63
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Age
    age = np.random.randint(age_range[0], age_range[1] + 1)

    # Gender (disease-specific distributions)
    if disease_type == DiseaseType.BPPV:
        # BPPV: 2:1 female:male
        gender = Gender.FEMALE if np.random.rand() < 0.67 else Gender.MALE
    elif disease_type == DiseaseType.VESTIBULAR_MIGRAINE:
        # Vestibular migraine: 3:1 female:male
        gender = Gender.FEMALE if np.random.rand() < 0.75 else Gender.MALE
    else:
        # Other: approximately equal
        gender = Gender.FEMALE if np.random.rand() < 0.5 else Gender.MALE

    # Risk factors (age-dependent probabilities)
    age_factor = (age - 40) / 40  # 0 at age 40, 1 at age 80

    atrial_fibrillation = np.random.rand() < (0.02 + 0.10 * age_factor)  # 2-12%
    hypertension = np.random.rand() < (0.20 + 0.40 * age_factor)  # 20-60%
    diabetes = np.random.rand() < (0.08 + 0.15 * age_factor)  # 8-23%
    prior_stroke = np.random.rand() < (0.02 + 0.05 * age_factor)  # 2-7%
    prior_bppv = (
        np.random.rand() < 0.30 if disease_type == DiseaseType.BPPV else False
    )  # 30% recurrence

    # Baseline characteristics
    health_literacy = np.clip(np.random.normal(0.7, 0.15), 0.3, 1.0)
    symptom_reporting_accuracy = np.clip(np.random.normal(0.8, 0.10), 0.5, 1.0)
    baseline_vestibular_function = np.clip(np.random.normal(1.0, 0.05), 0.8, 1.0)

    # Disease-specific parameters
    disease_params = {}

    if disease_type == DiseaseType.BPPV:
        disease_params = {
            "canal_affected": np.random.choice(
                ["posterior", "horizontal", "anterior"], p=[0.85, 0.14, 0.01]
            ),
            "severity": np.clip(np.random.normal(7.0, 1.5), 3.0, 10.0),  # 0-10 scale
        }
    elif disease_type == DiseaseType.POSTERIOR_STROKE:
        disease_params = {
            "territory": np.random.choice(["PICA", "SCA", "basilar"], p=[0.60, 0.30, 0.10]),
            "severity_nihss": np.random.randint(4, 15),  # NIHSS score
        }

    return PatientArchetype(
        patient_id=patient_id,
        age=age,
        gender=gender,
        disease_type=disease_type,
        atrial_fibrillation=atrial_fibrillation,
        hypertension=hypertension,
        diabetes=diabetes,
        prior_stroke=prior_stroke,
        prior_bppv=prior_bppv,
        health_literacy=health_literacy,
        symptom_reporting_accuracy=symptom_reporting_accuracy,
        baseline_vestibular_function=baseline_vestibular_function,
        disease_params=disease_params,
    )


def generate_archetype_cohort(
    n_patients: int,
    disease_distribution: Dict[DiseaseType, float] = None,
    random_seed: Optional[int] = None,
) -> List[PatientArchetype]:
    """Generate cohort of patient archetypes.

    Args:
        n_patients: Number of patients to generate
        disease_distribution: Disease type probabilities (default: ED distribution)
        random_seed: Random seed for reproducibility

    Returns:
        List of patient archetypes

    Example:
        >>> cohort = generate_archetype_cohort(100, random_seed=42)
        >>> len(cohort)
        100
        >>> disease_counts = {}
        >>> for patient in cohort:
        ...     disease = patient.disease_type
        ...     disease_counts[disease] = disease_counts.get(disease, 0) + 1
        >>> disease_counts[DiseaseType.BPPV] / 100  # Should be ~0.40
        0.38
    """
    if disease_distribution is None:
        # Default: Emergency department distribution
        disease_distribution = {
            DiseaseType.BPPV: 0.40,
            DiseaseType.VESTIBULAR_NEURITIS: 0.20,
            DiseaseType.VESTIBULAR_MIGRAINE: 0.15,
            DiseaseType.MENIERES: 0.10,
            DiseaseType.POSTERIOR_STROKE: 0.05,
        }

    if random_seed is not None:
        np.random.seed(random_seed)

    # Sample disease types
    disease_types = list(disease_distribution.keys())
    probabilities = list(disease_distribution.values())

    # Normalize probabilities
    probabilities = np.array(probabilities) / np.sum(probabilities)

    sampled_diseases = np.random.choice(disease_types, size=n_patients, p=probabilities)

    # Generate archetypes
    cohort = []
    for i, disease_type in enumerate(sampled_diseases):
        patient_id = f"P{i+1:04d}"
        archetype = generate_archetype(
            patient_id=patient_id,
            disease_type=disease_type,
            age_range=(40, 80),
            random_seed=random_seed + i if random_seed is not None else None,
        )
        cohort.append(archetype)

    return cohort

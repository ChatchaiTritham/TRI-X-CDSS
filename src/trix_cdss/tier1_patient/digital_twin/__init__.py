"""Digital Twin Disease Models.

This module provides disease models for vestibular disorders:
- BPPV (Benign Paroxysmal Positional Vertigo)
- Vestibular Neuritis
- Posterior Circulation Stroke
- Meniere's Disease
- Vestibular Migraine
"""

from trix_cdss.tier1_patient.digital_twin.disease_model import (
    DiseaseModel,
    DiseaseTrajectory,
    SymptomState,
)
from trix_cdss.tier1_patient.digital_twin.disease_models.bppv import BPPVModel
from trix_cdss.tier1_patient.digital_twin.disease_models.menieres import MenieresModel
from trix_cdss.tier1_patient.digital_twin.disease_models.posterior_stroke import (
    PosteriorStrokeModel,
)
from trix_cdss.tier1_patient.digital_twin.disease_models.vestibular_migraine import (
    VestibularMigraineModel,
)
from trix_cdss.tier1_patient.digital_twin.disease_models.vestibular_neuritis import (
    VestibularNeuritisModel,
)
from trix_cdss.tier1_patient.digital_twin.patient_archetype import (
    DiseaseType,
    Gender,
    PatientArchetype,
    generate_archetype,
    generate_archetype_cohort,
)

__all__ = [
    # Archetypes
    "PatientArchetype",
    "DiseaseType",
    "Gender",
    "generate_archetype",
    "generate_archetype_cohort",
    # Base models
    "DiseaseModel",
    "DiseaseTrajectory",
    "SymptomState",
    # Disease models
    "BPPVModel",
    "VestibularNeuritisModel",
    "PosteriorStrokeModel",
    "MenieresModel",
    "VestibularMigraineModel",
]

"""Disease Models for Dizziness/Vertigo Conditions.

This module contains disease-specific models for the most common causes of
dizziness presenting to the Emergency Department:

1. BPPV (40% prevalence) - Benign Paroxysmal Positional Vertigo
2. Vestibular Neuritis (20%) - Acute unilateral vestibular loss
3. Vestibular Migraine (15%) - Migraine-associated vertigo
4. Meniere's Disease (10%) - Episodic vertigo with hearing loss
5. Posterior Stroke (5%) - Time-critical central cause

Each model implements:
- Natural history simulation
- Patient-specific parameters
- Intervention effects (timing-dependent)
- DRAS-5 urgency classification
- Clinical decision support protocols
"""

from trix_cdss.tier1_patient.digital_twin.disease_models.bppv import BPPVModel
from trix_cdss.tier1_patient.digital_twin.disease_models.menieres import MenieresModel
from trix_cdss.tier1_patient.digital_twin.disease_models.posterior_stroke import (
    PosteriorStrokeModel,
    StrokeTerritory,
)
from trix_cdss.tier1_patient.digital_twin.disease_models.vestibular_migraine import (
    VestibularMigraineModel,
)
from trix_cdss.tier1_patient.digital_twin.disease_models.vestibular_neuritis import (
    VestibularNeuritisModel,
)

__all__ = [
    "BPPVModel",
    "VestibularNeuritisModel",
    "PosteriorStrokeModel",
    "StrokeTerritory",
    "MenieresModel",
    "VestibularMigraineModel",
]

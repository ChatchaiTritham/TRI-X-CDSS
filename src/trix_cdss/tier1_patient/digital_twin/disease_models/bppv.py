"""BPPV (Benign Paroxysmal Positional Vertigo) Disease Model.

BPPV is the most common cause of vertigo (40% of ED dizziness presentations).
Key characteristics:
- Episodic vertigo (30-60 seconds per episode)
- Positional triggers (lying down, turning head, looking up)
- Nystagmus during episodes
- Natural resolution: 50% at 1 week, 70% at 1 month
- Treatment: Epley maneuver (60-80% success rate for posterior canal)
"""

from typing import Dict, List, Optional

import numpy as np

from trix_cdss.tier1_patient.digital_twin.disease_model import (
    DiseaseModel,
    DiseaseTrajectory,
    SymptomState,
)
from trix_cdss.tier1_patient.digital_twin.patient_archetype import PatientArchetype


class BPPVModel(DiseaseModel):
    """Digital twin model for BPPV.

    Disease Parameters:
        - Canal affected: Posterior (85%), Horizontal (14%), Anterior (1%)
        - Episode duration: 30-60 seconds
        - Episode frequency: Varies with position changes
        - Natural resolution rate: 50% at 1 week
        - Epley success rate: 60-80% (posterior canal)
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize BPPV model.

        Args:
            random_seed: Random seed for reproducibility
        """
        super().__init__(name="BPPV", random_seed=random_seed)

        # Model parameters
        self.canal_resolution_rates = {
            "posterior": 0.50,  # 50% spontaneous resolution at 1 week
            "horizontal": 0.40,  # Slightly lower spontaneous resolution
            "anterior": 0.35,  # Lowest spontaneous resolution
        }

        self.epley_success_rates = {
            "posterior": 0.70,  # 60-80% success rate
            "horizontal": 0.60,  # Lower for horizontal canal (needs different maneuver)
            "anterior": 0.50,  # Lowest success rate
        }

    def simulate_progression(
        self,
        patient: PatientArchetype,
        time_points: List[float],
        intervention: Optional[str] = None,
        intervention_time: Optional[float] = None,
    ) -> DiseaseTrajectory:
        """Simulate BPPV progression over time.

        Args:
            patient: Patient archetype
            time_points: Time points (hours since onset)
            intervention: "epley" or "brandt_daroff" or None
            intervention_time: Time when intervention applied

        Returns:
            DiseaseTrajectory with symptom evolution
        """
        # Get disease-specific parameters
        canal = patient.disease_params.get("canal_affected", "posterior")
        severity = patient.disease_params.get("severity", 7.0)

        # Natural history parameters
        natural_resolution_rate = self.canal_resolution_rates[canal]

        # Generate symptom states at each time point
        symptom_states = []
        interventions_dict = {}

        for t in time_points:
            # Check if intervention applied before this time
            intervention_applied = (
                intervention is not None
                and intervention_time is not None
                and t >= intervention_time
            )

            if intervention_applied:
                if intervention == "epley":
                    # Epley maneuver effect
                    success_prob = self.epley_success_rates[canal]

                    # Determine if Epley was successful (happens once at intervention time)
                    if t == intervention_time or (
                        t > intervention_time and t == time_points[time_points.index(t)]
                    ):
                        epley_successful = np.random.rand() < success_prob
                        interventions_dict["epley"] = intervention_time
                        interventions_dict["epley_successful"] = epley_successful
                    else:
                        epley_successful = interventions_dict.get("epley_successful", False)

                    if epley_successful:
                        # Rapid symptom improvement after successful Epley
                        time_since_intervention = t - intervention_time
                        resolution_factor = min(
                            1.0, time_since_intervention / 4.0
                        )  # Full resolution in 4 hours
                        vertigo = severity * (1 - resolution_factor)
                    else:
                        # Epley failed, natural resolution
                        resolution_factor = min(
                            1.0, t / (168 * natural_resolution_rate)
                        )  # 168 hours = 1 week
                        vertigo = severity * (1 - resolution_factor)
                else:
                    # Unknown intervention
                    resolution_factor = min(1.0, t / (168 * natural_resolution_rate))
                    vertigo = severity * (1 - resolution_factor)
            else:
                # Natural resolution (exponential decay)
                # 50% resolution at 1 week (168 hours) for posterior canal
                half_life = 168 * natural_resolution_rate  # hours
                decay_rate = np.log(2) / half_life
                vertigo = severity * np.exp(-decay_rate * t)

            # BPPV characteristics
            # - Episodic: Simulate episodes (in reality triggered by position)
            # - For simulation, model average severity over time
            # - Episodes last 30-60 seconds, but patient reports overall impact

            # Nausea correlates with vertigo (but lower magnitude)
            nausea = vertigo * 0.6

            # Nystagmus present during episodes (here: if vertigo > 3)
            nystagmus_present = vertigo > 3.0

            # BPPV specific: No ataxia between episodes, no hearing loss, no tinnitus
            ataxia_present = False
            hearing_loss_present = False
            tinnitus_present = False

            # Add measurement noise
            vertigo_reported = self.add_measurement_noise(vertigo, patient)
            nausea_reported = self.add_measurement_noise(nausea, patient)

            # Create symptom state
            state = SymptomState(
                time_hours=t,
                vertigo_severity=vertigo_reported,
                nausea_severity=nausea_reported,
                ataxia_present=ataxia_present,
                nystagmus_present=nystagmus_present,
                hearing_loss_present=hearing_loss_present,
                tinnitus_present=tinnitus_present,
                extra_symptoms={
                    "canal_affected": canal,
                    "episode_duration_seconds": np.random.uniform(
                        30, 60
                    ),  # Typical episode duration
                    "positional_trigger": True,  # BPPV is positional
                },
            )
            symptom_states.append(state)

        # Create trajectory
        trajectory = DiseaseTrajectory(
            patient_id=patient.patient_id,
            disease_type="BPPV",
            time_points=time_points,
            symptom_states=symptom_states,
            interventions=interventions_dict,
        )

        return trajectory

    def get_natural_history(self, patient: PatientArchetype) -> Dict[str, float]:
        """Get BPPV natural history for this patient.

        Args:
            patient: Patient archetype

        Returns:
            Dictionary with natural history parameters
        """
        canal = patient.disease_params.get("canal_affected", "posterior")

        return {
            "canal_affected": canal,
            "spontaneous_resolution_1week": self.canal_resolution_rates[canal],
            "spontaneous_resolution_1month": min(1.0, self.canal_resolution_rates[canal] + 0.20),
            "recurrence_rate_6months": 0.15,  # 15% recurrence within 6 months
            "recurrence_rate_1year": 0.30,  # 30% recurrence within 1 year
            "epley_success_rate": self.epley_success_rates[canal],
        }

    def calculate_intervention_effect(
        self,
        patient: PatientArchetype,
        intervention: str,
        time_since_onset: float,
    ) -> float:
        """Calculate effect of Epley maneuver.

        Args:
            patient: Patient archetype
            intervention: "epley" or "brandt_daroff"
            time_since_onset: Time when intervention applied

        Returns:
            Probability of success (0-1)
        """
        canal = patient.disease_params.get("canal_affected", "posterior")

        if intervention == "epley":
            base_success = self.epley_success_rates[canal]

            # Factors affecting success
            # - Prior BPPV: Slightly higher success (familiar with procedure)
            if patient.prior_bppv:
                base_success += 0.05

            # - Age: Slight decrease with age (compliance/positioning difficulty)
            if patient.age > 75:
                base_success -= 0.05

            return np.clip(base_success, 0.0, 1.0)

        elif intervention == "brandt_daroff":
            # Brandt-Daroff exercises: Lower immediate success, but effective over time
            return 0.40  # 40% success over 1-2 weeks

        else:
            return 0.0

    def generate_epley_sequence(self) -> List[Dict[str, any]]:
        """Generate step-by-step Epley maneuver sequence.

        Returns:
            List of steps with timing and positioning

        This is useful for:
        - Multi-agent simulation (clinician performs Epley)
        - Workflow modeling
        - Educational materials
        """
        epley_steps = [
            {
                "step": 1,
                "action": "Dix-Hallpike test",
                "position": "Patient supine, head turned 45° toward affected ear",
                "duration_seconds": 30,
                "wait_for": "Vertigo and nystagmus to stop",
            },
            {
                "step": 2,
                "action": "Head turn",
                "position": "Turn head 90° to opposite side",
                "duration_seconds": 30,
                "wait_for": "Position maintained",
            },
            {
                "step": 3,
                "action": "Body roll",
                "position": "Roll patient to side-lying (nose down)",
                "duration_seconds": 30,
                "wait_for": "Position maintained",
            },
            {
                "step": 4,
                "action": "Sit upright",
                "position": "Slowly sit patient upright",
                "duration_seconds": 30,
                "wait_for": "Patient stabilized",
            },
            {
                "step": 5,
                "action": "Post-procedure instructions",
                "position": "Patient sitting",
                "duration_seconds": 180,  # 3 minutes for instructions
                "instructions": [
                    "Sleep semi-recumbent (45° elevation) for 1 night",
                    "Avoid rapid head movements for 24 hours",
                    "Return if symptoms persist or worsen",
                ],
            },
        ]

        return epley_steps

    def calculate_dras_level(
        self,
        symptom_state: SymptomState,
        patient: PatientArchetype,
    ) -> int:
        """Calculate DRAS level for BPPV patient.

        Args:
            symptom_state: Current symptom state
            patient: Patient archetype

        Returns:
            DRAS level (1-5)

        BPPV DRAS logic:
        - Classic BPPV (positional, episodic, no neuro signs) → DRAS-2 or DRAS-3
        - DRAS-3: ENT clinic for Epley (reduces ED crowding)
        - Higher DRAS only if concerning features (atypical, high stroke risk)
        """
        # Check for red flags
        if symptom_state.ataxia_present:
            return 5  # Ataxia with vertigo → central concern

        # Check stroke risk
        stroke_risk = patient.calculate_stroke_risk()
        if stroke_risk >= 4.0:
            return 4  # High stroke risk → urgent evaluation despite BPPV diagnosis

        # Classic BPPV
        if symptom_state.extra_symptoms.get("positional_trigger", False):
            if symptom_state.vertigo_severity >= 7.0:
                return 3  # Severe BPPV → Scheduled ENT for Epley
            else:
                return 2  # Mild-moderate BPPV → GP follow-up

        # Uncertain diagnosis
        return 3

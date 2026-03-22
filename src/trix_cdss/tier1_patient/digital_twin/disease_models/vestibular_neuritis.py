"""Vestibular Neuritis (VN) Disease Model.

Vestibular Neuritis is acute unilateral vestibular loss due to viral inflammation
of the vestibular nerve. Accounts for ~20% of ED dizziness presentations.

Clinical Features:
- Acute onset continuous vertigo (hours to days)
- Severe nausea/vomiting
- Horizontal-torsional nystagmus
- Normal hearing (vs labyrinthitis)
- Gait instability
- Symptoms worst at onset, gradual improvement over weeks

Natural History:
- Acute phase: 2-3 days of severe symptoms
- Subacute phase: 1-2 weeks of moderate symptoms
- Recovery phase: 4-12 weeks of gradual compensation
- Spontaneous recovery: ~80% at 6 months
- Central compensation: Brain adapts to unilateral loss

Interventions:
- Vestibular suppressants (acute phase only, <3 days)
  - Benefit: Symptom relief
  - Trade-off: Delays central compensation if used too long
- Corticosteroids (prednisone 60mg x 5 days)
  - Evidence mixed (some benefit if started <72hr)
- Vestibular Rehabilitation Therapy (VRT)
  - Most effective intervention
  - Accelerates central compensation
  - Should start after acute phase (day 3-7)

Clinical Decision Support:
- HINTS exam: Normal (peripheral pattern)
- Time-critical: Corticosteroids window <72hr
- DRAS-3: Scheduled neurology/ENT for VRT
- Red flags: Ataxia, vertical nystagmus, new headache → consider stroke

References:
- Strupp M, Brandt T. Neurology. 2013;81:1114-1122.
- Kim JS, et al. J Neurol. 2019;266:184-197.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..disease_model import (
    DiseaseModel,
    DiseaseTrajectory,
    SymptomState,
)
from ..patient_archetype import PatientArchetype


class VestibularNeuritisModel(DiseaseModel):
    """Vestibular Neuritis disease progression model.

    Models:
    1. Viral inflammation decay (exponential)
    2. Central compensation learning curve (sigmoid)
    3. Intervention effects:
       - Corticosteroids: Reduces inflammation
       - Vestibular suppressants: Symptom relief vs compensation delay
       - VRT: Accelerates compensation
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize VN model.

        Args:
            random_seed: Random seed for reproducibility
        """
        super().__init__(name="Vestibular_Neuritis", random_seed=random_seed)

        # Natural history parameters
        self.inflammation_half_life_hours = 72.0  # 3 days
        self.compensation_half_time_hours = 336.0  # 14 days (50% compensation)
        self.spontaneous_recovery_6months = 0.80

        # Intervention parameters
        self.corticosteroid_inflammation_reduction = 0.35  # 35% reduction
        self.corticosteroid_window_hours = 72.0  # Must start <72hr

        self.suppressant_symptom_relief = 0.60  # 60% symptom reduction
        self.suppressant_compensation_delay = 1.5  # 50% slower compensation

        self.vrt_compensation_acceleration = 2.0  # 2x faster compensation
        self.vrt_optimal_start_hours = 72.0  # Start after acute phase

    def get_natural_history(self, patient: PatientArchetype) -> Dict[str, float]:
        """Get natural history parameters for patient.

        Args:
            patient: Patient archetype

        Returns:
            Dictionary with natural history parameters
        """
        # Age affects compensation rate
        age_factor = 1.0
        if patient.age >= 65:
            age_factor = 0.7  # 30% slower compensation in elderly
        elif patient.age >= 50:
            age_factor = 0.85

        # Prior vestibular disorders
        baseline_factor = patient.disease_params.get("baseline_vestibular_function", 1.0)

        # Diabetes/vascular disease affects recovery
        vascular_factor = 1.0
        if patient.diabetes or patient.hypertension:
            vascular_factor = 0.9

        compensation_rate = age_factor * baseline_factor * vascular_factor

        return {
            "inflammation_half_life_hours": self.inflammation_half_life_hours,
            "compensation_half_time_hours": self.compensation_half_time_hours / compensation_rate,
            "spontaneous_recovery_6months": self.spontaneous_recovery_6months * compensation_rate,
            "corticosteroid_benefit": self.corticosteroid_inflammation_reduction,
            "vrt_acceleration": self.vrt_compensation_acceleration,
        }

    def _calculate_inflammation_level(
        self,
        time_hours: float,
        initial_severity: float,
        corticosteroid_given: bool = False,
        corticosteroid_time: float = 0.0,
    ) -> float:
        """Calculate inflammation level (viral infection) over time.

        Exponential decay model.

        Args:
            time_hours: Time since onset
            initial_severity: Initial inflammation (0-10)
            corticosteroid_given: Whether corticosteroids given
            corticosteroid_time: Time corticosteroids started

        Returns:
            Inflammation level (0-10)
        """
        decay_rate = np.log(2) / self.inflammation_half_life_hours
        inflammation = initial_severity * np.exp(-decay_rate * time_hours)

        # Corticosteroids reduce inflammation if started <72hr
        if corticosteroid_given and time_hours >= corticosteroid_time:
            time_since_steroid = time_hours - corticosteroid_time
            if corticosteroid_time < self.corticosteroid_window_hours:
                reduction = self.corticosteroid_inflammation_reduction
                inflammation *= 1 - reduction * min(1.0, time_since_steroid / 24.0)

        return max(0.0, inflammation)

    def _calculate_compensation_level(
        self,
        time_hours: float,
        compensation_rate: float,
        vrt_given: bool = False,
        vrt_start_time: float = 0.0,
        suppressant_duration_hours: float = 0.0,
    ) -> float:
        """Calculate central compensation level over time.

        Sigmoid learning curve. Brain adapts to unilateral vestibular loss.

        Args:
            time_hours: Time since onset
            compensation_rate: Rate of compensation (patient-specific)
            vrt_given: Whether VRT was performed
            vrt_start_time: Time VRT started
            suppressant_duration_hours: Duration of suppressant use (delays compensation)

        Returns:
            Compensation level (0-1, where 1 = fully compensated)
        """
        # Sigmoid function: starts slow, accelerates, then plateaus
        # f(t) = 1 / (1 + exp(-k*(t - t_half)))

        effective_time = time_hours

        # Vestibular suppressants delay compensation
        if suppressant_duration_hours > 0:
            delay_factor = self.suppressant_compensation_delay
            effective_time = time_hours - (suppressant_duration_hours * (delay_factor - 1))
            effective_time = max(0, effective_time)

        # VRT accelerates compensation if started appropriately
        if vrt_given and time_hours >= vrt_start_time:
            if vrt_start_time >= self.vrt_optimal_start_hours:
                # Optimal timing: after acute phase
                acceleration = self.vrt_compensation_acceleration
            else:
                # Too early: less effective
                acceleration = 1.3
            effective_time = vrt_start_time + (time_hours - vrt_start_time) * acceleration

        k = np.log(9) / self.compensation_half_time_hours  # Steepness
        t_half = self.compensation_half_time_hours

        compensation = 1.0 / (1.0 + np.exp(-k * (effective_time - t_half)))

        return min(1.0, compensation)

    def simulate_progression(
        self,
        patient: PatientArchetype,
        time_points: List[float],
        intervention: Optional[str] = None,
        intervention_time: float = 0.0,
    ) -> DiseaseTrajectory:
        """Simulate VN disease progression.

        Args:
            patient: Patient archetype
            time_points: Time points to simulate (hours since onset)
            intervention: Intervention type ("corticosteroid", "suppressant", "vrt", "combined")
            intervention_time: Time intervention applied (hours)

        Returns:
            Disease trajectory with symptom states over time
        """
        # Get patient-specific parameters
        natural_history = self.get_natural_history(patient)
        initial_severity = patient.disease_params.get("severity", 8.0)

        # Determine interventions
        corticosteroid_given = intervention in ["corticosteroid", "combined"]
        suppressant_given = intervention in ["suppressant", "combined"]
        vrt_given = intervention in ["vrt", "combined"]

        # Suppressant typically used for 2-3 days
        suppressant_duration = 72.0 if suppressant_given else 0.0

        symptom_states = []

        for t in time_points:
            # Calculate inflammation level
            inflammation = self._calculate_inflammation_level(
                t, initial_severity, corticosteroid_given, intervention_time
            )

            # Calculate compensation level
            compensation = self._calculate_compensation_level(
                t,
                1.0,  # Base rate
                vrt_given,
                intervention_time if vrt_given else 0.0,
                suppressant_duration,
            )

            # Vertigo severity = inflammation - compensation
            # Early: high inflammation, low compensation → severe vertigo
            # Late: low inflammation, high compensation → mild vertigo
            vertigo_raw = inflammation * (1 - compensation * 0.8)

            # Suppressants provide temporary symptom relief
            if (
                suppressant_given
                and intervention_time <= t < intervention_time + suppressant_duration
            ):
                vertigo_severity = vertigo_raw * (1 - self.suppressant_symptom_relief)
            else:
                vertigo_severity = vertigo_raw

            # Nausea tracks with vertigo (strong correlation in VN)
            nausea_severity = vertigo_severity * 0.9 + np.random.normal(0, 0.3)
            nausea_severity = np.clip(nausea_severity, 0, 10)

            # Clinical signs
            # Nystagmus: Present when vertigo >3, beats away from affected side
            nystagmus_present = vertigo_severity > 3.0

            # Ataxia: Gait instability, improves with compensation
            # In peripheral VN: mild ataxia, can walk with assistance
            # In central stroke: severe ataxia, cannot walk
            ataxia_present = vertigo_severity > 5.0 and compensation < 0.3

            # Hearing: NORMAL in VN (vs labyrinthitis)
            hearing_loss_present = False
            tinnitus_present = False

            state = SymptomState(
                time_hours=t,
                vertigo_severity=np.clip(vertigo_severity, 0, 10),
                nausea_severity=nausea_severity,
                ataxia_present=ataxia_present,
                nystagmus_present=nystagmus_present,
                hearing_loss_present=hearing_loss_present,
                tinnitus_present=tinnitus_present,
                extra_symptoms={
                    "inflammation_level": inflammation,
                    "compensation_level": compensation,
                    "continuous_vertigo": True,  # Continuous, not episodic
                    "gait_instability": compensation < 0.5,
                    "head_motion_worsens": True,
                    "nystagmus_direction": "horizontal_torsional",
                },
            )
            symptom_states.append(state)

        # Record interventions
        interventions_dict = {}
        if corticosteroid_given:
            interventions_dict["corticosteroid"] = intervention_time
            interventions_dict["corticosteroid_within_window"] = (
                intervention_time < self.corticosteroid_window_hours
            )
        if suppressant_given:
            interventions_dict["suppressant"] = intervention_time
            interventions_dict["suppressant_duration_hours"] = suppressant_duration
        if vrt_given:
            interventions_dict["vrt_start"] = intervention_time
            interventions_dict["vrt_optimal_timing"] = (
                intervention_time >= self.vrt_optimal_start_hours
            )

        trajectory = DiseaseTrajectory(
            patient_id=patient.patient_id,
            disease_type="Vestibular_Neuritis",
            time_points=time_points,
            symptom_states=symptom_states,
            interventions=interventions_dict,
        )

        return trajectory

    def calculate_dras_level(
        self,
        symptom_state: SymptomState,
        patient: PatientArchetype,
    ) -> int:
        """Calculate DRAS urgency level for VN patient.

        VN Classification Logic:
        - DRAS-5: Central features (severe ataxia, vertical nystagmus, focal neuro)
        - DRAS-4: High stroke risk + acute vertigo (rule out stroke)
        - DRAS-3: Classic VN (peripheral pattern, corticosteroid window)
        - DRAS-2: Subacute VN (>72hr, needs VRT referral)
        - DRAS-1: Recovery phase, mild symptoms

        Args:
            symptom_state: Current symptom state
            patient: Patient archetype

        Returns:
            DRAS urgency level (1-5)
        """
        score = 3.0  # Start at DRAS-3 baseline

        # CRITICAL: Central features suggest stroke, not VN
        if symptom_state.ataxia_present and symptom_state.vertigo_severity > 7:
            # Severe ataxia with severe vertigo → likely stroke
            score += 2.5

        nystagmus_direction = symptom_state.extra_symptoms.get("nystagmus_direction", "")
        if "vertical" in nystagmus_direction or "bidirectional" in nystagmus_direction:
            # Central nystagmus pattern
            score += 2.0

        # Stroke risk factors
        stroke_risk = patient.calculate_stroke_risk()
        if stroke_risk >= 7.0:
            score += 1.5
        elif stroke_risk >= 5.0:
            score += 0.5

        # Hearing loss suggests labyrinthitis or central cause (not simple VN)
        if symptom_state.hearing_loss_present:
            score += 1.0

        # Time window for corticosteroids (<72hr)
        if symptom_state.time_hours < 72.0 and symptom_state.vertigo_severity > 6:
            score += 0.5  # Benefit from early steroid treatment

        # Symptom severity
        if symptom_state.vertigo_severity >= 8:
            score += 1.0
        elif symptom_state.vertigo_severity <= 3:
            score -= 1.0

        # Recovery phase (high compensation)
        compensation = symptom_state.extra_symptoms.get("compensation_level", 0)
        if compensation > 0.7:
            score -= 1.5

        # Age >65 with severe symptoms
        if patient.age >= 65 and symptom_state.vertigo_severity > 6:
            score += 0.5

        # Map score to DRAS level
        if score >= 8.0:
            return 5  # Immediate emergency (likely central)
        elif score >= 6.0:
            return 4  # Urgent same-day (rule out stroke)
        elif score >= 4.0:
            return 3  # Scheduled neurology/ENT (corticosteroids + VRT)
        elif score >= 2.0:
            return 2  # Lower urgency (VRT referral)
        else:
            return 1  # Safe (self-care, recovery phase)

    def calculate_intervention_effect(
        self,
        patient: PatientArchetype,
        intervention: str,
        time_since_onset: float,
    ) -> float:
        """Calculate effect of intervention on VN.

        Args:
            patient: Patient archetype
            intervention: Intervention name
            time_since_onset: Time when intervention applied

        Returns:
            Effect size (0-1 scale, higher = better)
        """
        if intervention == "corticosteroid":
            # Effective if <72hr
            if time_since_onset < self.corticosteroid_window_hours:
                return self.corticosteroid_inflammation_reduction
            else:
                return 0.1  # Minimal benefit if late

        elif intervention == "vrt":
            # Most effective if started after acute phase (72hr+)
            if time_since_onset >= self.vrt_optimal_start_hours:
                return 0.5  # 50% faster compensation
            else:
                return 0.2  # Less effective if too early

        elif intervention == "suppressant":
            # Acute symptom relief
            return 0.6

        elif intervention == "combined":
            # Additive effects
            return 0.7

        return 0.0

    def generate_vrt_protocol(self, patient: PatientArchetype) -> List[Dict]:
        """Generate Vestibular Rehabilitation Therapy protocol.

        VRT exercises promote central compensation.

        Args:
            patient: Patient archetype

        Returns:
            List of VRT exercises with instructions
        """
        protocol = [
            {
                "exercise": "Gaze Stabilization (VOR training)",
                "description": "Hold target at arm's length, turn head side-to-side while keeping eyes on target",
                "duration_minutes": 2,
                "repetitions": 10,
                "frequency_per_day": 3,
                "difficulty": "beginner",
            },
            {
                "exercise": "Static Balance Training",
                "description": "Stand on firm surface, feet together, eyes open → eyes closed",
                "duration_minutes": 1,
                "repetitions": 5,
                "frequency_per_day": 3,
                "difficulty": "beginner",
            },
            {
                "exercise": "Dynamic Balance Training",
                "description": "Walk in straight line, turn head side-to-side while walking",
                "duration_minutes": 2,
                "repetitions": 5,
                "frequency_per_day": 3,
                "difficulty": "intermediate",
            },
            {
                "exercise": "Habituation Exercises",
                "description": "Repeatedly trigger dizziness (head turns, bending) to reduce sensitivity",
                "duration_minutes": 5,
                "repetitions": 1,
                "frequency_per_day": 2,
                "difficulty": "advanced",
            },
        ]

        # Adjust for elderly patients
        if patient.age >= 70:
            for exercise in protocol:
                exercise["repetitions"] = max(3, exercise["repetitions"] - 2)
                exercise["note"] = "Modified for elderly patient - use assistance if needed"

        return protocol

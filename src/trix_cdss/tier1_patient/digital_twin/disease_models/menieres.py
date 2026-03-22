"""Meniere's Disease Model.

Meniere's disease is characterized by episodic vertigo with fluctuating hearing loss,
tinnitus, and aural fullness. Accounts for ~10% of ED dizziness presentations.

Clinical Features:
- Episodic spontaneous vertigo (20 min - 12 hours per episode)
- Fluctuating sensorineural hearing loss (low frequencies initially)
- Tinnitus (roaring, low-pitched)
- Aural fullness (ear pressure)
- Episodes separated by symptom-free intervals

Pathophysiology:
- Endolymphatic hydrops (excess fluid in inner ear)
- Unclear etiology (autoimmune, viral, vascular theories)
- Fluctuating pressure damages cochlea and vestibular organs

Natural History:
- Variable course: some have few episodes, others have frequent attacks
- Hearing loss progressively worsens over years
- Vertigo severity may decrease over time (vestibular "burnout")
- ~80% have bilateral disease within 10-15 years

Interventions:
- Acute attack: Vestibular suppressants, antiemetics
- Preventive:
  - Low-salt diet (<1500 mg/day sodium)
  - Diuretics (HCTZ)
  - Betahistine (48 mg/day)
  - Intratympanic gentamicin (for refractory cases)
  - Endolymphatic sac decompression surgery

Clinical Decision Support:
- DRAS-2/3: Acute attack without red flags
- DRAS-4/5: If first episode, rule out stroke
- Need ENT follow-up for audiometry and long-term management

References:
- Basura GJ, et al. Otolaryngol Head Neck Surg. 2020;162:S1-S55.
- Lopez-Escamez JA, et al. J Vestib Res. 2015;25:1-7. (Diagnostic criteria)
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


class MenieresModel(DiseaseModel):
    """Meniere's disease progression model.

    Models:
    1. Episodic attacks (Poisson process for attack frequency)
    2. Attack duration (20 min - 12 hours, typically 2-4 hours)
    3. Hearing loss progression (fluctuating initially, then permanent)
    4. Intervention effects (preventive vs acute)
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize Meniere's model.

        Args:
            random_seed: Random seed for reproducibility
        """
        super().__init__(name="Menieres_Disease", random_seed=random_seed)

        # Attack parameters
        self.typical_attack_duration_hours = 3.0  # 2-4 hours typical
        self.attack_frequency_per_month = 4.0  # Variable, 1-20 attacks/month

        # Hearing loss progression
        self.hearing_loss_progression_rate_db_per_year = 5.0  # dB/year
        self.low_frequency_hearing_loss = True  # Initially affects 250-500 Hz

        # Intervention parameters
        self.diuretic_attack_reduction = 0.40  # 40% reduction in attack frequency
        self.betahistine_attack_reduction = 0.30  # 30% reduction
        self.low_salt_diet_reduction = 0.25  # 25% reduction

        self.suppressant_symptom_relief = 0.70  # Acute relief during attack

    def get_natural_history(self, patient: PatientArchetype) -> Dict[str, float]:
        """Get natural history parameters for patient.

        Args:
            patient: Patient archetype

        Returns:
            Dictionary with natural history parameters
        """
        # Disease stage
        stage = patient.disease_params.get("stage", 1)  # 1=early, 2=mid, 3=late

        # Attack frequency varies by stage
        if stage == 1:
            attack_freq = 4.0  # Early: moderate frequency
        elif stage == 2:
            attack_freq = 8.0  # Mid: high frequency
        else:
            attack_freq = 2.0  # Late: "burnout", fewer attacks

        # Baseline hearing loss (dB)
        if stage == 1:
            baseline_hearing_loss = 20.0  # Mild
        elif stage == 2:
            baseline_hearing_loss = 40.0  # Moderate
        else:
            baseline_hearing_loss = 60.0  # Severe (permanent)

        # Bilateral disease
        bilateral = patient.disease_params.get("bilateral", False)
        if bilateral:
            attack_freq *= 1.5

        return {
            "attack_frequency_per_month": attack_freq,
            "attack_duration_hours": self.typical_attack_duration_hours,
            "baseline_hearing_loss_db": baseline_hearing_loss,
            "disease_stage": stage,
            "bilateral_disease": bilateral,
            "diuretic_benefit": self.diuretic_attack_reduction,
            "betahistine_benefit": self.betahistine_attack_reduction,
        }

    def _is_in_attack(
        self,
        time_hours: float,
        attack_frequency_per_month: float,
        attack_duration_hours: float,
    ) -> Tuple[bool, float]:
        """Determine if patient is experiencing Meniere's attack at given time.

        Uses deterministic pattern based on time for reproducibility.

        Args:
            time_hours: Current time
            attack_frequency_per_month: Attacks per month
            attack_duration_hours: Duration of each attack

        Returns:
            (in_attack, severity_multiplier)
        """
        # Convert to days
        time_days = time_hours / 24.0

        # Attack cycle (deterministic for reproducibility)
        days_per_attack = 30.0 / attack_frequency_per_month
        time_in_cycle = time_days % days_per_attack

        # Attack lasts for attack_duration_hours
        attack_duration_days = attack_duration_hours / 24.0

        in_attack = time_in_cycle < attack_duration_days

        # Attack severity profile: ramps up quickly, peaks, then declines
        if in_attack:
            progress = time_in_cycle / attack_duration_days
            if progress < 0.2:
                # Ramp up (0-20%)
                severity = progress / 0.2
            elif progress < 0.6:
                # Peak (20-60%)
                severity = 1.0
            else:
                # Decline (60-100%)
                severity = 1.0 - (progress - 0.6) / 0.4
        else:
            severity = 0.0

        return in_attack, severity

    def simulate_progression(
        self,
        patient: PatientArchetype,
        time_points: List[float],
        intervention: Optional[str] = None,
        intervention_time: float = 0.0,
    ) -> DiseaseTrajectory:
        """Simulate Meniere's disease progression.

        Args:
            patient: Patient archetype
            time_points: Time points to simulate (hours since evaluation)
            intervention: Intervention type ("diuretic", "betahistine", "suppressant", "combined")
            intervention_time: Time intervention started (hours)

        Returns:
            Disease trajectory with symptom states over time
        """
        # Get patient-specific parameters
        natural_history = self.get_natural_history(patient)
        attack_freq = natural_history["attack_frequency_per_month"]
        attack_duration = natural_history["attack_duration_hours"]
        baseline_hearing = natural_history["baseline_hearing_loss_db"]

        # Determine preventive interventions
        diuretic_given = intervention in ["diuretic", "combined"]
        betahistine_given = intervention in ["betahistine", "combined"]
        suppressant_given = intervention in ["suppressant", "combined"]

        # Preventive interventions reduce attack frequency
        attack_freq_modified = attack_freq
        if diuretic_given:
            attack_freq_modified *= 1 - self.diuretic_attack_reduction
        if betahistine_given:
            attack_freq_modified *= 1 - self.betahistine_attack_reduction

        symptom_states = []

        for t in time_points:
            # Determine if in attack
            in_attack, attack_severity = self._is_in_attack(
                t, attack_freq_modified, attack_duration
            )

            # Vertigo severity during attack
            if in_attack:
                vertigo_severity = 8.0 * attack_severity  # Severe during attack

                # Suppressants reduce vertigo during attack
                if suppressant_given and t >= intervention_time:
                    vertigo_severity *= 1 - self.suppressant_symptom_relief
            else:
                # Between attacks: no vertigo
                vertigo_severity = 0.0

            # Nausea tracks with vertigo
            nausea_severity = vertigo_severity * 0.9 if in_attack else 0.0

            # Tinnitus: persistent, worsens during attacks
            tinnitus_baseline = 3.0  # Always present
            tinnitus_severity = tinnitus_baseline + (5.0 * attack_severity if in_attack else 0)

            # Hearing loss: fluctuates during attacks, progressive baseline
            # Time-dependent progression
            time_years = t / (365.25 * 24)
            progressive_loss = (
                baseline_hearing + self.hearing_loss_progression_rate_db_per_year * time_years
            )

            # During attack: additional temporary loss
            if in_attack:
                hearing_loss_db = progressive_loss + 20.0 * attack_severity
            else:
                hearing_loss_db = progressive_loss

            hearing_loss_present = hearing_loss_db > 25.0  # >25 dB = abnormal

            # Clinical signs
            # No ataxia between attacks (peripheral vestibular)
            ataxia_present = in_attack and vertigo_severity > 6.0

            # Nystagmus during attacks (horizontal)
            nystagmus_present = in_attack and vertigo_severity > 4.0

            state = SymptomState(
                time_hours=t,
                vertigo_severity=np.clip(vertigo_severity, 0, 10),
                nausea_severity=np.clip(nausea_severity, 0, 10),
                ataxia_present=ataxia_present,
                nystagmus_present=nystagmus_present,
                hearing_loss_present=hearing_loss_present,
                tinnitus_present=True,  # Always present
                extra_symptoms={
                    "in_attack": in_attack,
                    "attack_severity": attack_severity,
                    "tinnitus_severity": np.clip(tinnitus_severity, 0, 10),
                    "aural_fullness": in_attack,  # Ear pressure during attack
                    "hearing_loss_db": hearing_loss_db,
                    "low_frequency_loss": True,
                    "episodic_pattern": True,
                    "nystagmus_direction": "horizontal" if nystagmus_present else None,
                },
            )
            symptom_states.append(state)

        # Record interventions
        interventions_dict = {
            "attack_frequency_per_month": attack_freq_modified,
        }

        if diuretic_given:
            interventions_dict["diuretic"] = intervention_time

        if betahistine_given:
            interventions_dict["betahistine"] = intervention_time

        if suppressant_given:
            interventions_dict["suppressant_for_acute_attack"] = intervention_time

        trajectory = DiseaseTrajectory(
            patient_id=patient.patient_id,
            disease_type="Menieres_Disease",
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
        """Calculate DRAS urgency level for Meniere's patient.

        Meniere's Classification Logic:
        - DRAS-5: First presentation (rule out stroke)
        - DRAS-4: Severe attack + high stroke risk
        - DRAS-3: Established Meniere's, acute attack (ENT follow-up)
        - DRAS-2: Between attacks or mild attack
        - DRAS-1: Well-controlled on treatment

        Args:
            symptom_state: Current symptom state
            patient: Patient archetype

        Returns:
            DRAS urgency level (1-5)
        """
        score = 2.0  # Baseline DRAS-2

        # First episode: need to rule out stroke
        prior_menieres = patient.disease_params.get("prior_diagnosis", False)
        if not prior_menieres and symptom_state.vertigo_severity > 5:
            score += 3.0  # DRAS-5 until stroke ruled out

        # In active attack
        in_attack = symptom_state.extra_symptoms.get("in_attack", False)
        if in_attack:
            score += 1.5

        # Severity
        if symptom_state.vertigo_severity >= 8:
            score += 1.0
        elif symptom_state.vertigo_severity >= 6:
            score += 0.5

        # Stroke risk factors (always concern with acute vertigo + hearing loss)
        stroke_risk = patient.calculate_stroke_risk()
        if stroke_risk >= 7.0:
            score += 1.5
        elif stroke_risk >= 5.0:
            score += 0.5

        # Bilateral disease (more severe)
        bilateral = patient.disease_params.get("bilateral", False)
        if bilateral:
            score += 0.5

        # Well-controlled (between attacks, on treatment)
        if not in_attack and symptom_state.vertigo_severity < 2:
            score -= 1.0

        # Map score to DRAS level
        if score >= 8.0:
            return 5  # First episode or central concern
        elif score >= 6.0:
            return 4  # Urgent ENT evaluation
        elif score >= 4.0:
            return 3  # Scheduled ENT (audiometry, treatment adjustment)
        elif score >= 2.0:
            return 2  # Routine follow-up
        else:
            return 1  # Self-care

    def calculate_intervention_effect(
        self,
        patient: PatientArchetype,
        intervention: str,
        time_since_onset: float,
    ) -> float:
        """Calculate effect of intervention on Meniere's.

        Args:
            patient: Patient archetype
            intervention: Intervention name
            time_since_onset: Time when intervention applied

        Returns:
            Effect size (0-1 scale, higher = better)
        """
        if intervention == "diuretic":
            return self.diuretic_attack_reduction
        elif intervention == "betahistine":
            return self.betahistine_attack_reduction
        elif intervention == "suppressant":
            return self.suppressant_symptom_relief * 0.5  # Acute relief only
        elif intervention == "combined":
            # Additive: diuretic + betahistine + suppressant
            return min(
                0.9, self.diuretic_attack_reduction + self.betahistine_attack_reduction + 0.1
            )
        return 0.0

    def generate_treatment_protocol(
        self, patient: PatientArchetype, acute_attack: bool = True
    ) -> List[Dict]:
        """Generate Meniere's treatment protocol.

        Args:
            patient: Patient archetype
            acute_attack: Whether for acute attack or prevention

        Returns:
            List of treatment recommendations
        """
        if acute_attack:
            protocol = [
                {
                    "intervention": "Vestibular Suppressant",
                    "medication": "Meclizine 25 mg PO",
                    "frequency": "Every 6 hours PRN",
                    "duration": "During attack only (avoid prolonged use)",
                },
                {
                    "intervention": "Antiemetic",
                    "medication": "Ondansetron 4-8 mg PO/IV",
                    "frequency": "Every 8 hours PRN",
                    "duration": "During attack",
                },
                {
                    "intervention": "Rest",
                    "description": "Lie still in quiet, dark room",
                    "duration": "Until attack subsides",
                },
            ]
        else:
            # Preventive
            protocol = [
                {
                    "intervention": "Low-Salt Diet",
                    "description": "Restrict sodium to <1500 mg/day",
                    "evidence": "Reduces endolymphatic pressure",
                },
                {
                    "intervention": "Diuretic",
                    "medication": "Hydrochlorothiazide 25 mg PO daily",
                    "evidence": "40% reduction in attack frequency",
                },
                {
                    "intervention": "Betahistine",
                    "medication": "Betahistine 16 mg PO TID (48 mg/day total)",
                    "evidence": "Improves microcirculation, 30% reduction attacks",
                },
                {
                    "intervention": "ENT Follow-up",
                    "timing": "Within 1-2 weeks",
                    "tests": "Audiometry, ECoG (electrocochleography)",
                },
            ]

        return protocol

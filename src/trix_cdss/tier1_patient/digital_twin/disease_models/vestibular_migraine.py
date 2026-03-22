"""Vestibular Migraine Model.

Vestibular migraine is the second most common cause of episodic vertigo after BPPV.
Accounts for ~15% of ED dizziness presentations, often misdiagnosed.

Clinical Features:
- Episodic spontaneous or positional vertigo (5 min - 72 hours)
- History of migraine headaches (current or prior)
- Migraine features during vestibular episodes:
  - Photophobia, phonophobia
  - Visual aura
  - Headache (but not required every episode)
- Triggers: Sleep deprivation, stress, menstruation, certain foods

Diagnostic Criteria (Bárány Society):
- ≥5 episodes of vestibular symptoms (moderate-severe, 5 min - 72 hr)
- Current or prior migraine history
- ≥50% episodes with migraine features
- Not better explained by another diagnosis

Natural History:
- Onset typically 20-40 years (younger than Meniere's)
- Episodic pattern with variable frequency
- Hearing typically normal (vs Meniere's)
- Can coexist with BPPV (20% overlap)

Interventions:
- Acute attack: Triptans (if headache present), vestibular suppressants
- Preventive:
  - Migraine prophylaxis (propranolol, topiramate, amitriptyline)
  - Lifestyle modification (sleep, stress, dietary triggers)
  - Vestibular rehabilitation (for chronic dizziness)

Clinical Decision Support:
- DRAS-2/3: Established diagnosis, typical episode
- DRAS-4/5: First episode or atypical features (rule out stroke)
- Need neurology/ENT referral for diagnosis confirmation

References:
- Lempert T, et al. J Vestib Res. 2012;22:167-172. (Diagnostic criteria)
- Dieterich M, Obermann M, Celebisoy N. Nat Rev Neurol. 2016;12:469-479.
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


class VestibularMigraineModel(DiseaseModel):
    """Vestibular migraine disease progression model.

    Models:
    1. Episodic attacks (triggered pattern)
    2. Attack phases (prodrome, attack, postdrome)
    3. Migraine features (photophobia, aura, headache)
    4. Intervention effects (abortive vs preventive)
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize Vestibular Migraine model.

        Args:
            random_seed: Random seed for reproducibility
        """
        super().__init__(name="Vestibular_Migraine", random_seed=random_seed)

        # Attack parameters
        self.typical_attack_duration_hours = 8.0  # 5 min - 72 hr, typically 4-12 hr
        self.attack_frequency_per_month = 3.0  # Variable, 1-10 attacks/month

        # Migraine features (probability during attack)
        self.headache_probability = 0.75  # 75% of episodes have headache
        self.photophobia_probability = 0.85
        self.phonophobia_probability = 0.80
        self.visual_aura_probability = 0.30

        # Intervention parameters
        self.triptan_headache_relief = 0.70  # 70% headache relief if taken early
        self.triptan_vertigo_relief = 0.40  # Less effective for vertigo
        self.suppressant_vertigo_relief = 0.50

        self.prophylaxis_attack_reduction = 0.50  # 50% reduction in attack frequency
        self.lifestyle_modification_reduction = 0.30  # 30% reduction

    def get_natural_history(self, patient: PatientArchetype) -> Dict[str, float]:
        """Get natural history parameters for patient.

        Args:
            patient: Patient archetype

        Returns:
            Dictionary with natural history parameters
        """
        # Migraine history
        migraine_years = patient.disease_params.get("migraine_duration_years", 10)
        chronic_migraine = patient.disease_params.get("chronic_migraine", False)

        # Attack frequency (higher if chronic)
        if chronic_migraine:
            attack_freq = 8.0  # High frequency
        else:
            attack_freq = 3.0  # Moderate

        # Gender (women 3x more common)
        if patient.gender.value == "female":
            # Hormonal influence
            menstrual_trigger = patient.disease_params.get("menstrual_trigger", False)
            if menstrual_trigger:
                attack_freq *= 1.3

        # Age (peak 20-40 years)
        if 20 <= patient.age <= 40:
            severity_factor = 1.0
        elif patient.age < 20 or patient.age > 60:
            severity_factor = 0.7
        else:
            severity_factor = 0.85

        # Coexisting BPPV
        coexisting_bppv = patient.disease_params.get("coexisting_bppv", False)

        return {
            "attack_frequency_per_month": attack_freq,
            "attack_duration_hours": self.typical_attack_duration_hours,
            "severity_factor": severity_factor,
            "chronic_migraine": chronic_migraine,
            "coexisting_bppv": coexisting_bppv,
            "prophylaxis_benefit": self.prophylaxis_attack_reduction,
        }

    def _get_attack_phase(
        self,
        time_in_attack_hours: float,
        attack_duration_hours: float,
    ) -> Tuple[str, float]:
        """Determine current phase of migraine attack.

        Phases:
        - Prodrome (10-20%): Warning symptoms
        - Attack (60%): Peak symptoms
        - Postdrome (20-30%): Recovery, fatigue

        Args:
            time_in_attack_hours: Time since attack onset
            attack_duration_hours: Total attack duration

        Returns:
            (phase_name, intensity_multiplier)
        """
        progress = time_in_attack_hours / attack_duration_hours

        if progress < 0.15:
            # Prodrome: building symptoms
            return "prodrome", 0.3 + (progress / 0.15) * 0.4
        elif progress < 0.75:
            # Attack phase: peak symptoms
            return "attack", 1.0
        else:
            # Postdrome: declining symptoms, fatigue
            decline = (progress - 0.75) / 0.25
            return "postdrome", 1.0 - decline * 0.8

    def simulate_progression(
        self,
        patient: PatientArchetype,
        time_points: List[float],
        intervention: Optional[str] = None,
        intervention_time: float = 0.0,
    ) -> DiseaseTrajectory:
        """Simulate Vestibular Migraine disease progression.

        Args:
            patient: Patient archetype
            time_points: Time points to simulate (hours since evaluation)
            intervention: Intervention type ("triptan", "suppressant", "prophylaxis", "combined")
            intervention_time: Time intervention given (hours)

        Returns:
            Disease trajectory with symptom states over time
        """
        # Get patient-specific parameters
        natural_history = self.get_natural_history(patient)
        attack_freq = natural_history["attack_frequency_per_month"]
        attack_duration = natural_history["attack_duration_hours"]
        severity_factor = natural_history["severity_factor"]

        # Determine interventions
        triptan_given = intervention in ["triptan", "combined"]
        suppressant_given = intervention in ["suppressant", "combined"]
        prophylaxis_given = intervention in ["prophylaxis", "combined"]

        # Prophylaxis reduces attack frequency
        if prophylaxis_given:
            attack_freq *= 1 - self.prophylaxis_attack_reduction

        # Determine attack pattern (similar to Meniere's but different duration)
        days_per_attack = 30.0 / attack_freq
        attack_duration_days = attack_duration / 24.0

        symptom_states = []

        for t in time_points:
            time_days = t / 24.0
            time_in_cycle = time_days % days_per_attack

            in_attack = time_in_cycle < attack_duration_days

            if in_attack:
                time_in_attack = time_in_cycle * 24.0  # Convert to hours
                phase, intensity = self._get_attack_phase(time_in_attack, attack_duration)
            else:
                phase = "interictal"
                intensity = 0.0

            # Vertigo severity
            if in_attack:
                vertigo_severity = 7.0 * intensity * severity_factor

                # Abortive treatment
                if triptan_given and t >= intervention_time and phase != "postdrome":
                    # Triptan more effective if taken early
                    if t - intervention_time < 2.0:  # Within 2 hours
                        vertigo_severity *= 1 - self.triptan_vertigo_relief
                    else:
                        vertigo_severity *= 1 - 0.2  # Less effective if late

                if suppressant_given and t >= intervention_time:
                    vertigo_severity *= 1 - self.suppressant_vertigo_relief
            else:
                vertigo_severity = 0.0

            # Headache (migraine component)
            has_headache = in_attack and np.random.random() < self.headache_probability
            if has_headache:
                headache_severity = 6.0 * intensity

                if triptan_given and t >= intervention_time:
                    # Triptan very effective for headache
                    headache_severity *= 1 - self.triptan_headache_relief
            else:
                headache_severity = 0.0

            # Nausea
            nausea_severity = max(vertigo_severity * 0.6, headache_severity * 0.5)

            # Photophobia and phonophobia
            photophobia = in_attack and np.random.random() < self.photophobia_probability
            phonophobia = in_attack and np.random.random() < self.phonophobia_probability

            # Visual aura (before or during attack)
            visual_aura = (
                in_attack
                and phase in ["prodrome", "attack"]
                and np.random.random() < self.visual_aura_probability
            )

            # Clinical signs
            # Mild ataxia possible during severe attacks (peripheral pattern)
            ataxia_present = in_attack and vertigo_severity > 7.0

            # Nystagmus during attacks (horizontal or horizontal-torsional)
            nystagmus_present = in_attack and vertigo_severity > 4.0

            # Hearing normal (key differentiator from Meniere's)
            hearing_loss_present = False
            tinnitus_present = False

            state = SymptomState(
                time_hours=t,
                vertigo_severity=np.clip(vertigo_severity, 0, 10),
                nausea_severity=np.clip(nausea_severity, 0, 10),
                ataxia_present=ataxia_present,
                nystagmus_present=nystagmus_present,
                hearing_loss_present=hearing_loss_present,
                tinnitus_present=tinnitus_present,
                extra_symptoms={
                    "in_attack": in_attack,
                    "attack_phase": phase,
                    "attack_intensity": intensity,
                    "headache_present": has_headache,
                    "headache_severity": np.clip(headache_severity, 0, 10),
                    "photophobia": photophobia,
                    "phonophobia": phonophobia,
                    "visual_aura": visual_aura,
                    "episodic_pattern": True,
                    "nystagmus_direction": "horizontal" if nystagmus_present else None,
                    "postdrome_fatigue": phase == "postdrome",
                },
            )
            symptom_states.append(state)

        # Record interventions
        interventions_dict = {
            "attack_frequency_per_month": attack_freq,
        }

        if triptan_given:
            interventions_dict["triptan"] = intervention_time

        if suppressant_given:
            interventions_dict["suppressant"] = intervention_time

        if prophylaxis_given:
            interventions_dict["prophylaxis"] = intervention_time

        trajectory = DiseaseTrajectory(
            patient_id=patient.patient_id,
            disease_type="Vestibular_Migraine",
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
        """Calculate DRAS urgency level for Vestibular Migraine patient.

        VM Classification Logic:
        - DRAS-5: First presentation + severe (rule out stroke)
        - DRAS-4: Atypical features or high stroke risk
        - DRAS-3: Established VM, severe attack (neurology referral)
        - DRAS-2: Established VM, typical attack
        - DRAS-1: Interictal or well-controlled

        Args:
            symptom_state: Current symptom state
            patient: Patient archetype

        Returns:
            DRAS urgency level (1-5)
        """
        score = 2.0  # Baseline DRAS-2

        # First episode: need to rule out stroke
        prior_diagnosis = patient.disease_params.get("prior_diagnosis", False)
        if not prior_diagnosis and symptom_state.vertigo_severity > 6:
            score += 3.0  # DRAS-5 until stroke ruled out

        # In active attack
        in_attack = symptom_state.extra_symptoms.get("in_attack", False)
        if in_attack:
            score += 1.0

        # Severity
        if symptom_state.vertigo_severity >= 8:
            score += 1.0
        elif symptom_state.vertigo_severity >= 6:
            score += 0.5

        # Atypical features (raise concern for central cause)
        if symptom_state.hearing_loss_present:
            score += 1.0  # Unexpected in VM

        if symptom_state.ataxia_present and symptom_state.vertigo_severity > 7:
            score += 1.5  # Severe ataxia concerning

        # Stroke risk factors
        stroke_risk = patient.calculate_stroke_risk()
        if stroke_risk >= 7.0:
            score += 1.5
        elif stroke_risk >= 5.0:
            score += 0.5

        # Age >50 with new-onset migraine (atypical, concerning)
        if patient.age > 50 and not prior_diagnosis:
            score += 1.0

        # Well-controlled (interictal, on prophylaxis)
        if not in_attack and symptom_state.vertigo_severity == 0:
            score -= 1.0

        # Map score to DRAS level
        if score >= 8.0:
            return 5  # First episode or central concern
        elif score >= 6.0:
            return 4  # Urgent neurology evaluation
        elif score >= 4.0:
            return 3  # Scheduled neurology (diagnosis confirmation, prophylaxis)
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
        """Calculate effect of intervention on Vestibular Migraine.

        Args:
            patient: Patient archetype
            intervention: Intervention name
            time_since_onset: Time when intervention applied

        Returns:
            Effect size (0-1 scale, higher = better)
        """
        if intervention == "triptan":
            if time_since_onset < 2.0:
                return self.triptan_headache_relief  # Best if early
            else:
                return 0.20  # Less effective if late
        elif intervention == "suppressant":
            return self.suppressant_vertigo_relief
        elif intervention == "prophylaxis":
            return self.prophylaxis_attack_reduction
        elif intervention == "combined":
            return min(0.9, self.triptan_headache_relief + self.suppressant_vertigo_relief * 0.3)
        return 0.0

    def generate_treatment_protocol(
        self, patient: PatientArchetype, acute_attack: bool = True
    ) -> List[Dict]:
        """Generate Vestibular Migraine treatment protocol.

        Args:
            patient: Patient archetype
            acute_attack: Whether for acute attack or prevention

        Returns:
            List of treatment recommendations
        """
        if acute_attack:
            protocol = [
                {
                    "intervention": "Triptan (if headache present)",
                    "medication": "Sumatriptan 50-100 mg PO or 6 mg SC",
                    "timing": "Early in attack (<2 hours for best effect)",
                    "evidence": "70% headache relief, 40% vertigo relief",
                },
                {
                    "intervention": "Vestibular Suppressant",
                    "medication": "Meclizine 25 mg PO",
                    "frequency": "Every 6 hours PRN",
                    "duration": "Short-term only",
                },
                {
                    "intervention": "Antiemetic",
                    "medication": "Metoclopramide 10 mg PO/IV",
                    "note": "Helps vertigo and nausea",
                },
                {
                    "intervention": "Avoid Triggers",
                    "description": "Rest in quiet, dark room (photophobia, phonophobia)",
                },
            ]
        else:
            # Preventive
            protocol = [
                {
                    "intervention": "Migraine Prophylaxis",
                    "medications": [
                        "Propranolol 80-160 mg/day (first-line)",
                        "Topiramate 50-100 mg/day",
                        "Amitriptyline 25-75 mg qHS",
                    ],
                    "evidence": "50% reduction in attack frequency",
                    "duration": "3-6 months trial minimum",
                },
                {
                    "intervention": "Lifestyle Modification",
                    "recommendations": [
                        "Regular sleep schedule (7-8 hours)",
                        "Stress management",
                        "Identify dietary triggers (caffeine, alcohol, MSG, aged cheese)",
                        "Regular exercise (but avoid overexertion triggers)",
                    ],
                    "evidence": "30% reduction in attacks",
                },
                {
                    "intervention": "Vestibular Rehabilitation",
                    "indication": "For chronic dizziness between attacks",
                    "referral": "Physical therapy",
                },
                {
                    "intervention": "Neurology Follow-up",
                    "timing": "Within 2-4 weeks",
                    "purpose": "Confirm diagnosis, optimize prophylaxis",
                },
            ]

        return protocol

"""Posterior Circulation Stroke Disease Model.

Posterior circulation strokes (PICA, SCA, basilar) present with dizziness/vertigo
and are the most critical time-sensitive condition in ED dizziness evaluation.
Accounts for ~5% of ED dizziness presentations but has highest morbidity/mortality.

Clinical Features:
- Acute onset continuous vertigo (sudden)
- Central nystagmus (vertical, bidirectional, or direction-changing)
- Severe ataxia (cannot walk without assistance)
- Focal neurological signs (diplopia, dysarthria, dysphagia, numbness)
- May have normal initial MRI (DWI-negative in first 24-48hr)
- HINTS exam: Central pattern

Vascular Territories:
- PICA (Posterior Inferior Cerebellar Artery): Lateral medullary syndrome (Wallenberg)
- SCA (Superior Cerebellar Artery): Cerebellar infarct, ataxia
- Basilar: Life-threatening, brainstem ischemia

Natural History:
- WITHOUT thrombolysis: Progressive worsening, infarct expansion
- WITH thrombolysis (<4.5hr): 40% good outcome (mRS 0-1)
- Time is brain: Every 15 min delay = 4% reduced odds of good outcome

Interventions:
- IV Thrombolysis (tPA) <4.5hr from onset
  - Benefit: 30% absolute increase in good outcome if <90min
  - Risk: 6% symptomatic hemorrhage
  - Contraindications: Recent surgery, bleeding, uncontrolled BP >185/110
- Mechanical Thrombectomy <24hr (if large vessel occlusion)
- Antiplatelet therapy (aspirin/clopidogrel)
- Blood pressure management

Clinical Decision Support:
- DRAS-5: Immediate emergency (stroke code, CT <15min)
- Time window: <4.5hr for tPA (golden hour <90min)
- HINTS exam sensitivity: 96.5% for stroke vs peripheral
- ABCD2 score for risk stratification

References:
- Newman-Toker DE, et al. Stroke. 2021;52:227-236.
- Tarnutzer AA, et al. JAMA Neurol. 2011;68:1010-1016.
- Powers WJ, et al. Stroke. 2019;50:e344-e418. (AHA/ASA Guidelines)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..disease_model import (
    DiseaseModel,
    DiseaseTrajectory,
    SymptomState,
)
from ..patient_archetype import PatientArchetype


class StrokeTerritory(Enum):
    """Posterior circulation stroke territories."""

    PICA = "PICA"  # Posterior Inferior Cerebellar Artery
    SCA = "SCA"  # Superior Cerebellar Artery
    BASILAR = "basilar"  # Basilar artery


class PosteriorStrokeModel(DiseaseModel):
    """Posterior circulation stroke disease progression model.

    Models:
    1. Infarct growth over time (diffusion → perfusion → core)
    2. NIHSS score progression
    3. mRS outcome based on intervention timing
    4. Hemorrhagic transformation risk
    """

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize Posterior Stroke model.

        Args:
            random_seed: Random seed for reproducibility
        """
        super().__init__(name="Posterior_Stroke", random_seed=random_seed)

        # Thrombolysis parameters
        self.thrombolysis_window_hours = 4.5
        self.golden_hour = 1.5  # 90 minutes
        self.hemorrhage_risk = 0.06  # 6% symptomatic ICH

        # Treatment effect (NNT = 7 for tPA)
        self.tpa_benefit_golden_hour = 0.30  # 30% absolute increase good outcome
        self.tpa_benefit_3to4_5hr = 0.15  # 15% benefit if 3-4.5hr

        # Mechanical thrombectomy
        self.thrombectomy_window_hours = 24.0
        self.thrombectomy_benefit = 0.40  # 40% benefit for large vessel occlusion

        # Infarct growth rate (time is brain)
        # 1.9 million neurons lost per minute in untreated stroke
        self.infarct_growth_rate_per_hour = 0.15  # 15% increase per hour

    def get_natural_history(self, patient: PatientArchetype) -> Dict[str, float]:
        """Get natural history parameters for patient.

        Args:
            patient: Patient archetype

        Returns:
            Dictionary with natural history parameters
        """
        # Territory-specific parameters
        territory = patient.disease_params.get("territory", "PICA")

        baseline_nihss = {
            "PICA": 5.0,  # Lateral medullary: vertigo, ataxia, sensory
            "SCA": 6.0,  # Cerebellar: severe ataxia, nystagmus
            "basilar": 12.0,  # Brainstem: life-threatening
        }.get(territory, 5.0)

        # Large vessel occlusion
        lvo = patient.disease_params.get("large_vessel_occlusion", False)

        # Hemorrhage risk factors
        hemorrhage_risk = self.hemorrhage_risk
        if patient.hypertension:
            hemorrhage_risk *= 1.5
        if patient.age >= 75:
            hemorrhage_risk *= 1.3
        if patient.disease_params.get("anticoagulation", False):
            hemorrhage_risk *= 2.0

        return {
            "baseline_nihss": baseline_nihss,
            "territory": territory,
            "large_vessel_occlusion": lvo,
            "thrombolysis_window_hours": self.thrombolysis_window_hours,
            "golden_hour": self.golden_hour,
            "hemorrhage_risk": hemorrhage_risk,
            "tpa_benefit": self.tpa_benefit_golden_hour if lvo else self.tpa_benefit_3to4_5hr,
            "thrombectomy_eligible": lvo,
        }

    def _calculate_nihss_score(
        self,
        time_hours: float,
        baseline_nihss: float,
        territory: str,
        tpa_given: bool = False,
        tpa_time: float = 0.0,
        tpa_successful: bool = False,
    ) -> float:
        """Calculate NIHSS (National Institutes of Health Stroke Scale) over time.

        NIHSS range: 0-42
        - 0: No stroke symptoms
        - 1-4: Minor stroke
        - 5-15: Moderate stroke
        - 16-20: Moderate-severe stroke
        - 21-42: Severe stroke

        Args:
            time_hours: Time since onset
            baseline_nihss: Initial NIHSS at presentation
            territory: Stroke territory
            tpa_given: Whether tPA was administered
            tpa_time: Time tPA given
            tpa_successful: Whether tPA resulted in recanalization

        Returns:
            NIHSS score
        """
        # Natural history: progressive worsening due to infarct growth
        nihss = baseline_nihss * (1 + self.infarct_growth_rate_per_hour * time_hours)

        # Cap progression at 24 hours (core infarct established)
        nihss = min(nihss, baseline_nihss * 2.5)

        # tPA effect
        if tpa_given and time_hours >= tpa_time:
            time_since_tpa = time_hours - tpa_time

            if tpa_successful:
                # Successful recanalization: rapid improvement
                # 50% improvement by 24 hours
                improvement_factor = min(0.5, time_since_tpa / 24.0 * 0.5)
                nihss = baseline_nihss * (1 - improvement_factor)
            else:
                # Failed recanalization: minimal benefit
                # 10% improvement
                improvement_factor = min(0.1, time_since_tpa / 24.0 * 0.1)
                nihss = (
                    baseline_nihss
                    * (1 + self.infarct_growth_rate_per_hour * tpa_time)
                    * (1 - improvement_factor)
                )

        return np.clip(nihss, 0, 42)

    def _calculate_mrs_outcome(
        self,
        final_nihss: float,
        tpa_given: bool,
        tpa_within_golden_hour: bool,
        hemorrhage_occurred: bool,
    ) -> int:
        """Calculate modified Rankin Scale (mRS) outcome.

        mRS scale:
        - 0: No symptoms
        - 1: No significant disability (can do usual activities)
        - 2: Slight disability (can look after own affairs without assistance)
        - 3: Moderate disability (requires some help, can walk without assistance)
        - 4: Moderately severe disability (cannot walk/attend to bodily needs without assistance)
        - 5: Severe disability (bedridden, incontinent, requires constant care)
        - 6: Dead

        Args:
            final_nihss: NIHSS score at 90 days
            tpa_given: Whether tPA was given
            tpa_within_golden_hour: Whether tPA given <90min
            hemorrhage_occurred: Whether symptomatic hemorrhage occurred

        Returns:
            mRS score (0-6)
        """
        # Map NIHSS to mRS (approximate)
        if final_nihss == 0:
            mrs = 0
        elif final_nihss <= 2:
            mrs = 1
        elif final_nihss <= 5:
            mrs = 2
        elif final_nihss <= 10:
            mrs = 3
        elif final_nihss <= 15:
            mrs = 4
        elif final_nihss <= 20:
            mrs = 5
        else:
            mrs = 5  # Severe disability

        # tPA benefit: shift toward better outcome
        if tpa_given and tpa_within_golden_hour:
            # 30% absolute increase in mRS 0-1
            if np.random.random() < 0.30:
                mrs = max(0, mrs - 1)

        # Hemorrhage worsens outcome
        if hemorrhage_occurred:
            mrs = min(6, mrs + 2)

        return mrs

    def simulate_progression(
        self,
        patient: PatientArchetype,
        time_points: List[float],
        intervention: Optional[str] = None,
        intervention_time: float = 0.0,
    ) -> DiseaseTrajectory:
        """Simulate posterior stroke disease progression.

        Args:
            patient: Patient archetype
            time_points: Time points to simulate (hours since onset)
            intervention: Intervention type ("tpa", "thrombectomy", "antiplatelet")
            intervention_time: Time intervention applied (hours)

        Returns:
            Disease trajectory with symptom states over time
        """
        # Get patient-specific parameters
        natural_history = self.get_natural_history(patient)
        baseline_nihss = natural_history["baseline_nihss"]
        territory = natural_history["territory"]

        # Determine intervention
        tpa_given = intervention in ["tpa", "combined"]
        thrombectomy_given = intervention in ["thrombectomy", "combined"]

        # tPA success depends on timing
        tpa_successful = False
        tpa_within_golden_hour = False
        if tpa_given:
            tpa_within_golden_hour = intervention_time < self.golden_hour

            # Success rate decreases with time
            if intervention_time < self.golden_hour:
                success_rate = 0.70  # 70% recanalization
            elif intervention_time < 3.0:
                success_rate = 0.50  # 50%
            elif intervention_time < self.thrombolysis_window_hours:
                success_rate = 0.35  # 35%
            else:
                success_rate = 0.0  # Outside window

            tpa_successful = np.random.random() < success_rate

        # Hemorrhagic transformation
        hemorrhage_occurred = False
        if tpa_given and np.random.random() < natural_history["hemorrhage_risk"]:
            hemorrhage_occurred = True

        symptom_states = []

        for t in time_points:
            # Calculate NIHSS
            nihss = self._calculate_nihss_score(
                t, baseline_nihss, territory, tpa_given, intervention_time, tpa_successful
            )

            # Vertigo severity correlates with NIHSS for posterior strokes
            # Severe vertigo at onset, but overshadowed by focal neuro signs
            vertigo_severity = min(10.0, 5.0 + nihss * 0.3)

            # If improving with tPA, vertigo also improves
            if tpa_given and tpa_successful and t > intervention_time:
                vertigo_severity *= 0.6

            # Nausea
            nausea_severity = min(10.0, vertigo_severity * 0.7 + nihss * 0.2)

            # Clinical signs
            # Severe ataxia (CENTRAL feature)
            ataxia_present = nihss > 4.0

            # Nystagmus: Central pattern (vertical, bidirectional)
            nystagmus_present = nihss > 3.0

            # Hearing loss (rare, if AICA territory)
            hearing_loss_present = territory == "PICA" and np.random.random() < 0.1

            # Tinnitus (uncommon)
            tinnitus_present = False

            # Focal neurological signs
            focal_neuro = {
                "diplopia": nihss > 5.0 and territory in ["BASILAR", "SCA"],
                "dysarthria": nihss > 4.0 and territory == "PICA",
                "dysphagia": nihss > 6.0 and territory == "PICA",
                "facial_numbness": territory == "PICA",
                "limb_weakness": territory == "BASILAR",
                "gcs_score": max(3, 15 - int(nihss / 3)),  # Decreased consciousness
            }

            state = SymptomState(
                time_hours=t,
                vertigo_severity=np.clip(vertigo_severity, 0, 10),
                nausea_severity=np.clip(nausea_severity, 0, 10),
                ataxia_present=ataxia_present,
                nystagmus_present=nystagmus_present,
                hearing_loss_present=hearing_loss_present,
                tinnitus_present=tinnitus_present,
                extra_symptoms={
                    "nihss_score": nihss,
                    "focal_neurological_signs": focal_neuro,
                    "nystagmus_direction": "vertical_or_bidirectional",  # CENTRAL
                    "severe_ataxia": ataxia_present and nihss > 8,
                    "altered_consciousness": focal_neuro["gcs_score"] < 15,
                    "sudden_onset": t == 0,  # Stroke is sudden
                    "continuous_symptoms": True,
                    "territory": territory,
                },
            )
            symptom_states.append(state)

        # Calculate final mRS outcome (at 90 days)
        final_nihss = self._calculate_nihss_score(
            2160, baseline_nihss, territory, tpa_given, intervention_time, tpa_successful  # 90 days
        )
        mrs_outcome = self._calculate_mrs_outcome(
            final_nihss, tpa_given, tpa_within_golden_hour, hemorrhage_occurred
        )

        # Record interventions
        interventions_dict = {
            "stroke_territory": territory,
            "baseline_nihss": baseline_nihss,
            "final_mrs_90day": mrs_outcome,
        }

        if tpa_given:
            interventions_dict["tpa_given"] = intervention_time
            interventions_dict["tpa_within_window"] = (
                intervention_time < self.thrombolysis_window_hours
            )
            interventions_dict["tpa_within_golden_hour"] = tpa_within_golden_hour
            interventions_dict["tpa_successful"] = tpa_successful
            interventions_dict["hemorrhage_occurred"] = hemorrhage_occurred

        if thrombectomy_given:
            interventions_dict["thrombectomy"] = intervention_time

        trajectory = DiseaseTrajectory(
            patient_id=patient.patient_id,
            disease_type="Posterior_Stroke",
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
        """Calculate DRAS urgency level for stroke patient.

        Stroke Classification Logic:
        - DRAS-5: ALWAYS (stroke is immediate emergency)
        - Time window: <4.5hr for tPA (activate stroke code)
        - HINTS exam: Central features
        - Focal neurological signs

        Args:
            symptom_state: Current symptom state
            patient: Patient archetype

        Returns:
            DRAS urgency level (always 5 for stroke)
        """
        # Stroke is ALWAYS DRAS-5 (immediate emergency)
        score = 10.0

        # Within thrombolysis window
        if symptom_state.time_hours < self.thrombolysis_window_hours:
            score += 2.0

        # Golden hour (<90min)
        if symptom_state.time_hours < self.golden_hour:
            score += 1.0

        # Central features
        nihss = symptom_state.extra_symptoms.get("nihss_score", 0)
        if nihss > 10:
            score += 2.0
        elif nihss > 5:
            score += 1.0

        # Severe ataxia
        if symptom_state.extra_symptoms.get("severe_ataxia", False):
            score += 1.0

        # Altered consciousness
        if symptom_state.extra_symptoms.get("altered_consciousness", False):
            score += 2.0

        # Stroke is ALWAYS DRAS-5
        return 5

    def calculate_intervention_effect(
        self,
        patient: PatientArchetype,
        intervention: str,
        time_since_onset: float,
    ) -> float:
        """Calculate effect of intervention on stroke outcome.

        Args:
            patient: Patient archetype
            intervention: Intervention name
            time_since_onset: Time when intervention applied

        Returns:
            Effect size (0-1 scale, higher = better)
        """
        if intervention == "tpa":
            if time_since_onset < self.golden_hour:
                return 0.70
            elif time_since_onset < 3.0:
                return 0.50
            elif time_since_onset < self.thrombolysis_window_hours:
                return 0.35
            else:
                return 0.0

        elif intervention == "thrombectomy":
            if time_since_onset < 6.0:
                return 0.50
            elif time_since_onset < self.thrombectomy_window_hours:
                return 0.30
            else:
                return 0.0

        elif intervention == "antiplatelet":
            return 0.15

        return 0.0

    def generate_stroke_code_protocol(self, patient: PatientArchetype) -> List[Dict]:
        """Generate stroke code activation protocol.

        Args:
            patient: Patient archetype

        Returns:
            List of time-critical actions
        """
        protocol = [
            {
                "action": "Activate Stroke Code",
                "time_target_minutes": 0,
                "description": "Notify stroke team, activate CT",
            },
            {
                "action": "Neurological Assessment",
                "time_target_minutes": 5,
                "description": "NIHSS score, HINTS exam, GCS",
            },
            {
                "action": "Non-Contrast Head CT",
                "time_target_minutes": 15,
                "description": "Rule out hemorrhage, assess early ischemic changes",
            },
            {
                "action": "Labs + Contraindication Screen",
                "time_target_minutes": 20,
                "description": "CBC, PT/INR, blood sugar, BP monitoring",
            },
            {
                "action": "Neurology Consultation",
                "time_target_minutes": 30,
                "description": "Stroke neurologist assessment, tPA decision",
            },
            {
                "action": "IV Thrombolysis (if indicated)",
                "time_target_minutes": 45,
                "description": "tPA 0.9 mg/kg (max 90mg), 10% bolus then infusion over 60min",
            },
            {
                "action": "CTA/MRI (if thrombectomy candidate)",
                "time_target_minutes": 60,
                "description": "Identify large vessel occlusion",
            },
            {
                "action": "Transfer to Stroke Unit / ICU",
                "time_target_minutes": 90,
                "description": "Continuous monitoring, BP management",
            },
        ]

        return protocol

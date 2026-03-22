"""Counterfactual Reasoning for Causal "What-If" Analysis.

Counterfactual reasoning answers causal questions about treatment effects:
- "What if this patient had received Epley immediately vs delayed?"
- "What if thrombolysis was given at T+1hr vs T+3hr?"
- "What would have happened without intervention?"

This is the foundation of causal inference in TRI-X-CDSS. It enables:
1. Individual Treatment Effect (ITE) estimation
2. Optimal treatment timing recommendations
3. Evidence-based clinical decision support
4. Validation of disease models

Mathematical Framework:
- Potential Outcomes: Y(1) = outcome if treated, Y(0) = outcome if untreated
- Individual Treatment Effect: ITE = Y(1) - Y(0)
- Counterfactual: Estimate Y(1) for untreated patient (or Y(0) for treated)

Approaches:
1. Digital Twin Simulation (used here): Directly simulate counterfactual trajectories
2. Propensity Score Matching: Match similar patients
3. Instrumental Variables: Use natural experiments
4. Regression Discontinuity: Exploit treatment thresholds

Clinical Applications:
- "This BPPV patient received Epley at ED visit. If delayed to ENT clinic (1 week),
   predicted vertigo reduction would be 2.3 points less."
- "This stroke patient arrived at T+2hr. If arrived at T+1hr (golden hour),
   predicted mRS would improve from 3 to 1 (independent vs moderate disability)."

References:
- Pearl J. Causality. 2009. (Counterfactual framework)
- Hernán MA, Robins JM. Causal Inference. 2020. (G-formula, potential outcomes)
- Shalit U, et al. ICML 2017. (ITE estimation with representation learning)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..digital_twin.disease_model import DiseaseModel, DiseaseTrajectory, SymptomState
from ..digital_twin.patient_archetype import PatientArchetype


class InterventionType(Enum):
    """Types of interventions for counterfactual analysis."""

    NONE = "none"
    EPLEY = "epley"
    CORTICOSTEROID = "corticosteroid"
    VRT = "vrt"
    TPA = "tpa"
    SUPPRESSANT = "suppressant"
    PROPHYLAXIS = "prophylaxis"


@dataclass
class CounterfactualScenario:
    """Counterfactual scenario specification.

    Attributes:
        scenario_name: Human-readable name
        intervention: Intervention type
        intervention_time: Time intervention applied (hours)
        description: Clinical description
    """

    scenario_name: str
    intervention: InterventionType
    intervention_time: float
    description: str


@dataclass
class CounterfactualOutcome:
    """Outcome under counterfactual scenario.

    Attributes:
        scenario_name: Scenario name
        final_vertigo_severity: Vertigo severity at final timepoint
        final_dras_level: DRAS level at final timepoint
        symptom_free_time: Time to symptom resolution (hours)
        total_symptom_burden: Area under vertigo curve (AUC)
        adverse_events: List of adverse events
    """

    scenario_name: str
    final_vertigo_severity: float
    final_dras_level: int
    symptom_free_time: Optional[float]
    total_symptom_burden: float
    adverse_events: List[str]


@dataclass
class TreatmentEffect:
    """Individual treatment effect (ITE) estimation.

    Attributes:
        patient_id: Patient identifier
        treatment_scenario: Name of treatment scenario
        control_scenario: Name of control scenario
        effect_on_vertigo: Difference in final vertigo severity
        effect_on_symptom_burden: Difference in total symptom burden
        effect_on_time_to_resolution: Difference in time to resolution
        clinical_interpretation: Human-readable interpretation
    """

    patient_id: str
    treatment_scenario: str
    control_scenario: str
    effect_on_vertigo: float
    effect_on_symptom_burden: float
    effect_on_time_to_resolution: float
    clinical_interpretation: str


class CounterfactualAnalyzer:
    """Counterfactual reasoning for treatment effect estimation.

    Uses digital twin simulation to estimate counterfactual outcomes.
    """

    def __init__(
        self,
        disease_model: DiseaseModel,
        time_points: Optional[List[float]] = None,
    ):
        """Initialize counterfactual analyzer.

        Args:
            disease_model: Disease model for simulation
            time_points: Time points for simulation (default: 0, 1, 4, 24, 168 hours)
        """
        self.disease_model = disease_model
        self.time_points = time_points or [0, 1, 4, 24, 168]  # 1 week

    def generate_counterfactual_trajectory(
        self,
        patient: PatientArchetype,
        scenario: CounterfactualScenario,
    ) -> DiseaseTrajectory:
        """Generate counterfactual disease trajectory under scenario.

        Args:
            patient: Patient archetype
            scenario: Counterfactual scenario

        Returns:
            Simulated disease trajectory
        """
        # Simulate progression under counterfactual intervention
        trajectory = self.disease_model.simulate_progression(
            patient=patient,
            time_points=self.time_points,
            intervention=(
                scenario.intervention.value
                if scenario.intervention != InterventionType.NONE
                else None
            ),
            intervention_time=scenario.intervention_time,
        )

        return trajectory

    def calculate_outcome(
        self,
        trajectory: DiseaseTrajectory,
        scenario_name: str,
    ) -> CounterfactualOutcome:
        """Calculate outcome measures from trajectory.

        Args:
            trajectory: Disease trajectory
            scenario_name: Scenario name

        Returns:
            Counterfactual outcome
        """
        # Final vertigo severity
        final_state = trajectory.symptom_states[-1]
        final_vertigo = final_state.vertigo_severity

        # Final DRAS level
        if trajectory.predicted_dras_levels:
            final_dras = trajectory.predicted_dras_levels[-1]
        else:
            final_dras = 0

        # Time to symptom resolution (vertigo < 2)
        symptom_free_time = None
        for i, state in enumerate(trajectory.symptom_states):
            if state.vertigo_severity < 2.0:
                symptom_free_time = trajectory.time_points[i]
                break

        # Total symptom burden (area under curve)
        # Trapezoidal integration
        symptom_burden = 0.0
        for i in range(len(trajectory.symptom_states) - 1):
            dt = trajectory.time_points[i + 1] - trajectory.time_points[i]
            avg_severity = (
                trajectory.symptom_states[i].vertigo_severity
                + trajectory.symptom_states[i + 1].vertigo_severity
            ) / 2.0
            symptom_burden += avg_severity * dt

        # Adverse events (check interventions)
        adverse_events = []
        if "hemorrhage_occurred" in trajectory.interventions:
            if trajectory.interventions["hemorrhage_occurred"]:
                adverse_events.append("Symptomatic hemorrhage (tPA complication)")

        if "epley_successful" in trajectory.interventions:
            if not trajectory.interventions["epley_successful"]:
                adverse_events.append("Epley maneuver unsuccessful")

        return CounterfactualOutcome(
            scenario_name=scenario_name,
            final_vertigo_severity=final_vertigo,
            final_dras_level=final_dras,
            symptom_free_time=symptom_free_time,
            total_symptom_burden=symptom_burden,
            adverse_events=adverse_events,
        )

    def estimate_treatment_effect(
        self,
        patient: PatientArchetype,
        treatment_scenario: CounterfactualScenario,
        control_scenario: CounterfactualScenario,
    ) -> TreatmentEffect:
        """Estimate individual treatment effect (ITE).

        ITE = Y(treated) - Y(control)

        Args:
            patient: Patient archetype
            treatment_scenario: Treatment scenario
            control_scenario: Control scenario

        Returns:
            Individual treatment effect
        """
        # Generate counterfactual trajectories
        traj_treatment = self.generate_counterfactual_trajectory(patient, treatment_scenario)
        traj_control = self.generate_counterfactual_trajectory(patient, control_scenario)

        # Calculate outcomes
        outcome_treatment = self.calculate_outcome(traj_treatment, treatment_scenario.scenario_name)
        outcome_control = self.calculate_outcome(traj_control, control_scenario.scenario_name)

        # Calculate treatment effects
        effect_vertigo = (
            outcome_control.final_vertigo_severity - outcome_treatment.final_vertigo_severity
        )
        effect_burden = (
            outcome_control.total_symptom_burden - outcome_treatment.total_symptom_burden
        )

        # Time to resolution effect
        if outcome_treatment.symptom_free_time and outcome_control.symptom_free_time:
            effect_time = outcome_control.symptom_free_time - outcome_treatment.symptom_free_time
        else:
            effect_time = 0.0

        # Clinical interpretation
        if effect_vertigo > 2.0:
            benefit = "substantial"
        elif effect_vertigo > 1.0:
            benefit = "moderate"
        elif effect_vertigo > 0.3:
            benefit = "small"
        else:
            benefit = "minimal"

        interpretation = (
            f"{treatment_scenario.scenario_name} vs {control_scenario.scenario_name}: "
            f"{benefit} benefit ({effect_vertigo:.1f} point vertigo reduction, "
            f"{effect_burden:.0f} hour-point symptom burden reduction)"
        )

        return TreatmentEffect(
            patient_id=patient.patient_id,
            treatment_scenario=treatment_scenario.scenario_name,
            control_scenario=control_scenario.scenario_name,
            effect_on_vertigo=effect_vertigo,
            effect_on_symptom_burden=effect_burden,
            effect_on_time_to_resolution=effect_time,
            clinical_interpretation=interpretation,
        )

    def optimal_treatment_timing(
        self,
        patient: PatientArchetype,
        intervention: InterventionType,
        time_grid: Optional[List[float]] = None,
    ) -> Dict[str, any]:
        """Find optimal intervention timing for patient.

        Args:
            patient: Patient archetype
            intervention: Intervention type
            time_grid: Time points to evaluate (default: 0, 1, 4, 24 hours)

        Returns:
            Dictionary with optimal timing and expected outcomes
        """
        time_grid = time_grid or [0, 1, 4, 24]

        # Evaluate intervention at each time point
        outcomes = []
        for t in time_grid:
            scenario = CounterfactualScenario(
                scenario_name=f"{intervention.value}_at_T{t}h",
                intervention=intervention,
                intervention_time=t,
                description=f"{intervention.value} at {t} hours",
            )

            trajectory = self.generate_counterfactual_trajectory(patient, scenario)
            outcome = self.calculate_outcome(trajectory, scenario.scenario_name)
            outcomes.append((t, outcome))

        # Find optimal timing (minimize final vertigo + symptom burden)
        best_time = time_grid[0]
        best_score = float('inf')

        for t, outcome in outcomes:
            # Composite score: final severity + 0.01 * burden
            score = outcome.final_vertigo_severity + 0.01 * outcome.total_symptom_burden
            if score < best_score:
                best_score = score
                best_time = t

        # Get outcomes at optimal time
        optimal_outcome = [outcome for t, outcome in outcomes if t == best_time][0]

        # Compare to no intervention
        control_scenario = CounterfactualScenario(
            scenario_name="No intervention",
            intervention=InterventionType.NONE,
            intervention_time=0,
            description="Natural history without intervention",
        )
        control_trajectory = self.generate_counterfactual_trajectory(patient, control_scenario)
        control_outcome = self.calculate_outcome(control_trajectory, "Control")

        benefit = control_outcome.final_vertigo_severity - optimal_outcome.final_vertigo_severity

        return {
            "optimal_time_hours": best_time,
            "intervention": intervention.value,
            "expected_final_vertigo": optimal_outcome.final_vertigo_severity,
            "expected_symptom_burden": optimal_outcome.total_symptom_burden,
            "benefit_vs_no_treatment": benefit,
            "time_to_symptom_free": optimal_outcome.symptom_free_time,
            "all_timings": [
                {
                    "time_hours": t,
                    "final_vertigo": outcome.final_vertigo_severity,
                    "symptom_burden": outcome.total_symptom_burden,
                }
                for t, outcome in outcomes
            ],
        }

    def sensitivity_analysis(
        self,
        patient: PatientArchetype,
        scenario: CounterfactualScenario,
        vary_parameter: str,
        parameter_range: List[float],
    ) -> pd.DataFrame:
        """Sensitivity analysis: vary patient parameter and see effect on outcome.

        Args:
            patient: Patient archetype
            scenario: Counterfactual scenario
            vary_parameter: Parameter to vary (e.g., "age", "stroke_risk")
            parameter_range: Range of values to test

        Returns:
            DataFrame with sensitivity results
        """
        results = []

        for value in parameter_range:
            # Create modified patient
            modified_patient = PatientArchetype(
                patient_id=patient.patient_id,
                age=value if vary_parameter == "age" else patient.age,
                gender=patient.gender,
                disease_type=patient.disease_type,
                atrial_fibrillation=patient.atrial_fibrillation,
                hypertension=patient.hypertension,
                diabetes=patient.diabetes,
                prior_stroke=patient.prior_stroke,
                health_literacy=patient.health_literacy,
                symptom_reporting_accuracy=patient.symptom_reporting_accuracy,
                disease_params=patient.disease_params.copy(),
            )

            # Simulate
            trajectory = self.generate_counterfactual_trajectory(modified_patient, scenario)
            outcome = self.calculate_outcome(trajectory, scenario.scenario_name)

            results.append(
                {
                    vary_parameter: value,
                    "final_vertigo_severity": outcome.final_vertigo_severity,
                    "final_dras_level": outcome.final_dras_level,
                    "symptom_burden": outcome.total_symptom_burden,
                }
            )

        return pd.DataFrame(results)

    def generate_clinical_recommendation(
        self,
        patient: PatientArchetype,
        intervention_options: List[InterventionType],
    ) -> str:
        """Generate clinical recommendation based on counterfactual analysis.

        Args:
            patient: Patient archetype
            intervention_options: List of intervention options to compare

        Returns:
            Clinical recommendation text
        """
        # Evaluate all intervention options
        recommendations = []

        for intervention in intervention_options:
            optimal_timing = self.optimal_treatment_timing(patient, intervention)

            recommendations.append(
                {
                    "intervention": intervention.value,
                    "optimal_time": optimal_timing["optimal_time_hours"],
                    "benefit": optimal_timing["benefit_vs_no_treatment"],
                    "time_to_resolution": optimal_timing.get("time_to_symptom_free", None),
                }
            )

        # Find best intervention
        best = max(recommendations, key=lambda x: x["benefit"])

        # Generate text recommendation
        recommendation_text = f"CLINICAL RECOMMENDATION for Patient {patient.patient_id}:\n"
        recommendation_text += f"\nDisease: {patient.disease_type.value}\n"
        recommendation_text += (
            f"Age: {patient.age}, Stroke Risk: {patient.calculate_stroke_risk():.1f}/10\n"
        )
        recommendation_text += f"\nBased on counterfactual analysis:\n"
        recommendation_text += (
            f"RECOMMENDED: {best['intervention'].upper()} at T+{best['optimal_time']:.0f}h\n"
        )
        recommendation_text += (
            f"Expected benefit: {best['benefit']:.1f} point vertigo reduction vs no treatment\n"
        )

        if best['time_to_resolution']:
            recommendation_text += (
                f"Expected symptom resolution: {best['time_to_resolution']:.0f} hours\n"
            )

        recommendation_text += f"\nAlternative options:\n"
        for rec in sorted(recommendations, key=lambda x: x["benefit"], reverse=True)[1:]:
            recommendation_text += f"  - {rec['intervention']} at T+{rec['optimal_time']:.0f}h: {rec['benefit']:.1f} point benefit\n"

        return recommendation_text

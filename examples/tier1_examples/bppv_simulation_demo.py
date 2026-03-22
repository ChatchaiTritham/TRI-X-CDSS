"""BPPV Simulation Demo: Complete End-to-End Example.

This example demonstrates:
1. Patient archetype generation
2. BPPV disease model simulation
3. Intervention comparison (Epley immediate vs delayed)
4. Visualization of disease trajectories
5. SHAP feature importance for DRAS-5 classification

Usage:
    python examples/tier1_examples/bppv_simulation_demo.py

Output:
    - Console output with simulation results
    - PNG figures in examples/output/ directory
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from datetime import datetime

# TRI-X imports
from trix_cdss.tier1_patient.digital_twin import (
    PatientArchetype,
    DiseaseType,
    Gender,
    generate_archetype,
    BPPVModel,
)
from trix_cdss.visualization import (
    plot_disease_trajectory,
    plot_multiple_trajectories,
    plot_shap_importance,
)


def main():
    """Run BPPV simulation demo."""
    print("=" * 80)
    print("TRI-X-CDSS: BPPV Simulation Demo")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # ========================================
    # Step 1: Generate Patient Archetype
    # ========================================
    print("Step 1: Generating Patient Archetype...")
    print("-" * 80)

    patient = PatientArchetype(
        patient_id="DEMO_001",
        age=65,
        gender=Gender.FEMALE,
        disease_type=DiseaseType.BPPV,
        atrial_fibrillation=False,
        hypertension=True,
        diabetes=False,
        prior_stroke=False,
        prior_bppv=True,  # Recurrent BPPV
        health_literacy=0.7,
        symptom_reporting_accuracy=0.8,
        disease_params={
            "canal_affected": "posterior",  # 85% of BPPV cases
            "severity": 8.0,  # 0-10 scale
        },
    )

    print("[OK] Patient Archetype Created:")
    print(f"   - ID: {patient.patient_id}")
    print(f"   - Age: {patient.age} years")
    print(f"   - Gender: {patient.gender.value}")
    print(f"   - Disease: {patient.disease_type.value}")
    print(f"   - Canal Affected: {patient.disease_params['canal_affected']}")
    print(f"   - Severity: {patient.disease_params['severity']}/10")
    print(f"   - Stroke Risk Score: {patient.calculate_stroke_risk():.1f}/10")
    print()

    # ========================================
    # Step 2: Initialize BPPV Model
    # ========================================
    print("Step 2: Initializing BPPV Disease Model...")
    print("-" * 80)

    model = BPPVModel(random_seed=42)

    natural_history = model.get_natural_history(patient)
    print("[OK] BPPV Natural History:")
    print(f"   - Spontaneous resolution at 1 week: {natural_history['spontaneous_resolution_1week']*100:.0f}%")
    print(f"   - Spontaneous resolution at 1 month: {natural_history['spontaneous_resolution_1month']*100:.0f}%")
    print(f"   - Epley success rate: {natural_history['epley_success_rate']*100:.0f}%")
    print(f"   - Recurrence rate at 6 months: {natural_history['recurrence_rate_6months']*100:.0f}%")
    print()

    # ========================================
    # Step 3: Simulate Disease Progression
    # ========================================
    print("Step 3: Simulating Disease Trajectories...")
    print("-" * 80)

    time_points = [0, 1, 4, 24, 168]  # 0h, 1h, 4h, 24h, 1 week

    # Scenario 1: Natural history (no intervention)
    print("[SCENARIO] Natural History (No Intervention)")
    trajectory_natural = model.simulate_progression(
        patient=patient,
        time_points=time_points,
        intervention=None,
    )
    print(f"   - Initial vertigo severity: {trajectory_natural.symptom_states[0].vertigo_severity:.1f}/10")
    print(f"   - Vertigo at 1 week: {trajectory_natural.symptom_states[-1].vertigo_severity:.1f}/10")
    print()

    # Scenario 2: Epley immediate (at presentation)
    print("[SCENARIO] Epley Maneuver Immediate (T0)")
    trajectory_epley_immediate = model.simulate_progression(
        patient=patient,
        time_points=time_points,
        intervention="epley",
        intervention_time=0,
    )
    epley_success = trajectory_epley_immediate.interventions.get("epley_successful", False)
    print(f"   - Epley successful: {epley_success}")
    print(f"   - Initial vertigo severity: {trajectory_epley_immediate.symptom_states[0].vertigo_severity:.1f}/10")
    print(f"   - Vertigo at 1 hour: {trajectory_epley_immediate.symptom_states[1].vertigo_severity:.1f}/10")
    print(f"   - Vertigo at 1 week: {trajectory_epley_immediate.symptom_states[-1].vertigo_severity:.1f}/10")
    print()

    # Scenario 3: Epley delayed (at 24 hours)
    print("[SCENARIO] Epley Maneuver Delayed (T24h)")
    trajectory_epley_delayed = model.simulate_progression(
        patient=patient,
        time_points=time_points,
        intervention="epley",
        intervention_time=24,
    )
    print(f"   - Initial vertigo severity: {trajectory_epley_delayed.symptom_states[0].vertigo_severity:.1f}/10")
    print(f"   - Vertigo at 24h (before Epley): {trajectory_epley_delayed.symptom_states[3].vertigo_severity:.1f}/10")
    print(f"   - Vertigo at 1 week (after Epley): {trajectory_epley_delayed.symptom_states[-1].vertigo_severity:.1f}/10")
    print()

    # ========================================
    # Step 4: Calculate DRAS Levels
    # ========================================
    print("Step 4: Calculating DRAS-5 Urgency Levels...")
    print("-" * 80)

    for trajectory in [trajectory_natural, trajectory_epley_immediate, trajectory_epley_delayed]:
        dras_levels = []
        for state in trajectory.symptom_states:
            dras = model.calculate_dras_level(state, patient)
            dras_levels.append(dras)
        trajectory.predicted_dras_levels = dras_levels

    print("[OK] DRAS Levels Calculated:")
    print(f"   - Natural: {trajectory_natural.predicted_dras_levels}")
    print(f"   - Epley Immediate: {trajectory_epley_immediate.predicted_dras_levels}")
    print(f"   - Epley Delayed: {trajectory_epley_delayed.predicted_dras_levels}")
    print()

    # ========================================
    # Step 5: Generate Visualizations
    # ========================================
    print("Step 5: Generating Visualizations...")
    print("-" * 80)

    # Plot 1: Single trajectory (Epley immediate)
    print("[PLOT] Creating Figure 1: BPPV Trajectory (Epley Immediate)...")
    fig1 = plot_disease_trajectory(
        trajectory_epley_immediate,
        save_path=output_dir / "fig1_bppv_trajectory_epley_immediate.png",
        show=False,
    )
    print(f"   [OK] Saved: {output_dir / 'fig1_bppv_trajectory_epley_immediate.png'}")

    # Plot 2: Comparison of all three scenarios
    print("[PLOT] Creating Figure 2: Trajectory Comparison (Natural vs Epley Immediate vs Delayed)...")
    fig2 = plot_multiple_trajectories(
        [trajectory_natural, trajectory_epley_immediate, trajectory_epley_delayed],
        save_path=output_dir / "fig2_bppv_trajectory_comparison.png",
        show=False,
    )
    print(f"   [OK] Saved: {output_dir / 'fig2_bppv_trajectory_comparison.png'}")

    # Plot 3: SHAP feature importance (simulated)
    print("[PLOT] Creating Figure 3: SHAP Feature Importance for DRAS-5...")
    feature_importance = {
        "positional_trigger": -2.0,  # Lowers urgency (suggests BPPV)
        "vertigo_severity_high": +1.5,  # Increases urgency
        "prior_bppv": -0.5,  # Lowers urgency (known condition)
        "age_65": +0.5,  # Slightly increases urgency
        "hypertension": +0.3,  # Slight risk factor
        "nystagmus_present": -1.0,  # In BPPV context, expected
        "ataxia_absent": -1.5,  # Absence lowers urgency (no central signs)
        "stroke_risk_low": -0.8,  # Low stroke risk
    }
    fig3 = plot_shap_importance(
        feature_importance,
        save_path=output_dir / "fig3_shap_feature_importance.png",
        show=False,
    )
    print(f"   [OK] Saved: {output_dir / 'fig3_shap_feature_importance.png'}")

    print()

    # ========================================
    # Step 6: Summary Statistics
    # ========================================
    print("Step 6: Summary Statistics")
    print("-" * 80)

    # Calculate treatment effect (ATE)
    vertigo_reduction_immediate = (
        trajectory_natural.symptom_states[-1].vertigo_severity
        - trajectory_epley_immediate.symptom_states[-1].vertigo_severity
    )
    vertigo_reduction_delayed = (
        trajectory_natural.symptom_states[-1].vertigo_severity
        - trajectory_epley_delayed.symptom_states[-1].vertigo_severity
    )

    print("[METRIC] Treatment Effect (ATE):")
    print(f"   - Epley Immediate vs Natural: {vertigo_reduction_immediate:.2f} points reduction at 1 week")
    print(f"   - Epley Delayed vs Natural: {vertigo_reduction_delayed:.2f} points reduction at 1 week")
    print(f"   - Benefit of Immediate vs Delayed: {vertigo_reduction_immediate - vertigo_reduction_delayed:.2f} points")
    print()

    # Counterfactual reasoning
    print("[INSIGHT] Counterfactual Reasoning:")
    print(f"   Question: \"What if Epley performed immediately vs delayed?\"")
    print(f"   Answer: Immediate Epley reduces vertigo by additional {vertigo_reduction_immediate - vertigo_reduction_delayed:.2f} points")
    print("   Clinical Impact: Faster symptom resolution -> Higher patient satisfaction -> Lower recurrence anxiety")
    print()

    # ========================================
    # Step 7: Clinical Recommendations
    # ========================================
    print("Step 7: Clinical Recommendations Based on TRI-X-CDSS")
    print("-" * 80)

    if trajectory_epley_immediate.predicted_dras_levels[0] <= 3:
        print("[OK] RECOMMENDATION: DRAS-3 (Scheduled Specialist OPD)")
        print("   - Route to ENT clinic for Epley maneuver (1-7 days)")
        print("   - Provide BPPV education and Brandt-Daroff exercises")
        print("   - Prescribe vestibular suppressant (meclizine 25mg PRN, max 3 days)")
        print("   - Discharge with return precautions")
        print("   - BENEFIT: Reduces ED crowding, appropriate care pathway")
    else:
        print("[WARN] RECOMMENDATION: Higher DRAS level requires urgent evaluation")

    print()

    # ========================================
    # Done!
    # ========================================
    print("=" * 80)
    print("[OK] BPPV Simulation Demo Completed Successfully!")
    print("=" * 80)
    print()
    print(f"[FILES] Output files saved to: {output_dir.absolute()}")
    print()
    print("Next Steps:")
    print("  1. Review generated figures in output/ directory")
    print("  2. Explore NMF pattern discovery (coming soon)")
    print("  3. Run causal inference examples (coming soon)")
    print("  4. Try multi-agent workflow simulation (Tier 2)")
    print()


if __name__ == "__main__":
    main()

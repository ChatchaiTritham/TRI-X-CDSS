"""Focused unit tests for deterministic pure functions in trix_cdss.core.

These exercise the gating / threshold / scoring logic that drives the TRI-X
triage framework. All inputs are tiny hand-made cases; no data files, no
training, no network. conftest.py puts ``src/`` on sys.path.
"""

import pytest

from trix_cdss.core.triage import (
    ESILevel,
    PatientRiskFactors,
    DizzinessSymptoms,
    VitalSigns,
    map_esi_to_dras,
)
from trix_cdss.core.titrate import (
    calculate_risk_score,
    simulate_risk_trajectory,
)
from trix_cdss.core.dras5 import (
    classify_urgency_level,
    DRAS5Features,
    DRAS5Level,
)


# ---------------------------------------------------------------------------
# triage.PatientRiskFactors.calculate_stroke_risk_score  (additive weights)
# ---------------------------------------------------------------------------

def test_stroke_risk_score_additive_weights():
    # age>=60 (2.0) + AF (3.0) + prior stroke (3.0) + HTN (1.0) = 9.0
    rf = PatientRiskFactors(
        age=72,
        atrial_fibrillation=True,
        prior_stroke_tia=True,
        hypertension=True,
    )
    assert rf.calculate_stroke_risk_score() == pytest.approx(9.0)


def test_stroke_risk_score_zero_for_low_risk_young():
    rf = PatientRiskFactors(age=30)
    assert rf.calculate_stroke_risk_score() == 0.0


def test_stroke_risk_age_band_midlife_vs_older():
    midlife = PatientRiskFactors(age=55).calculate_stroke_risk_score()  # 1.0
    older = PatientRiskFactors(age=65).calculate_stroke_risk_score()    # 2.0
    assert midlife == pytest.approx(1.0)
    assert older == pytest.approx(2.0)
    assert older > midlife


# ---------------------------------------------------------------------------
# triage.map_esi_to_dras  (gating boundaries)
# ---------------------------------------------------------------------------

def test_map_esi_level1_always_dras5():
    # ESI 1 is an immediate emergency regardless of stroke score.
    assert map_esi_to_dras(ESILevel.LEVEL_1, stroke_risk_score=0.0) == 5
    assert map_esi_to_dras(ESILevel.LEVEL_1, stroke_risk_score=99.0) == 5


def test_map_esi_level2_threshold_on_stroke_risk():
    # VERY_HIGH threshold is 4.0: at/above -> 5, below -> 4.
    assert map_esi_to_dras(ESILevel.LEVEL_2, stroke_risk_score=4.0) == 5
    assert map_esi_to_dras(ESILevel.LEVEL_2, stroke_risk_score=3.99) == 4


def test_map_esi_level5_always_dras1():
    assert map_esi_to_dras(ESILevel.LEVEL_5, stroke_risk_score=10.0) == 1


# ---------------------------------------------------------------------------
# triage dataclass helper logic
# ---------------------------------------------------------------------------

def test_vitals_abnormal_detection():
    normal = VitalSigns(
        heart_rate=72, systolic_bp=120, diastolic_bp=80,
        respiratory_rate=16, temperature=36.8, oxygen_saturation=98,
    )
    assert normal.is_abnormal() is False

    hypoxic = VitalSigns(
        heart_rate=72, systolic_bp=120, diastolic_bp=80,
        respiratory_rate=16, temperature=36.8, oxygen_saturation=85,
    )
    assert hypoxic.is_abnormal() is True


def test_symptoms_central_warning_vs_bppv():
    # Classic BPPV: positional, brief, no neuro signs -> benign, no warning.
    bppv = DizzinessSymptoms(
        symptom_description="spinning",
        duration_hours=2.0,
        episode_duration_seconds=20.0,
        positional_trigger=True,
        continuous_vertigo=False,
    )
    assert bppv.has_central_warning_signs() is False
    assert bppv.suggests_bppv() is True

    # Dysarthria is a central (stroke) warning sign.
    central = DizzinessSymptoms(
        symptom_description="vertigo",
        duration_hours=2.0,
        episode_duration_seconds=None,
        dysarthria=True,
    )
    assert central.has_central_warning_signs() is True
    assert central.suggests_bppv() is False


# ---------------------------------------------------------------------------
# titrate.calculate_risk_score  (weights + clamp to [0,10])
# ---------------------------------------------------------------------------

def test_titrate_risk_score_clamped_to_max():
    # central(4) + focal(3) + imaging(5) + not improving(1) = 13, +thrombolysis
    # bonus, all clamped to MAX_RISK_SCORE = 10.
    risk = calculate_risk_score(
        hints_central=True,
        focal_deficit=True,
        imaging_positive=True,
        symptom_improving=False,
        time_from_onset_hours=2.0,
    )
    assert risk == pytest.approx(10.0)


def test_titrate_risk_score_improving_reduces_and_floors_at_zero():
    # Nothing positive + improving (-1.0) would be negative -> floored at 0.
    risk = calculate_risk_score(
        hints_central=False,
        focal_deficit=False,
        imaging_positive=False,
        symptom_improving=True,
        time_from_onset_hours=12.0,
    )
    assert risk == pytest.approx(0.0)
    assert 0.0 <= risk <= 10.0


# ---------------------------------------------------------------------------
# titrate.simulate_risk_trajectory  (deterministic, bounded, length)
# ---------------------------------------------------------------------------

def test_simulate_trajectory_deterministic_and_bounded():
    a = simulate_risk_trajectory(5.0, "bppv", "immediate")
    b = simulate_risk_trajectory(5.0, "bppv", "immediate")
    assert a == b  # deterministic, no RNG
    assert len(a) == 4  # default num_timepoints
    assert all(0.0 <= v <= 10.0 for v in a)
    assert a[0] == 5.0  # starts at initial risk
    # Immediate BPPV treatment should be monotonically non-increasing.
    assert all(a[i + 1] <= a[i] for i in range(len(a) - 1))


# ---------------------------------------------------------------------------
# dras5.classify_urgency_level  (end-to-end gating extremes)
# ---------------------------------------------------------------------------

def test_dras5_high_risk_maps_to_level5():
    hi = DRAS5Features(
        age=72, gender="male", atrial_fibrillation=True,
        symptom_duration_hours=1.5, hints_performed=True,
        hints_central=True, stroke_probability=0.85,
    )
    c = classify_urgency_level(
        hi, esi_level=2, red_flags_critical=True, titrate_risk_score=8.5
    )
    assert c.level is DRAS5Level.LEVEL_5


def test_dras5_benign_case_low_level():
    lo = DRAS5Features(
        age=30, gender="female", symptom_duration_hours=48.0,
        hints_performed=True, hints_central=False, stroke_probability=0.05,
    )
    c = classify_urgency_level(lo, esi_level=4)
    assert isinstance(c.level, DRAS5Level)
    # Benign presentation must not be classed as an emergency.
    assert c.level.value <= 3

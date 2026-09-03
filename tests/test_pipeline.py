"""Tests that assert the estimators recover known truth from simulated races.

The suite is organised around a rule learned the hard way: a test that shares an
assumption with the code it tests proves nothing. The original synthetic race
was generated with exactly the fuel coefficient the pipeline assumes, so the
fuel correction was perfect by construction and every test passed while the real
analysis was quietly wrong. The misspecification tests below exist to keep that
from being true again.
"""

import numpy as np
import pytest

from src.clean import clean_laps
from src.config import ASSUMPTIONS, Circuit
from src.model import (
    compare_linear_quadratic,
    estimate_compound_offsets,
    estimate_pit_loss,
    fit_joint_model,
    fit_stint_slopes,
    identifiability_report,
    observed_stint_limits,
    pool_degradation,
)
from src.strategy import optimise_one_stop, stint_time, usable_compounds
from src.synthetic import (
    TRUE_DEG,
    TRUE_FUEL_EFFECT,
    TRUE_OFFSET,
    TRUE_PIT_LOSS,
    simulate_race,
)


@pytest.fixture(scope="module")
def raw():
    return simulate_race()


@pytest.fixture(scope="module")
def cleaned(raw):
    laps, _ = clean_laps(raw, ASSUMPTIONS)
    return laps


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def test_cleaning_removes_safety_car_and_pit_laps(raw, cleaned):
    assert (cleaned["TrackStatus"] == "1").all()
    assert not cleaned["IsPitInLap"].any()
    assert not cleaned["IsPitOutLap"].any()
    # Cleaning should be selective, not destructive.
    assert 0.45 < len(cleaned) / len(raw) < 0.95


def test_warmup_assumption_actually_removes_warmup_laps(raw):
    """The warm-up filter must bite, and must scale with its assumption.

    This is the test the original suite lacked. `stint_warmup_laps` was
    documented, swept in the sensitivity analysis and printed in the audit, but
    the comparison behind it was off by one and it never removed a single lap
    that the pit filter had not already taken. An assumption nothing depends on
    is worse than no assumption: it looks like rigour.
    """
    import dataclasses

    none_dropped, _ = clean_laps(raw, dataclasses.replace(ASSUMPTIONS, stint_warmup_laps=0))
    one_dropped, _ = clean_laps(raw, dataclasses.replace(ASSUMPTIONS, stint_warmup_laps=1))
    two_dropped, _ = clean_laps(raw, dataclasses.replace(ASSUMPTIONS, stint_warmup_laps=2))

    assert len(none_dropped) > len(one_dropped) > len(two_dropped)
    assert one_dropped["TyreLife"].min() > 2
    assert two_dropped["TyreLife"].min() > 3


def test_fuel_correction_removes_the_lap_number_trend(cleaned):
    """After correction, pace should not still be improving with lap number.

    Uncorrected, every car gets faster through the race as fuel burns off. If
    the correction works, the remaining correlation between lap number and
    corrected pace should be weak - what is left is degradation, which pushes
    the other way.
    """
    raw_corr = np.corrcoef(cleaned["LapNumber"], cleaned["LapTimeSeconds"])[0, 1]
    corrected_corr = np.corrcoef(cleaned["LapNumber"], cleaned["LapTimeFuelCorrected"])[0, 1]
    assert raw_corr < -0.4, "synthetic data should show a strong fuel-burn trend"
    assert corrected_corr > raw_corr


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

def test_degradation_slopes_recover_truth(cleaned):
    fits = fit_stint_slopes(cleaned)
    pooled = pool_degradation(fits, ASSUMPTIONS)
    assert not pooled.empty

    for _, row in pooled.iterrows():
        truth = TRUE_DEG[row["Compound"]]
        assert abs(row["DegSecPerLap"] - truth) < 0.02, (
            f"{row['Compound']}: estimated {row['DegSecPerLap']:.4f} vs true {truth:.4f}"
        )
        assert row["DegCILow"] <= truth <= row["DegCIHigh"], (
            f"{row['Compound']}: true value outside bootstrap CI"
        )


def test_degradation_ordering_is_recovered(cleaned):
    pooled = pool_degradation(fit_stint_slopes(cleaned), ASSUMPTIONS)
    by_compound = pooled.set_index("Compound")["DegSecPerLap"]
    assert by_compound["SOFT"] > by_compound["MEDIUM"] > by_compound["HARD"]


def test_thin_and_negative_cells_are_excluded_not_silently_used():
    """Guard the failure that produced a 65-lap soft stint recommendation.

    A single short stint that happens to fit a negative slope must never reach
    the optimiser, which cannot tell a physically impossible rate from a fast
    tyre and will build a whole strategy on it.
    """
    import pandas as pd

    fits = pd.DataFrame(
        [
            # One lonely stint with an impossible slope.
            {"Circuit": "X", "Compound": "SOFT", "NLaps": 12, "SlopeSecPerLap": -0.108,
             "SlopeCILow": -0.2, "SlopeCIHigh": 0.0},
            # A properly evidenced cell.
            *[
                {"Circuit": "X", "Compound": "HARD", "NLaps": 20, "SlopeSecPerLap": 0.05,
                 "SlopeCILow": 0.04, "SlopeCIHigh": 0.06}
                for _ in range(5)
            ],
        ]
    )
    pooled = pool_degradation(fits, ASSUMPTIONS)
    soft = pooled[pooled["Compound"] == "SOFT"].iloc[0]
    hard = pooled[pooled["Compound"] == "HARD"].iloc[0]

    assert not soft["Usable"] and soft["ExcludedReason"]
    assert hard["Usable"] and not hard["ExcludedReason"]


# ---------------------------------------------------------------------------
# Fuel, offsets and identifiability - the joint model
# ---------------------------------------------------------------------------

def test_joint_model_recovers_the_fuel_effect():
    """The coefficient the rest of the pipeline merely assumes."""
    raw = simulate_race(circuit="Synth", fuel_effect=0.050, seed=5)
    laps, _ = clean_laps(raw, ASSUMPTIONS)
    _, fuel = fit_joint_model(laps, ASSUMPTIONS)

    assert len(fuel) == 1
    estimated = float(fuel["ImpliedSecPerKg"].iloc[0])
    assert abs(estimated - 0.050) < 0.006, f"estimated {estimated:.4f} vs true 0.050"


def test_joint_model_survives_fuel_misspecification_and_the_naive_path_does_not():
    """The test that would have caught the real bug.

    With the true fuel effect at 0.050 and the pipeline assuming 0.035, the
    residual fuel trend runs through every stint in the same direction as tyre
    age. The naive route subtracts the wrong amount and books the difference as
    degradation; the joint model estimates the coefficient instead and is
    unmoved. On real 2024 data this is the difference between a hard-tyre
    degradation of 0.08 s/lap and one that comes out negative.
    """
    raw = simulate_race(circuit="Synth", fuel_effect=0.050, seed=5)
    laps, _ = clean_laps(raw, ASSUMPTIONS)

    joint, _ = fit_joint_model(laps, ASSUMPTIONS)
    naive = pool_degradation(fit_stint_slopes(laps), ASSUMPTIONS).set_index("Compound")

    joint_by_compound = joint.set_index("Compound")["JointDegSecPerLap"]
    for compound, truth in TRUE_DEG.items():
        assert abs(joint_by_compound[compound] - truth) < 0.012, (
            f"joint {compound}: {joint_by_compound[compound]:.4f} vs true {truth:.4f}"
        )

    # And the naive path is genuinely broken here, not merely noisier - if this
    # ever stops being true the misspecification is no longer being simulated.
    naive_hard = naive.loc["HARD", "DegSecPerLap"]
    assert naive_hard < TRUE_DEG["HARD"] - 0.015, (
        f"naive HARD {naive_hard:.4f} should be badly biased low under misspecification"
    )


def test_joint_offsets_are_closer_to_truth_than_naive_offsets(cleaned):
    joint, _ = fit_joint_model(cleaned, ASSUMPTIONS)
    naive = estimate_compound_offsets(fit_stint_slopes(cleaned)).set_index("Compound")["OffsetSec"]
    joint_off = joint.set_index("Compound")["OffsetSec"]

    truth_min = min(TRUE_OFFSET.values())
    joint_err = sum(abs(joint_off[c] - (TRUE_OFFSET[c] - truth_min)) for c in TRUE_OFFSET)
    naive_err = sum(abs(naive[c] - (TRUE_OFFSET[c] - truth_min)) for c in TRUE_OFFSET)
    assert joint_err < naive_err


def test_identifiability_flags_a_confounded_design():
    """A race where every soft lap is heavy and every hard lap is light.

    The estimator must not report a confident compound offset from a design that
    cannot support one. This is the guard for the real finding at Bahrain, where
    hards ran only in stints 2-3 and softs only in stint 1.
    """
    confounded = simulate_race(circuit="Confounded", strategy_mix="confounded", seed=5)
    laps, _ = clean_laps(confounded, ASSUMPTIONS)
    joint, _ = fit_joint_model(laps, ASSUMPTIONS)
    report = identifiability_report(joint, ASSUMPTIONS)

    assert not bool(report["OffsetsIdentified"].iloc[0])
    assert report["MinFuelOverlap"].iloc[0] < ASSUMPTIONS.min_fuel_overlap


def test_identifiability_passes_a_varied_design(cleaned):
    joint, _ = fit_joint_model(cleaned, ASSUMPTIONS)
    report = identifiability_report(joint, ASSUMPTIONS)
    assert bool(report["OffsetsIdentified"].iloc[0])


def test_compound_offsets_recover_relative_pace(cleaned):
    joint, _ = fit_joint_model(cleaned, ASSUMPTIONS)
    got = joint.set_index("Compound")["OffsetSec"]
    truth_min = min(TRUE_OFFSET.values())
    for compound in got.index:
        expected = TRUE_OFFSET[compound] - truth_min
        assert abs(got[compound] - expected) < 0.35, (
            f"{compound}: offset {got[compound]:.3f} vs expected {expected:.3f}"
        )


# ---------------------------------------------------------------------------
# Pit loss and linearity
# ---------------------------------------------------------------------------

def test_pit_loss_is_recovered(raw):
    loss = estimate_pit_loss(raw)
    assert not loss.empty
    estimate = float(loss["PitLossMedian"].iloc[0])
    assert abs(estimate - TRUE_PIT_LOSS) < 4.0, f"pit loss {estimate:.1f} vs true {TRUE_PIT_LOSS}"


def test_linear_model_preferred_when_truth_is_linear(cleaned):
    """The synthetic tyre degrades linearly, so the quadratic should rarely win.

    This guards against a subtle failure: if the comparison were broken and
    always favoured the more flexible model, a real non-linear finding would be
    meaningless.
    """
    comparison = compare_linear_quadratic(cleaned)
    assert not comparison.empty
    assert comparison["QuadraticWins"].mean() < 0.5


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

def test_stint_time_matches_closed_form():
    deg, offset, n = 0.05, 0.4, 20
    expected = offset * n + deg * n * (n + 1) / 2
    assert stint_time(deg, offset, n) == pytest.approx(expected)


def test_optimiser_respects_the_two_compound_rule():
    deg = {"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.02}
    offsets = {"SOFT": 0.0, "MEDIUM": 0.45, "HARD": 0.95}
    plans = optimise_one_stop(57, deg, offsets, pit_loss=21.0)
    assert (plans["Compound1"] != plans["Compound2"]).all()


def test_optimiser_refuses_negative_degradation():
    """A tyre that gets faster with age must never enter a plan."""
    deg = {"SOFT": -0.05, "MEDIUM": 0.05, "HARD": 0.02}
    offsets = {"SOFT": 0.0, "MEDIUM": 0.45, "HARD": 0.95}
    assert "SOFT" not in usable_compounds(deg, offsets)

    plans = optimise_one_stop(70, deg, offsets, pit_loss=20.0)
    assert not plans.empty
    assert not plans["Plan"].str.contains("SOFT").any()


def test_stint_length_cap_is_respected():
    """Without a cap, linear degradation licenses a stint no tyre survives."""
    deg = {"MEDIUM": 0.02, "HARD": 0.01}
    offsets = {"MEDIUM": 0.0, "HARD": 0.30}
    caps = {"MEDIUM": 40, "HARD": 45}

    plans = optimise_one_stop(70, deg, offsets, pit_loss=20.0, max_stint=caps)
    assert not plans.empty
    for _, row in plans.iterrows():
        assert row["StopLap"] <= caps[row["Compound1"]]
        assert (70 - row["StopLap"]) <= caps[row["Compound2"]]

    # The cap must actually remove options rather than decorate the call.
    uncapped = optimise_one_stop(70, deg, offsets, pit_loss=20.0)
    assert len(uncapped) > len(plans)


def test_impossible_stint_caps_yield_no_one_stop_plan():
    """When no pair of legal stints covers the distance, say so.

    Returning nothing is the correct answer here, and a more useful one than a
    plan that quietly assumes a tyre lasts twice as long as any team dared run
    it. The caller reports the circuit as unplannable rather than optimising an
    empty set.
    """
    deg = {"MEDIUM": 0.02, "HARD": 0.01}
    offsets = {"MEDIUM": 0.0, "HARD": 0.30}
    caps = {"MEDIUM": 25, "HARD": 30}  # 25 + 30 < 70

    assert optimise_one_stop(70, deg, offsets, pit_loss=20.0, max_stint=caps).empty


def test_observed_stint_limits_track_the_data(cleaned):
    limits = observed_stint_limits(cleaned, ASSUMPTIONS)
    assert not limits.empty
    for _, row in limits.iterrows():
        assert row["MaxAllowedStintLaps"] == (
            row["MaxObservedStintLaps"] + ASSUMPTIONS.max_stint_margin_laps
        )
        assert row["MedianStintLaps"] <= row["MaxObservedStintLaps"]


def test_higher_pit_loss_delays_the_stop():
    """A sanity check with an obvious right answer.

    If stopping costs more, the model must want to stop later. A strategy model
    that fails this is wrong in a way no confidence interval would reveal.
    """
    deg = {"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.02}
    offsets = {"SOFT": 0.0, "MEDIUM": 0.45, "HARD": 0.95}
    cheap = optimise_one_stop(57, deg, offsets, pit_loss=16.0).iloc[0]["StopLap"]
    dear = optimise_one_stop(57, deg, offsets, pit_loss=30.0).iloc[0]["StopLap"]
    assert dear >= cheap


def test_faster_degradation_brings_the_stop_forward():
    offsets = {"SOFT": 0.0, "MEDIUM": 0.45, "HARD": 0.95}
    gentle = optimise_one_stop(
        57, {"SOFT": 0.05, "MEDIUM": 0.03, "HARD": 0.015}, offsets, pit_loss=21.0
    ).iloc[0]["StopLap"]
    savage = optimise_one_stop(
        57, {"SOFT": 0.20, "MEDIUM": 0.12, "HARD": 0.06}, offsets, pit_loss=21.0
    ).iloc[0]["StopLap"]
    assert savage <= gentle


def test_undercut_credit_brings_stops_forward_and_adds_them():
    """An undercut makes stopping cheaper, so the model should stop more."""
    import dataclasses

    from src.strategy import optimal_summary

    deg = {"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.02}
    offsets = {"SOFT": 0.0, "MEDIUM": 0.45, "HARD": 0.95}

    none = optimal_summary("X", 57, deg, offsets, 22.0, a=ASSUMPTIONS)
    big = optimal_summary(
        "X", 57, deg, offsets, 22.0, a=dataclasses.replace(ASSUMPTIONS, undercut_gain_s=8.0)
    )
    assert big["TwoStopMinusOneStopSeconds"] < none["TwoStopMinusOneStopSeconds"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def test_circuit_config_is_unambiguous():
    """Every circuit must carry a full official event name, not a loose one.

    The study previously asked FastF1 for "Great Britain" and was handed the
    Austrian Grand Prix, which is a different circuit with a different tyre
    allocation and 71 laps instead of 52. Nothing downstream noticed. Round
    numbers cannot be hard-coded either, because they move between seasons.
    """
    from src.config import CIRCUITS, SEASONS, TARGET_SEASON

    assert all(isinstance(c, Circuit) for c in CIRCUITS)
    assert all(c.event_name.endswith("Grand Prix") for c in CIRCUITS)
    assert len({c.event_name for c in CIRCUITS}) == len(CIRCUITS)
    assert len({c.label for c in CIRCUITS}) == len(CIRCUITS)
    assert len(SEASONS) >= 2, "compound offsets need more than one season to be identified"
    assert TARGET_SEASON in SEASONS


def test_infeasible_one_stop_falls_back_to_the_two_stop():
    """When no legal pair of stints covers the race, still return a plan.

    Bahrain 2024 lands here: the longest hard stint the field ran was 29 laps and
    the longest soft 21, so under observed limits two stints cannot cover 57
    laps. The right answer is "this race cannot be one-stopped, here is the
    two-stop", not a row of NaN.
    """
    from src.strategy import optimal_summary

    deg = {"SOFT": 0.12, "HARD": 0.10}
    offsets = {"HARD": 0.0, "SOFT": 0.20}
    caps = {"HARD": 32, "SOFT": 24}  # 32 + 24 = 56 < 57

    summary = optimal_summary("Bahrain-like", 57, deg, offsets, 24.0, max_stint=caps)

    assert summary["OneStopFeasible"] is False
    assert summary.get("BestOneStopPlan") is None
    assert summary["RecommendedStops"] == 2
    assert summary["BestTwoStopPlan"]
    assert "OneStopInfeasibleReason" in summary


def test_uncapped_answer_is_always_reported_alongside_the_capped_one():
    """The cap encodes what teams did, so its effect must stay visible."""
    from src.strategy import optimal_summary

    deg = {"SOFT": 0.12, "HARD": 0.10}
    offsets = {"HARD": 0.0, "SOFT": 0.20}
    caps = {"HARD": 32, "SOFT": 24}

    summary = optimal_summary("X", 57, deg, offsets, 24.0, max_stint=caps)
    assert summary["BestOneStopPlanUncapped"] is not None
    assert np.isfinite(summary["BestOneStopLapUncapped"])


def test_late_stint_penalty_detects_a_cliff_and_ignores_a_straight_line():
    """The diagnostic must fire on accelerating wear and stay quiet otherwise.

    A one-sided detector is useless here: the whole point is to distinguish a
    tyre that falls off from one that degrades steadily, and a test that only
    checks the cliff case would pass just as happily on a broken function that
    always reports acceleration.
    """
    import pandas as pd

    from src.model import late_stint_penalty

    def stint(curve, driver):
        n = len(curve)
        return pd.DataFrame(
            {
                "Season": 2024, "Circuit": "X", "Driver": driver, "Stint": 1.0,
                "Compound": "HARD", "TyreLife": np.arange(1.0, n + 1),
                "LapTimeFuelCorrected": curve,
            }
        )

    ages = np.arange(1.0, 25.0)
    linear = pd.concat([stint(90 + 0.05 * ages, f"L{i}") for i in range(4)])
    cliff = pd.concat(
        [stint(90 + 0.05 * ages + 0.02 * np.maximum(0, ages - 16) ** 2, f"C{i}") for i in range(4)]
    )

    linear_extra = late_stint_penalty(linear)["LateStintExtraSeconds"]
    cliff_extra = late_stint_penalty(cliff)["LateStintExtraSeconds"]

    assert not linear_extra.empty and not cliff_extra.empty
    assert abs(linear_extra.median()) < 0.02, "a straight line must not look like a cliff"
    assert cliff_extra.median() > 0.2, "accelerating wear must be detected"
    assert cliff_extra.median() > linear_extra.median()


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------

def _curved_laps(curve, n_stints=12, n_laps=22):
    import pandas as pd

    frames = []
    rng = np.random.default_rng(3)
    for i in range(n_stints):
        age = np.arange(1.0, n_laps + 1)
        y = 90 + 0.05 * age + curve * age**2 + rng.normal(0, 0.15, len(age))
        frames.append(
            pd.DataFrame(
                {
                    "Season": 2024, "Circuit": "X", "Driver": f"D{i}", "Stint": 1.0,
                    "Compound": "HARD", "TyreLife": age, "LapTimeFuelCorrected": y,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_curvature_is_recovered_when_it_is_real():
    from src.model import fit_stint_curvature, pool_curvature

    pooled = pool_curvature(fit_stint_curvature(_curved_laps(0.004), ASSUMPTIONS), ASSUMPTIONS)
    row = pooled.iloc[0]
    assert row["Usable"], row["ExcludedReason"]
    assert abs(row["CurvatureSecPerLap2"] - 0.004) < 0.0015
    assert row["AppliedCurvature"] == row["CurvatureSecPerLap2"]


def test_curvature_falls_back_to_zero_on_a_straight_line():
    """The asymmetry that matters: inventing a cliff is worse than missing one.

    A spurious positive curvature stops the car far too early. Falling back to
    linear only returns the model to a bias it already has and already reports.
    """
    from src.model import fit_stint_curvature, pool_curvature

    pooled = pool_curvature(fit_stint_curvature(_curved_laps(0.0), ASSUMPTIONS), ASSUMPTIONS)
    row = pooled.iloc[0]
    assert not row["Usable"]
    assert row["AppliedCurvature"] == 0.0


def test_curvature_shortens_the_longest_stint():
    """Pricing the cliff must make long stints less attractive.

    Note what this does *not* claim. Curvature does not simply move the stop
    earlier: the cost of a stint grows with the square of its length, so equal
    curvature on both compounds penalises an unbalanced split hardest and pushes
    the two stints toward each other. Here that moves the stop from 27 to 30 -
    later, not earlier, while still shortening the longest stint from 33 laps to
    30. The invariant worth testing is the longest stint, not the stop lap.
    """
    deg = {"MEDIUM": 0.05, "HARD": 0.03}
    offsets = {"MEDIUM": 0.0, "HARD": 0.35}
    total = 60

    def longest(curve):
        stop = int(
            optimise_one_stop(total, deg, offsets, pit_loss=22.0, curvature=curve)
            .iloc[0]["StopLap"]
        )
        return max(stop, total - stop)

    assert longest({"MEDIUM": 0.004, "HARD": 0.004}) < longest(None)


def test_curvature_makes_stopping_twice_relatively_better():
    """A one-stop runs the longest stints, so the cliff costs it the most."""
    from src.strategy import optimal_summary

    deg = {"MEDIUM": 0.05, "HARD": 0.03}
    offsets = {"MEDIUM": 0.0, "HARD": 0.35}

    flat = optimal_summary("X", 60, deg, offsets, 22.0)
    curved = optimal_summary("X", 60, deg, offsets, 22.0, curvature={"MEDIUM": 0.004, "HARD": 0.004})
    assert (
        curved["TwoStopMinusOneStopSeconds"] < flat["TwoStopMinusOneStopSeconds"]
    )


def test_stint_time_curvature_matches_the_closed_form():
    n, deg, curve, offset = 20, 0.05, 0.003, 0.4
    ages = np.arange(1, n + 1)
    expected = offset * n + float(np.sum(deg * ages + curve * ages**2))
    assert stint_time(deg, offset, n, curve) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Multi-season handling
# ---------------------------------------------------------------------------

def test_wet_races_are_detected_from_the_tyres_fitted():
    import pandas as pd

    from src.model import wet_race_report

    dry = pd.DataFrame({"Season": 2024, "Circuit": "Dry", "Compound": ["HARD"] * 100})
    wet = pd.DataFrame(
        {"Season": 2024, "Circuit": "Wet", "Compound": ["HARD"] * 70 + ["INTERMEDIATE"] * 30}
    )
    report = wet_race_report(pd.concat([dry, wet]), ASSUMPTIONS).set_index("Circuit")

    assert not report.loc["Dry", "WetAffected"]
    assert report.loc["Wet", "WetAffected"]


def test_season_heterogeneity_flags_seasons_that_disagree():
    """Pooling three seasons assumes 'SOFT' means the same rubber each year."""
    import pandas as pd

    from src.model import season_heterogeneity

    rows = []
    for season, slope in [(2022, 0.02), (2023, 0.02), (2024, 0.20)]:
        for i in range(6):
            rows.append(
                {
                    "Season": season, "Circuit": "X", "Compound": "SOFT", "Driver": f"D{i}",
                    "NLaps": 20, "SlopeSecPerLap": slope,
                    "SlopeCILow": slope - 0.004, "SlopeCIHigh": slope + 0.004,
                }
            )
    report = season_heterogeneity(pd.DataFrame(rows), ASSUMPTIONS)
    assert bool(report.iloc[0]["SeasonsDisagree"])
    assert report.iloc[0]["SpreadSecPerLap"] > 0.15


def test_joint_model_separates_drivers_by_season():
    """The same name in 2022 and 2024 is a different car, not the same baseline."""
    from src.model import _build_design

    import pandas as pd

    sub = pd.DataFrame(
        {
            "Season": [2022, 2022, 2024, 2024],
            "Driver": ["VER", "VER", "VER", "VER"],
            "Compound": ["HARD", "SOFT", "HARD", "SOFT"],
            "TyreLife": [5.0, 6.0, 7.0, 8.0],
            "LapNumber": [5.0, 6.0, 7.0, 8.0],
            "TotalLaps": [57.0] * 4,
        }
    )
    _, names = _build_design(sub, ["HARD", "SOFT"], "HARD")
    driver_terms = [n for n in names if n.startswith("driver[")]
    assert len(driver_terms) == 2, "one driver across two seasons must get two baselines"


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def test_backtest_never_fits_on_the_season_it_predicts():
    """The leakage guard. Without it, out-of-sample validation is theatre."""
    import pandas as pd

    from src import backtest as bt
    from src.synthetic import simulate_race

    raw = pd.concat(
        [
            simulate_race(circuit="Park", total_laps=57, seed=s, season=year)
            for year, s in [(2022, 1), (2023, 2), (2024, 3)]
        ],
        ignore_index=True,
    )

    seen = []
    original = bt.fit_on

    def spy(train_raw, a=ASSUMPTIONS):
        seen.append(sorted(train_raw["Season"].unique().tolist()))
        return original(train_raw, a)

    bt.fit_on = spy
    try:
        bt.rolling_backtest(raw, [2022, 2023, 2024], ASSUMPTIONS)
    finally:
        bt.fit_on = original

    assert seen, "backtest never fitted anything"
    assert [2022] in seen and [2022, 2023] in seen
    for train_seasons in seen:
        assert 2024 not in train_seasons or train_seasons == [2022, 2023, 2024]
    # The season being predicted must never appear in its own training set.
    assert all(2023 not in s for s in seen if s == [2022])


def test_backtest_reports_a_baseline_it_can_be_judged_against():
    import pandas as pd

    from src.backtest import summarise_backtest

    results = pd.DataFrame(
        [
            {"TestSeason": 2024, "TrainSeasons": "2022,2023", "StopLapError": 5.0,
             "LinearOnlyError": 7.0, "BaselineError": 9.0, "PersistenceError": 3.0,
             "StopCountCorrect": True, "WetAffected": False},
            {"TestSeason": 2024, "TrainSeasons": "2022,2023", "StopLapError": -3.0,
             "LinearOnlyError": -5.0, "BaselineError": 11.0, "PersistenceError": -1.0,
             "StopCountCorrect": False, "WetAffected": False},
        ]
    )
    summary = summarise_backtest(results)
    row = summary.iloc[0]
    assert row["ModelMAELaps"] == pytest.approx(4.0)
    assert row["PersistenceMAELaps"] == pytest.approx(2.0)
    assert row["StopCountAccuracy"] == pytest.approx(0.5)


def test_pipeline_runs_on_a_single_season():
    """Multi-season machinery must degrade gracefully, not crash.

    Season heterogeneity has nothing to compare when only one season is present,
    and `run.py demo` is exactly that case. An empty frame is the right answer;
    an exception in the middle of the offline entry point is not.
    """
    import pandas as pd

    from src.clean import clean_laps
    from src.model import (
        fit_stint_curvature,
        fit_stint_slopes,
        pool_curvature,
        season_heterogeneity,
        wet_race_report,
    )
    from src.synthetic import simulate_race

    raw = simulate_race(circuit="Solo", season=2024, seed=4)
    laps, _ = clean_laps(raw, ASSUMPTIONS)
    fits = fit_stint_slopes(laps)

    assert season_heterogeneity(fits, ASSUMPTIONS).empty
    assert not wet_race_report(raw, ASSUMPTIONS).empty
    # Curvature on an empty input must also return a frame, not raise.
    assert pool_curvature(fit_stint_curvature(laps.head(0), ASSUMPTIONS), ASSUMPTIONS).empty
    assert isinstance(pool_curvature(pd.DataFrame(), ASSUMPTIONS), pd.DataFrame)

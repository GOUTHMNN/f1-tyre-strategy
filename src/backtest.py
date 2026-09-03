"""Out-of-sample validation: fit on past seasons, predict a season never seen.

Every other check in this project is either in-sample or synthetic. Recovering a
degradation rate that was planted by hand proves the estimator is not broken; it
does not prove the model says anything useful about a race. The only test that
does is to hide a season, build the whole model without it, and ask where the
car should stop - then look at what the teams actually did.

The prediction is deliberately made harder than it needs to be. Nothing from the
held-out season reaches the model: not its degradation, not its compound
offsets, not its curvature. Only two facts about the race itself are supplied,
because they are fixtures rather than outcomes - the scheduled lap count, and the
pit-lane loss, which is a property of the circuit's geometry and is measured from
in- and out-laps rather than from anybody's strategy.

Two baselines are reported alongside, because "eight laps out" means nothing
without knowing what a naive answer scores. Half distance is the straw man. The
one that matters is persistence - stop where this circuit was stopped at last
time - because it is free, obvious, and what any engineer would reach for before
opening a model. A model that cannot beat persistence has not earned its
complexity, and this one does not.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .clean import clean_laps
from .config import ASSUMPTIONS, Assumptions
from .model import (
    estimate_pit_loss,
    fit_joint_model,
    fit_stint_curvature,
    fit_stint_slopes,
    observed_stint_limits,
    pool_curvature,
    pool_degradation,
    wet_race_report,
)
from .strategy import actual_stop_laps, optimal_summary

log = logging.getLogger(__name__)


def fit_on(raw: pd.DataFrame, a: Assumptions = ASSUMPTIONS) -> dict:
    """Build every model input from one set of races."""
    laps, _ = clean_laps(raw, a)
    stint_fits = fit_stint_slopes(laps)
    pooled = pool_degradation(stint_fits, a)
    joint, _ = fit_joint_model(laps, a)
    curvature = pool_curvature(fit_stint_curvature(laps, a), a)
    limits = observed_stint_limits(laps, a)
    return {
        "degradation": pooled[pooled["Usable"]] if not pooled.empty else pd.DataFrame(),
        "offsets": joint,
        "curvature": curvature,
        "limits": limits,
    }


def predict_season(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    a: Assumptions = ASSUMPTIONS,
) -> pd.DataFrame:
    """Fit on `train_raw`, recommend stops for the circuits in `test_raw`.

    Pit loss and scheduled distance come from the test race because they are
    known before the lights go out. Everything about the tyres comes from the
    training seasons alone.
    """
    fitted = fit_on(train_raw, a)

    # The baseline that actually deserves beating: what happened here last time.
    # Half distance is a straw man - a rule nobody uses. "Stop when this circuit
    # was stopped at in previous seasons" is free, obvious, and is what any
    # engineer would reach for before opening a model.
    train_actual = actual_stop_laps(train_raw)
    persistence = (
        train_actual.groupby("Circuit")["MedianFirstStopLap"].median().to_dict()
        if not train_actual.empty
        else {}
    )
    deg_all, off_all = fitted["degradation"], fitted["offsets"]
    curve_all, limits_all = fitted["curvature"], fitted["limits"]

    pit_loss_df = estimate_pit_loss(test_raw)
    pit_loss = (
        pit_loss_df.set_index("Circuit")["PitLossMedian"].to_dict()
        if not pit_loss_df.empty
        else {}
    )
    total_laps = test_raw.groupby("Circuit")["TotalLaps"].max().to_dict()
    wet = wet_race_report(test_raw, a).set_index("Circuit")["WetAffected"].to_dict()
    actual = actual_stop_laps(test_raw)
    actual_stop = (
        actual.set_index("Circuit")["MedianFirstStopLap"].to_dict() if not actual.empty else {}
    )
    actual_count = (
        actual.set_index("Circuit")["MedianStopsPerDriver"].to_dict() if not actual.empty else {}
    )

    rows = []
    for circuit in sorted(test_raw["Circuit"].unique()):
        if deg_all.empty or circuit not in set(deg_all["Circuit"]):
            continue
        deg = deg_all[deg_all["Circuit"] == circuit].set_index("Compound")["DegSecPerLap"].to_dict()
        offs = (
            off_all[off_all["Circuit"] == circuit].set_index("Compound")["OffsetSec"].to_dict()
            if not off_all.empty
            else {}
        )
        shared = {c: deg[c] for c in deg if c in offs}
        if len(shared) < 2 or circuit not in pit_loss or circuit not in total_laps:
            continue

        caps = (
            limits_all[limits_all["Circuit"] == circuit]
            .set_index("Compound")["MaxAllowedStintLaps"]
            .to_dict()
            if not limits_all.empty
            else {}
        )
        curve = (
            curve_all[curve_all["Circuit"] == circuit]
            .set_index("Compound")["AppliedCurvature"]
            .to_dict()
            if not curve_all.empty
            else {}
        )

        laps_total = int(total_laps[circuit])
        summary = optimal_summary(
            circuit, laps_total, shared, offs, float(pit_loss[circuit]),
            max_stint=caps, curvature=curve, a=a,
        )

        real_lap = actual_stop.get(circuit, np.nan)
        real_count = actual_count.get(circuit, np.nan)

        # Compare like with like.
        #
        # The target is the field's median *first* stop. Scoring a one-stop
        # recommendation against that is only fair when the model is actually
        # recommending one stop: where it recommends two, its first stop is the
        # comparable decision, and a field that two-stopped naturally makes its
        # first stop far earlier than any one-stop lap. An earlier version always
        # compared the one-stop lap and so charged the model roughly ten laps of
        # error for a strategy it had not recommended.
        def first_stop(stops_key, one_key, two_key):
            """The first stop of whichever strategy was recommended."""
            if summary.get(stops_key) == 2 and summary.get(two_key) is not None:
                return float(summary[two_key])
            return summary.get(one_key, np.nan)

        with_curvature = first_stop(
            "RecommendedStops", "BestOneStopLap", "BestTwoStopLap1"
        )
        without_curvature = first_stop(
            "RecommendedStopsLinearOnly",
            "BestOneStopLapLinearOnly",
            "BestTwoStopLap1LinearOnly",
        )

        # Both are always computed and reported. Which one counts as *the*
        # prediction is a configuration decision, and one the backtest itself
        # settled - see `Assumptions.apply_curvature`.
        predicted = with_curvature if a.apply_curvature else without_curvature
        linear_only = without_curvature

        rows.append(
            {
                "Circuit": circuit,
                "TotalLaps": laps_total,
                "PredictedStopLap": predicted,
                "PredictedStopLapLinearOnly": linear_only,
                "PredictedStopLapWithCurvature": with_curvature,
                "HalfDistanceBaseline": laps_total / 2.0,
                "PersistenceBaseline": persistence.get(circuit, np.nan),
                "ActualStopLap": real_lap,
                "PredictedStops": summary.get("RecommendedStops", np.nan),
                "ActualStops": real_count,
                "StopLapError": (predicted - real_lap) if predicted == predicted else np.nan,
                "LinearOnlyError": (
                    (linear_only - real_lap) if linear_only == linear_only else np.nan
                ),
                "WithCurvatureError": (
                    (with_curvature - real_lap) if with_curvature == with_curvature else np.nan
                ),
                "BaselineError": laps_total / 2.0 - real_lap,
                "PersistenceError": (
                    persistence.get(circuit, np.nan) - real_lap
                    if circuit in persistence
                    else np.nan
                ),
                "StopCountCorrect": (
                    bool(summary.get("RecommendedStops") == round(real_count))
                    if real_count == real_count and summary.get("RecommendedStops") is not None
                    else False
                ),
                "OneStopFeasible": summary.get("OneStopFeasible"),
                "WetAffected": bool(wet.get(circuit, False)),
            }
        )

    return pd.DataFrame(rows)


def rolling_backtest(
    raw: pd.DataFrame, seasons: list[int], a: Assumptions = ASSUMPTIONS
) -> pd.DataFrame:
    """Predict each season from every season before it.

    A single train/test split can be lucky. Walking forward through the seasons -
    2022 predicts 2023, then 2022-23 predict 2024 - gives more than one honest
    trial and shows whether an extra season of history actually helps.
    """
    seasons = sorted(seasons)
    frames = []
    for i in range(1, len(seasons)):
        train_seasons, test_season = seasons[:i], seasons[i]
        train = raw[raw["Season"].isin(train_seasons)]
        test = raw[raw["Season"] == test_season]
        if train.empty or test.empty:
            continue

        result = predict_season(train, test, a)
        if result.empty:
            continue
        result.insert(0, "TestSeason", test_season)
        result.insert(1, "TrainSeasons", ",".join(str(s) for s in train_seasons))
        frames.append(result)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Score the held-out predictions against the naive baseline.

    Mean absolute error on the stop lap, the share of stop counts called
    correctly, and the same error for the straight-line model and for stopping
    at half distance. If the model cannot beat half distance it is an expensive
    way to do arithmetic.
    """
    if results.empty:
        return pd.DataFrame()

    rows = []
    for (season, dry_only), group in [
        ((s, False), g) for s, g in results.groupby("TestSeason")
    ] + [
        ((s, True), g[~g["WetAffected"]])
        for s, g in results.groupby("TestSeason")
        if "WetAffected" in results.columns and g["WetAffected"].any()
    ]:
        if group.empty:
            continue
        scored = group.dropna(subset=["StopLapError"])
        rows.append(
            {
                "TestSeason": season,
                "DryRacesOnly": dry_only,
                "TrainSeasons": group["TrainSeasons"].iloc[0],
                "NCircuits": int(len(group)),
                "NWithOneStop": int(len(scored)),
                "ModelMAELaps": float(scored["StopLapError"].abs().mean()) if len(scored) else np.nan,
                "LinearOnlyMAELaps": (
                    float(scored["LinearOnlyError"].abs().mean()) if len(scored) else np.nan
                ),
                "WithCurvatureMAELaps": (
                    float(scored["WithCurvatureError"].abs().mean())
                    if len(scored) and "WithCurvatureError" in scored
                    else np.nan
                ),
                "BaselineMAELaps": (
                    float(scored["BaselineError"].abs().mean()) if len(scored) else np.nan
                ),
                "PersistenceMAELaps": (
                    float(scored["PersistenceError"].abs().mean())
                    if len(scored) and scored["PersistenceError"].notna().any()
                    else np.nan
                ),
                "MeanSignedErrorLaps": (
                    float(scored["StopLapError"].mean()) if len(scored) else np.nan
                ),
                "StopCountAccuracy": float(group["StopCountCorrect"].mean()),
            }
        )
    return pd.DataFrame(rows)

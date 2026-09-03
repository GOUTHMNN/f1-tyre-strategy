"""From degradation rates to a decision: when should the car stop?

This is the part that makes the project an analysis rather than a description.
Everything upstream measures how fast the tyre falls away; this module turns
those measurements into a recommended stop lap and then checks that
recommendation against what the teams actually did.

The model is a tyre-time model and nothing more. It has no view of traffic, the
undercut, safety-car probability or tyre allocation. That is a deliberate
limit rather than an oversight - each of those needs data this study does not
have - but it means the gap against reality in `compare_to_actual` is a
measurement of what the missing physics is worth, and should be read that way.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .config import ASSUMPTIONS, Assumptions


def stint_time(
    deg_per_lap: float, offset: float, n_laps: int, curvature: float = 0.0
) -> float:
    """Total time lost by one stint, relative to an arbitrary base pace.

    Lap k of a stint runs on a tyre of age k, so with a loss per lap of
    `deg + curvature * k` the stint totals

        offset * n  +  deg * n(n+1)/2  +  curvature * n(n+1)(2n+1)/6

    using the closed forms for the sum of the first n integers and their
    squares. With `curvature = 0` this is exactly the linear model.

    The curvature term is why the third term is here at all. Pricing a stint
    linearly says a tyre's twentieth lap costs the same as its second, which the
    study's own late-stint measurements refute at five of six circuits - and
    since under-charging long stints is precisely what makes an optimiser stop
    too late, the missing term was a candidate explanation for the model's
    largest disagreement with reality.

    Base pace is excluded because it is identical for every strategy and cancels
    in any comparison. Fuel is absent for the same reason: its cost depends on
    the lap number, not on the strategy, so it is the same whenever the car
    stops. That is exactly why the slopes had to be fuel-corrected first.
    """
    if n_laps <= 0:
        return 0.0
    n = float(n_laps)
    total = offset * n + deg_per_lap * n * (n + 1) / 2.0
    if curvature:
        total += curvature * n * (n + 1) * (2 * n + 1) / 6.0
    return total


def evaluate_plan(
    stint_lengths: tuple[int, ...],
    compounds: tuple[str, ...],
    deg: dict[str, float],
    offsets: dict[str, float],
    pit_loss: float,
    undercut_gain_s: float = 0.0,
    curvature: dict[str, float] | None = None,
) -> float:
    """Total relative race time for a full strategy.

    `undercut_gain_s` credits each stop with the track position a team expects
    to take from a rival by pitting first. It is zero by default: this model
    cannot see the cars around it, so any value is an input to be argued about
    rather than a result. It exists so the sensitivity sweep can answer a
    specific question - how big would the undercut have to be to explain why
    real teams stop so much earlier than the tyre arithmetic alone suggests?
    """
    curvature = curvature or {}
    total = sum(
        stint_time(deg[c], offsets[c], n, curvature.get(c, 0.0))
        for c, n in zip(compounds, stint_lengths)
    )
    n_stops = len(stint_lengths) - 1
    return total + (pit_loss - undercut_gain_s) * n_stops


def usable_compounds(
    deg: dict[str, float], offsets: dict[str, float]
) -> list[str]:
    """Compounds with a finite, physically possible degradation rate and an offset.

    A negative slope means the fit found a tyre that gets faster with age. That
    is a fitting artefact, but the optimiser cannot tell: fed a negative rate it
    will extend that stint as far as the rules allow to harvest the free time,
    and return a confident recommendation built entirely on noise.
    """
    return [
        c
        for c in deg
        if c in offsets and np.isfinite(deg[c]) and np.isfinite(offsets[c]) and deg[c] >= 0
    ]


def _limit(max_stint: dict[str, int] | None, compound: str, total_laps: int) -> int:
    if not max_stint:
        return total_laps
    return int(max_stint.get(compound, total_laps))


def optimise_one_stop(
    total_laps: int,
    deg: dict[str, float],
    offsets: dict[str, float],
    pit_loss: float,
    min_stint: int = 5,
    max_stint: dict[str, int] | None = None,
    undercut_gain_s: float = 0.0,
    curvature: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Search every one-stop plan: which two compounds, and which stop lap.

    The dry-race regulations require at least two different compounds, so plans
    are restricted to genuine compound changes. `max_stint` caps each compound
    at the longest stint the field actually completed on it (plus a margin);
    without that cap, linear degradation extrapolates indefinitely and the
    search happily returns stints far beyond anything a tyre survives.
    """
    available = usable_compounds(deg, offsets)
    rows = []
    for c1, c2 in itertools.permutations(available, 2):
        cap1 = _limit(max_stint, c1, total_laps)
        cap2 = _limit(max_stint, c2, total_laps)
        for stop_lap in range(min_stint, total_laps - min_stint + 1):
            n1, n2 = stop_lap, total_laps - stop_lap
            if n1 > cap1 or n2 > cap2:
                continue
            rows.append(
                {
                    "Plan": f"{c1}->{c2}",
                    "Compound1": c1,
                    "Compound2": c2,
                    "StopLap": stop_lap,
                    "RelativeRaceTime": evaluate_plan(
                        (n1, n2), (c1, c2), deg, offsets, pit_loss, undercut_gain_s, curvature
                    ),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["DeltaToBest"] = df["RelativeRaceTime"] - df["RelativeRaceTime"].min()
    return df.sort_values("RelativeRaceTime").reset_index(drop=True)


def optimise_two_stop(
    total_laps: int,
    deg: dict[str, float],
    offsets: dict[str, float],
    pit_loss: float,
    min_stint: int = 5,
    step: int = 1,
    max_stint: dict[str, int] | None = None,
    undercut_gain_s: float = 0.0,
    curvature: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Search two-stop plans, so the one-stop recommendation has competition.

    A one-stop that is only optimal because no alternative was considered is not
    a finding. Comparing the best one-stop against the best two-stop is what
    turns the degradation numbers into an actual strategic claim.
    """
    available = usable_compounds(deg, offsets)
    rows = []
    for combo in itertools.product(available, repeat=3):
        if len(set(combo)) < 2:  # two-compound rule
            continue
        caps = [_limit(max_stint, c, total_laps) for c in combo]
        for stop1 in range(min_stint, total_laps - 2 * min_stint + 1, step):
            for stop2 in range(stop1 + min_stint, total_laps - min_stint + 1, step):
                lengths = (stop1, stop2 - stop1, total_laps - stop2)
                if any(n > cap for n, cap in zip(lengths, caps)):
                    continue
                rows.append(
                    {
                        "Plan": "->".join(combo),
                        "StopLap1": stop1,
                        "StopLap2": stop2,
                        "RelativeRaceTime": evaluate_plan(
                            lengths, combo, deg, offsets, pit_loss, undercut_gain_s, curvature
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("RelativeRaceTime").reset_index(drop=True)


def optimal_summary(
    circuit: str,
    total_laps: int,
    deg: dict[str, float],
    offsets: dict[str, float],
    pit_loss: float,
    max_stint: dict[str, int] | None = None,
    curvature: dict[str, float] | None = None,
    a: Assumptions = ASSUMPTIONS,
) -> dict:
    """Best one-stop, best two-stop, and the margin between them.

    Reported twice over: once under the observed stint-length caps, and once
    without them. The caps stop the optimiser extrapolating a straight line
    across a stint no tyre survives, but they encode revealed preference rather
    than physics, and the reasoning is partly circular - if the field two-stopped,
    no long stint was observed, so a long stint looks impossible, so the model
    must two-stop as well. Publishing the uncapped answer alongside is what keeps
    that honest: where the two agree the cap is doing no work, and where they
    disagree the reader can see exactly how much of the recommendation rests on
    it.
    """
    kwargs = {
        "max_stint": max_stint,
        "undercut_gain_s": a.undercut_gain_s,
        "curvature": curvature,
    }
    one = optimise_one_stop(total_laps, deg, offsets, pit_loss, **kwargs)
    two = optimise_two_stop(total_laps, deg, offsets, pit_loss, **kwargs)

    uncapped = optimise_one_stop(
        total_laps, deg, offsets, pit_loss,
        undercut_gain_s=a.undercut_gain_s, curvature=curvature,
    )

    # What the straight-line model would have said, kept beside the curved
    # answer so the effect of pricing the cliff is visible rather than asserted.
    # Both plan shapes are re-searched without curvature, because comparing a
    # curved two-stop against a linear *one*-stop would measure the plan shape
    # rather than the curvature.
    linear_only = optimise_one_stop(
        total_laps, deg, offsets, pit_loss,
        max_stint=max_stint, undercut_gain_s=a.undercut_gain_s,
    )
    linear_two = optimise_two_stop(
        total_laps, deg, offsets, pit_loss,
        max_stint=max_stint, undercut_gain_s=a.undercut_gain_s,
    )

    result = {
        "Circuit": circuit,
        "TotalLaps": total_laps,
        "PitLossSeconds": pit_loss,
        "OneStopFeasible": not one.empty,
        "BestOneStopPlanUncapped": uncapped.iloc[0]["Plan"] if not uncapped.empty else None,
        "BestOneStopLapUncapped": int(uncapped.iloc[0]["StopLap"]) if not uncapped.empty else np.nan,
        "BestOneStopLapLinearOnly": (
            int(linear_only.iloc[0]["StopLap"]) if not linear_only.empty else np.nan
        ),
        "BestTwoStopLap1LinearOnly": (
            int(linear_two.iloc[0]["StopLap1"]) if not linear_two.empty else np.nan
        ),
        "RecommendedStopsLinearOnly": (
            2
            if linear_only.empty
            or (
                not linear_two.empty
                and linear_two.iloc[0]["RelativeRaceTime"] <= linear_only.iloc[0]["RelativeRaceTime"]
            )
            else 1
        ),
    }

    if one.empty and two.empty:
        result["Error"] = "no viable plans"
        return result

    best_two = two.iloc[0] if not two.empty else None

    if one.empty:
        # No legal pair of stints covers the distance. That is a finding, not a
        # failure: on this circuit's own tyre evidence the race cannot be done on
        # one stop, and the two-stop below is the recommendation.
        result.update(
            {
                "BestOneStopPlan": None,
                "BestOneStopLap": np.nan,
                "OneStopWindowLow": np.nan,
                "OneStopWindowHigh": np.nan,
                "OneStopInfeasibleReason": (
                    "no two compounds can cover "
                    f"{total_laps} laps within their observed stint limits"
                ),
            }
        )
    else:
        best_one = one.iloc[0]
        # How wide is the window in which the stop lap barely matters? A team can
        # act on a 12-lap window; a 2-lap window means the call is knife-edge and
        # the model's precision is being oversold.
        same_plan = one[one["Plan"] == best_one["Plan"]]
        within_one_sec = same_plan[same_plan["DeltaToBest"] <= 1.0]["StopLap"]
        result.update(
            {
                "BestOneStopPlan": best_one["Plan"],
                "BestOneStopLap": int(best_one["StopLap"]),
                "OneStopWindowLow": int(within_one_sec.min()) if len(within_one_sec) else np.nan,
                "OneStopWindowHigh": int(within_one_sec.max()) if len(within_one_sec) else np.nan,
            }
        )

    if best_two is not None:
        margin = (
            float(best_two["RelativeRaceTime"] - one.iloc[0]["RelativeRaceTime"])
            if not one.empty
            else np.nan
        )
        result.update(
            {
                "BestTwoStopPlan": best_two["Plan"],
                # Split into two scalar columns: a tuple round-trips through CSV
                # as the string "(21, 42)" and stops being a number.
                "BestTwoStopLap1": int(best_two["StopLap1"]),
                "BestTwoStopLap2": int(best_two["StopLap2"]),
                "TwoStopMinusOneStopSeconds": margin,
                "RecommendedStops": 2 if (one.empty or margin <= 0) else 1,
            }
        )
    elif not one.empty:
        result["RecommendedStops"] = 1

    return result


def actual_stop_laps(raw_laps: pd.DataFrame) -> pd.DataFrame:
    """What the teams actually did: green-flag stop laps per circuit.

    Safety-car stops are excluded because they are a response to an event, not a
    considered strategic choice, and including them would bias the comparison
    toward whatever lap the safety car happened to appear on.
    """
    stops = raw_laps[raw_laps["IsPitInLap"] & (raw_laps["TrackStatus"] == "1")]
    if stops.empty:
        return pd.DataFrame()

    per_driver = (
        stops.groupby(["Season", "Circuit", "Driver"])["LapNumber"]
        .agg(NStops="size", FirstStopLap="min")
        .reset_index()
    )
    return (
        per_driver.groupby(["Season", "Circuit"])
        .agg(
            MedianFirstStopLap=("FirstStopLap", "median"),
            MedianStopsPerDriver=("NStops", "median"),
            NDrivers=("Driver", "size"),
        )
        .reset_index()
    )


def compare_to_actual(summaries: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Put the model's recommendation next to reality and report the gap.

    This is the honesty check. The model ignores traffic, the undercut,
    safety-car probability and tyre allocation, so it should not match reality
    exactly - and where it disagrees, the disagreement is the interesting
    result, not an embarrassment.

    Merged on season *and* circuit. Merging on circuit alone works only while
    the study covers a single season, and fails silently rather than loudly the
    day a second one is added.
    """
    if summaries.empty or actual.empty:
        return pd.DataFrame()

    keys = ["Season", "Circuit"] if "Season" in summaries.columns else ["Circuit"]
    merged = summaries.merge(actual, on=keys, how="left")
    merged["StopLapError"] = merged["BestOneStopLap"] - merged["MedianFirstStopLap"]
    merged["ModelStopsEarlier"] = merged["StopLapError"] < 0
    if "RecommendedStops" in merged.columns:
        merged["StopCountAgrees"] = (
            merged["RecommendedStops"] == merged["MedianStopsPerDriver"].round()
        )
    return merged

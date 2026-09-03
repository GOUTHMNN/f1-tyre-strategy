"""How much do the conclusions depend on the assumptions?

Two sweeps, answering two different questions.

`--fuel` (the default) sweeps the fuel-correction constants. If the recommended
stop lap swings wildly across their plausible range, the headline number is an
artefact of a guess and should be presented as such. Since the joint model now
estimates the fuel coefficient from the laps, this sweep also shows how much of
the study still depends on the assumed value - which should be much less than it
once did.

`--undercut` asks the question the model-versus-reality gap raises. The tyre
model recommends stopping far later than teams actually do, and the obvious
suspect is track position, which the model cannot see. Rather than assert that,
the sweep credits each stop with an undercut of increasing size and reports how
large it would have to be to reconcile the two. A plausible answer supports the
explanation; an implausible one means something else is missing.

    python sensitivity.py                # fuel sweep on data/laps_raw.parquet
    python sensitivity.py --undercut     # undercut sweep
    python sensitivity.py --demo         # either sweep, on simulated races
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import logging

import numpy as np
import pandas as pd

from src.clean import clean_laps
from src.config import ASSUMPTIONS
from src.data import load_raw
from src.model import (
    estimate_pit_loss,
    fit_joint_model,
    fit_stint_slopes,
    observed_stint_limits,
    pool_degradation,
)
from src.strategy import actual_stop_laps, optimal_summary

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("sensitivity")

FUEL_START_GRID = [90.0, 100.0, 110.0]
FUEL_EFFECT_GRID = [0.030, 0.035, 0.040]
UNDERCUT_GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
CAP_MARGIN_GRID = [0, 2, 3, 5, 8, 12]


def _circuit_inputs(raw: pd.DataFrame, a):
    """Everything the optimiser needs, per circuit, under one set of assumptions."""
    laps, _ = clean_laps(raw, a)
    pooled = pool_degradation(fit_stint_slopes(laps), a)
    joint, _ = fit_joint_model(laps, a)
    limits = observed_stint_limits(laps, a)
    pit_loss_df = estimate_pit_loss(raw)

    usable = pooled[pooled["Usable"]] if not pooled.empty else pd.DataFrame()
    pit_loss = (
        pit_loss_df.set_index("Circuit")["PitLossMedian"].to_dict()
        if not pit_loss_df.empty
        else {}
    )
    total_laps = raw.groupby("Circuit")["TotalLaps"].max().to_dict()

    for circuit in sorted(usable["Circuit"].unique()) if not usable.empty else []:
        deg = usable[usable["Circuit"] == circuit].set_index("Compound")["DegSecPerLap"].to_dict()
        offs = (
            joint[joint["Circuit"] == circuit].set_index("Compound")["OffsetSec"].to_dict()
            if not joint.empty
            else {}
        )
        shared = {c: deg[c] for c in deg if c in offs}
        if len(shared) < 2 or circuit not in pit_loss:
            continue
        caps = (
            limits[limits["Circuit"] == circuit]
            .set_index("Compound")["MaxAllowedStintLaps"]
            .to_dict()
        )
        yield circuit, int(total_laps[circuit]), shared, offs, float(pit_loss[circuit]), caps


def sweep_fuel(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fuel_start, fuel_effect in itertools.product(FUEL_START_GRID, FUEL_EFFECT_GRID):
        a = dataclasses.replace(
            ASSUMPTIONS, fuel_start_kg=fuel_start, fuel_effect_s_per_kg=fuel_effect
        )
        for circuit, laps_total, deg, offs, loss, caps in _circuit_inputs(raw, a):
            summary = optimal_summary(circuit, laps_total, deg, offs, loss, max_stint=caps, a=a)
            rows.append(
                {
                    "Circuit": circuit,
                    "FuelStartKg": fuel_start,
                    "FuelEffectSPerKg": fuel_effect,
                    "BestOneStopLap": summary.get("BestOneStopLap"),
                    "BestOneStopPlan": summary.get("BestOneStopPlan"),
                    "RecommendedStops": summary.get("RecommendedStops"),
                    "SoftDeg": deg.get("SOFT"),
                    "MediumDeg": deg.get("MEDIUM"),
                    "HardDeg": deg.get("HARD"),
                }
            )
    return pd.DataFrame(rows)


def sweep_undercut(raw: pd.DataFrame) -> pd.DataFrame:
    """What does crediting each stop with an undercut actually change?

    Note carefully what this sweep can and cannot show. The credit enters
    `evaluate_plan` as `(pit_loss - undercut) * n_stops`, so within the set of
    one-stop plans it subtracts the same constant from every candidate and
    cannot move the optimal stop lap at all. What it moves is the balance
    between stopping once and stopping twice.

    That limitation is the point rather than a defect to hide: it demonstrates
    that track position modelled as a flat per-stop credit cannot explain why
    the model stops later than the teams do. Explaining that needs a term whose
    value depends on *when* the stop happens, which this study has no data to
    calibrate - so the search for the late bias moves to `late_stint_penalty`
    instead, where there is evidence to work with.
    """
    actual = actual_stop_laps(raw)
    actual_by_circuit = (
        actual.set_index("Circuit")["MedianFirstStopLap"].to_dict() if not actual.empty else {}
    )

    rows = []
    for undercut in UNDERCUT_GRID:
        a = dataclasses.replace(ASSUMPTIONS, undercut_gain_s=undercut)
        for circuit, laps_total, deg, offs, loss, caps in _circuit_inputs(raw, a):
            summary = optimal_summary(circuit, laps_total, deg, offs, loss, max_stint=caps, a=a)
            stop = summary.get("BestOneStopLap")
            real = actual_by_circuit.get(circuit, np.nan)
            rows.append(
                {
                    "Circuit": circuit,
                    "UndercutGainSeconds": undercut,
                    "BestOneStopLap": stop,
                    "RecommendedStops": summary.get("RecommendedStops"),
                    "MedianFirstStopLap": real,
                    "StopLapError": (stop - real) if stop is not None else np.nan,
                    "TwoStopMinusOneStopSeconds": summary.get("TwoStopMinusOneStopSeconds"),
                }
            )
    return pd.DataFrame(rows)


def sweep_stint_caps(raw: pd.DataFrame) -> pd.DataFrame:
    """How much of the answer rests on the observed stint-length cap?

    The cap is the least principled input in the study - it says a tyre cannot
    do more than the field was seen to do, which is revealed preference dressed
    as physics. At Bahrain it is the difference between "one-stop infeasible"
    and "one-stop optimal", decided by a single lap, so its influence has to be
    measured rather than asserted.
    """
    rows = []
    for margin in CAP_MARGIN_GRID:
        a = dataclasses.replace(ASSUMPTIONS, max_stint_margin_laps=margin)
        for circuit, laps_total, deg, offs, loss, caps in _circuit_inputs(raw, a):
            summary = optimal_summary(circuit, laps_total, deg, offs, loss, max_stint=caps, a=a)
            rows.append(
                {
                    "Circuit": circuit,
                    "CapMarginLaps": margin,
                    "OneStopFeasible": summary.get("OneStopFeasible"),
                    "BestOneStopLap": summary.get("BestOneStopLap"),
                    "BestOneStopLapUncapped": summary.get("BestOneStopLapUncapped"),
                    "RecommendedStops": summary.get("RecommendedStops"),
                }
            )
    return pd.DataFrame(rows)


def _report_caps(results: pd.DataFrame) -> None:
    log.info("Effect of the stint-length cap margin:")
    for circuit, group in results.groupby("Circuit"):
        feasible = group[group["OneStopFeasible"]]
        laps = group["BestOneStopLap"].dropna()
        uncapped = group["BestOneStopLapUncapped"].dropna()
        log.info(
            "  %-16s one-stop feasible at margins %s  |  stop lap %s  |  uncapped %s",
            circuit,
            sorted(feasible["CapMarginLaps"].tolist()) or "none",
            f"{int(laps.min())}-{int(laps.max())}" if len(laps) else "n/a",
            f"{int(uncapped.min())}" if len(uncapped) else "n/a",
        )
    log.info(
        "A circuit whose stop count flips inside this grid is one where the cap, not "
        "the tyre data, is making the recommendation."
    )


def _report_fuel(results: pd.DataFrame) -> None:
    spread = (
        results.groupby("Circuit")["BestOneStopLap"]
        .agg(MinLap="min", MaxLap="max", Median="median")
        .assign(SwingLaps=lambda d: d["MaxLap"] - d["MinLap"])
    )
    plan_stability = results.groupby("Circuit")["BestOneStopPlan"].nunique()

    log.info("Recommended stop lap across the fuel-assumption grid:")
    for circuit, row in spread.iterrows():
        log.info(
            "  %-16s laps %d-%d (median %d, swing %d)  |  %d distinct compound plan(s)",
            circuit, row["MinLap"], row["MaxLap"], row["Median"], row["SwingLaps"],
            plan_stability[circuit],
        )
    log.info(
        "Read this honestly: a swing comparable to the 1-second window from the main "
        "run means the assumptions matter as much as the data, and the stop lap should "
        "be quoted as a range."
    )


def _report_undercut(results: pd.DataFrame) -> None:
    log.info("Effect of an undercut credit, by circuit:")
    for circuit, group in results.groupby("Circuit"):
        laps = group["BestOneStopLap"].dropna().unique()
        counts = sorted(group["RecommendedStops"].dropna().unique().tolist())
        margins = group["TwoStopMinusOneStopSeconds"].dropna()
        log.info(
            "  %-16s stop lap %s  |  stop count %s  |  two-stop margin %s",
            circuit,
            "unchanged" if len(laps) <= 1 else f"{int(min(laps))}-{int(max(laps))}",
            "unchanged" if len(counts) <= 1 else counts,
            f"{margins.min():+.1f} to {margins.max():+.1f} s" if len(margins) else "n/a",
        )
    log.info(
        "The stop lap is unchanged everywhere, and that is structural, not a null "
        "result: a flat per-stop credit subtracts the same constant from every "
        "one-stop plan, so it can only move the one-versus-two-stop decision. "
        "Track position modelled this way therefore cannot account for the model "
        "stopping later than the teams did - see the late-stint penalty for a "
        "candidate that the data does support."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="use simulated races instead of real data")
    parser.add_argument("--undercut", action="store_true", help="sweep undercut credit instead of fuel")
    parser.add_argument("--caps", action="store_true", help="sweep the stint-length cap margin")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.demo:
        from src.synthetic import simulate_race

        raw = pd.concat(
            [simulate_race(circuit=n, total_laps=t, seed=s)
             for n, t, s in [("Synthetic Park", 57, 7), ("Synthetic Ring", 44, 23)]],
            ignore_index=True,
        )
    else:
        raw = load_raw()

    if args.caps:
        results = sweep_stint_caps(raw)
        out = args.out or "outputs/sensitivity_caps.csv"
        results.to_csv(out, index=False)
        _report_caps(results)
    elif args.undercut:
        results = sweep_undercut(raw)
        out = args.out or "outputs/sensitivity_undercut.csv"
        results.to_csv(out, index=False)
        _report_undercut(results)
    else:
        results = sweep_fuel(raw)
        out = args.out or "outputs/sensitivity_fuel.csv"
        results.to_csv(out, index=False)
        _report_fuel(results)

    log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

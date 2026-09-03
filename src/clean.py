"""Turning raw laps into laps that actually measure tyre degradation.

Most of the honest work in this project is here. A raw lap time is the sum of
tyre condition, fuel load, traffic, track status, driver effort and luck. The
cleaning steps below remove the effects we can identify, and the fuel correction
removes the one large effect we can model. What is left is close enough to a
degradation signal to fit a line through — and the residual diagnostics in
`model.py` are there to check that claim rather than assume it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ASSUMPTIONS, DRY_COMPOUNDS, Assumptions

STINT_KEYS = ["Season", "Circuit", "Driver", "Stint"]


def _green_flag_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep laps run entirely under green flags.

    FastF1 concatenates every track-status code seen during a lap into one
    string, so '1' means the lap was green from start to finish. Anything
    containing a 4 (safety car), 5 (red flag), 6/7 (VSC) or 2 (yellow) is a lap
    driven to a delta, not to the limit, and tells us nothing about the tyre.
    """
    return df[df["TrackStatus"] == "1"]


def _drop_pit_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Remove in-laps and out-laps, which measure the pit lane, not the tyre."""
    return df[~df["IsPitInLap"] & ~df["IsPitOutLap"]]


def _drop_warmup(df: pd.DataFrame, a: Assumptions) -> pd.DataFrame:
    """Remove the first racing lap(s) of each stint, after the out-lap.

    Tyre life 1 is the out-lap (or, in the first stint, the standing-start lap),
    which is not a racing lap at all. The warm-up lap this assumption is about is
    tyre life 2: the first flying lap, on a tyre that has not yet reached its
    working window.

    The obvious `TyreLife > stint_warmup_laps` is wrong and was the bug here for
    a while - it only ever removed tyre life 1, which the pit and accuracy
    filters had already taken, so the assumption was documented, swept in the
    sensitivity analysis, and silently inert. The out-lap has to be discounted
    first for the number to mean what its name says.
    """
    return df[df["TyreLife"] > 1 + a.stint_warmup_laps]


def _drop_mad_outliers(df: pd.DataFrame, a: Assumptions) -> pd.DataFrame:
    """Reject laps far from the stint median, using median absolute deviation.

    MAD rather than standard deviation because a single lap stuck behind a
    backmarker can be 4 seconds slow and would drag a mean-based threshold out
    far enough to keep itself. The rejection is deliberately one-sided-ish in
    effect: traffic makes laps slower, almost never faster.
    """

    df = df.copy()
    median = df.groupby(STINT_KEYS)["LapTimeSeconds"].transform("median")
    df["_AbsDev"] = (df["LapTimeSeconds"] - median).abs()
    mad = df.groupby(STINT_KEYS)["_AbsDev"].transform("median")

    # 1.4826 scales MAD to a standard-deviation equivalent for normal data.
    robust_sigma = 1.4826 * mad
    # A stint with zero spread gives no basis for rejection, so keep it whole.
    keep = (robust_sigma <= 0) | ~np.isfinite(robust_sigma)
    keep = keep | (df["_AbsDev"] <= a.outlier_mad_threshold * robust_sigma)

    return df[keep].drop(columns=["_AbsDev"])


def _drop_short_stints(df: pd.DataFrame, a: Assumptions) -> pd.DataFrame:
    sizes = df.groupby(STINT_KEYS)["LapNumber"].transform("size")
    return df[sizes >= a.min_stint_laps]


def add_fuel_correction(df: pd.DataFrame, a: Assumptions = ASSUMPTIONS) -> pd.DataFrame:
    """Add a fuel-corrected lap time, normalised to an empty car.

    A car burns fuel monotonically through a race, so it gets lighter and faster
    lap by lap. Left uncorrected, this masks degradation: the tyre slows the car
    down while the fuel burn speeds it up, and a naive fit underestimates
    degradation badly. Assuming a linear burn across the scheduled race
    distance:

        fuel_remaining(lap) = fuel_start * (1 - (lap - 1) / total_laps)
        corrected           = observed - fuel_remaining * fuel_effect_s_per_kg

    Linear burn is an approximation - real consumption varies with lift-and-coast
    and safety cars - and the fuel effect itself is an estimate. Both are
    exposed in `config.Assumptions` and swept in the sensitivity analysis.
    """
    df = df.copy()
    laps_done = (df["LapNumber"] - 1.0).clip(lower=0.0)
    fraction_remaining = (1.0 - laps_done / df["TotalLaps"]).clip(lower=0.0, upper=1.0)
    df["FuelRemainingKg"] = a.fuel_start_kg * fraction_remaining
    df["FuelPenaltySeconds"] = df["FuelRemainingKg"] * a.fuel_effect_s_per_kg
    df["LapTimeFuelCorrected"] = df["LapTimeSeconds"] - df["FuelPenaltySeconds"]
    return df


def clean_laps(df: pd.DataFrame, a: Assumptions = ASSUMPTIONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline.

    Returns the cleaned laps and an audit table recording how many laps each
    step removed. The audit is not decoration: if a step is silently discarding
    most of the data, the conclusions are worthless and the reader deserves to
    see it.
    """
    audit = []

    def record(step: str, frame: pd.DataFrame) -> pd.DataFrame:
        audit.append({"Step": step, "LapsRemaining": len(frame)})
        return frame

    out = record("raw", df)
    out = record("valid lap time", out[out["LapTimeSeconds"].notna()])
    out = record("flagged accurate by FastF1", out[out["IsAccurate"].fillna(False).astype(bool)])
    out = record("dry compounds only", out[out["Compound"].isin(DRY_COMPOUNDS)])
    out = record("green flag only", _green_flag_only(out))
    out = record("pit in/out laps dropped", _drop_pit_laps(out))
    out = record("stint warm-up laps dropped", _drop_warmup(out, a))
    out = record("stint outliers removed", _drop_mad_outliers(out, a))
    out = record("short stints dropped", _drop_short_stints(out, a))

    out = add_fuel_correction(out, a)

    audit_df = pd.DataFrame(audit)
    audit_df["RemovedByStep"] = audit_df["LapsRemaining"].shift(1) - audit_df["LapsRemaining"]
    audit_df["PctOfRaw"] = (100 * audit_df["LapsRemaining"] / max(len(df), 1)).round(1)

    return out.reset_index(drop=True), audit_df

"""Estimating degradation rate, compound pace offsets and pit-lane loss.

The modelling choices here are deliberately conservative. A stint gives maybe
15-25 usable laps of a noisy signal, so the aim is a defensible slope with an
honest interval around it, not the lowest possible training error.

Two estimators run side by side:

* a per-stint Theil-Sen fit, pooled across stints. Robust, transparent, and it
  makes no cross-stint assumptions - but it can only see compound pace *after*
  the fuel correction has already been applied, so it inherits whatever error is
  in the assumed fuel coefficient.
* a per-circuit joint regression that estimates the fuel coefficient, the
  per-compound degradation slopes and the per-compound new-tyre offsets
  together, from raw lap times.

Both are kept because agreement between them is evidence and disagreement is a
finding. The joint model is authoritative for compound offsets; the Theil-Sen
pooling stays authoritative for degradation, where its robustness against
traffic matters more than the joint model's efficiency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .clean import STINT_KEYS
from .config import ASSUMPTIONS, DRY_COMPOUNDS, Assumptions


# ---------------------------------------------------------------------------
# Per-stint degradation slopes
# ---------------------------------------------------------------------------

def fit_stint_slopes(laps: pd.DataFrame) -> pd.DataFrame:
    """Fit one degradation slope per stint using Theil-Sen regression.

    Theil-Sen takes the median of the slopes between all pairs of points. It has
    a breakdown point of about 29%, meaning up to roughly a third of the laps in
    a stint can be corrupted by traffic before the estimate is dragged off - and
    traffic is the dominant contaminant in race lap times. Ordinary least
    squares has a breakdown point of zero: one bad lap moves the answer.

    The slope is in seconds lost per lap of tyre life; the intercept is the
    fuel-corrected pace of a notionally new tyre.
    """
    rows = []
    for keys, stint in laps.groupby(STINT_KEYS):
        x = stint["TyreLife"].to_numpy(dtype=float)
        y = stint["LapTimeFuelCorrected"].to_numpy(dtype=float)
        if len(x) < 3 or np.ptp(x) == 0:
            continue

        slope, intercept, lo, hi = stats.theilslopes(y, x, alpha=0.95)
        residuals = y - (intercept + slope * x)

        rows.append(
            {
                **dict(zip(STINT_KEYS, keys)),
                "Team": stint["Team"].iloc[0],
                "Compound": stint["Compound"].iloc[0],
                "NLaps": len(x),
                "TyreLifeStart": float(x.min()),
                "TyreLifeEnd": float(x.max()),
                "SlopeSecPerLap": float(slope),
                "SlopeCILow": float(lo),
                "SlopeCIHigh": float(hi),
                "InterceptSec": float(intercept),
                "ResidualStd": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _bootstrap_mean(
    values: np.ndarray, weights: np.ndarray, std_errors: np.ndarray, a: Assumptions
) -> tuple[float, float]:
    """Two-level percentile bootstrap CI for a weighted mean of stint slopes.

    There are two independent sources of uncertainty and an interval that
    ignores either one is too narrow:

    1. Between stints - different drivers, cars and phases of the race genuinely
       degrade at different rates. Captured by resampling whole stints.
    2. Within a stint - each slope is itself an estimate from ~20 noisy laps.
       Captured by perturbing each resampled slope by its own standard error.

    An earlier version modelled only (1) and produced intervals so tight that a
    known-correct value fell outside them in synthetic testing, which is what
    prompted the second level. Resampling laps instead would be worse still:
    laps within a stint are not independent, so that interval would be
    narrower again and more wrong.
    """
    rng = np.random.default_rng(a.random_seed)
    n = len(values)
    if n < 2:
        return (np.nan, np.nan)

    se = np.where(np.isfinite(std_errors), std_errors, 0.0)
    draws = np.empty(a.n_bootstrap)
    for i in range(a.n_bootstrap):
        idx = rng.integers(0, n, n)
        w = weights[idx]
        perturbed = values[idx] + rng.normal(0.0, se[idx])
        draws[i] = np.average(perturbed, weights=w) if w.sum() > 0 else np.nan

    return tuple(np.nanpercentile(draws, [2.5, 97.5]))


def pool_degradation(stint_fits: pd.DataFrame, a: Assumptions = ASSUMPTIONS) -> pd.DataFrame:
    """Pool stint slopes into one degradation rate per circuit and compound.

    Stints are weighted by length, since a 22-lap stint constrains a slope far
    better than a 7-lap one. The confidence interval comes from a stint-level
    bootstrap, so it reflects disagreement between stints - different drivers,
    cars and phases of the race - rather than the artificially tight scatter of
    laps within a single stint.

    Every cell is returned, including the ones that fail the evidence guards,
    with `Usable` and `ExcludedReason` recording which and why. Filtering
    silently would hide how thin a circuit's coverage is; callers filter on
    `Usable` before optimising, and the excluded rows are written to disk so the
    exclusions can be argued with.
    """
    rows = []
    for (circuit, compound), group in stint_fits.groupby(["Circuit", "Compound"]):
        values = group["SlopeSecPerLap"].to_numpy(dtype=float)
        weights = group["NLaps"].to_numpy(dtype=float)
        # Theil-Sen returns a 95% interval; converting back to a standard error
        # lets the bootstrap propagate each stint's own estimation uncertainty.
        std_errors = (
            group["SlopeCIHigh"].to_numpy(dtype=float)
            - group["SlopeCILow"].to_numpy(dtype=float)
        ) / (2 * 1.96)

        mask = np.isfinite(values) & np.isfinite(weights)
        values, weights, std_errors = values[mask], weights[mask], std_errors[mask]
        if len(values) == 0:
            continue

        mean = float(np.average(values, weights=weights))
        lo, hi = _bootstrap_mean(values, weights, std_errors, a)

        n_stints, n_laps = int(len(values)), int(weights.sum())
        reason = ""
        if n_stints < a.min_stints_for_pooling:
            reason = f"only {n_stints} stint(s), need {a.min_stints_for_pooling}"
        elif n_laps < a.min_laps_for_pooling:
            reason = f"only {n_laps} laps, need {a.min_laps_for_pooling}"
        elif not (np.isfinite(lo) and np.isfinite(hi)):
            reason = "no finite confidence interval"
        elif a.reject_negative_degradation and mean < 0:
            reason = f"negative degradation ({mean:+.4f} s/lap) is not physical"

        rows.append(
            {
                "Circuit": circuit,
                "Compound": compound,
                "NStints": n_stints,
                "NLaps": n_laps,
                "DegSecPerLap": mean,
                "DegCILow": lo,
                "DegCIHigh": hi,
                "StintSpread": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "Usable": reason == "",
                "ExcludedReason": reason,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Circuit", "Compound"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Joint per-circuit model: fuel, degradation and compound offsets together
# ---------------------------------------------------------------------------

def _huber_irls(
    X: np.ndarray, y: np.ndarray, delta: float, n_iter: int = 25
) -> tuple[np.ndarray, int]:
    """Robust linear fit by iteratively reweighted least squares.

    Least squares assumes the errors are Gaussian. Race lap times are not: a lap
    spent behind a car that will not move is several seconds slow, and there are
    hundreds of them. Huber weighting keeps the efficiency of least squares in
    the middle of the distribution and down-weights the tail linearly, so
    traffic informs the fit without dictating it.
    """
    w = np.ones(len(y))
    beta = np.zeros(X.shape[1])
    rank = 0
    for _ in range(n_iter):
        sw = np.sqrt(w)
        beta, _, rank, _ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        resid = y - X @ beta
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if not np.isfinite(scale) or scale <= 1e-9:
            break
        u = np.abs(resid / scale)
        w_new = np.where(u <= delta / scale, 1.0, (delta / scale) / np.maximum(u, 1e-9))
        if np.max(np.abs(w_new - w)) < 1e-4:
            w = w_new
            break
        w = w_new
    return beta, int(rank)


def _fuel_fraction(df: pd.DataFrame) -> np.ndarray:
    """Fraction of the starting fuel load still on board, in [0, 1].

    Deliberately *not* expressed in kilograms. The mass of fuel a car starts
    with is not public, so a coefficient in seconds-per-kg cannot be estimated
    without assuming the very number that is unknown. In fractional units the
    fitted coefficient means "seconds per full tank", which the data can
    identify on its own; dividing by an assumed start mass afterwards is a
    presentational step rather than an input to the estimate.
    """
    laps_done = (df["LapNumber"].to_numpy(dtype=float) - 1.0).clip(min=0.0)
    return np.clip(1.0 - laps_done / df["TotalLaps"].to_numpy(dtype=float), 0.0, 1.0)


def _build_design(sub: pd.DataFrame, compounds: list[str], reference: str):
    """Design matrix for one circuit's joint fit.

        lap time = driver baseline
                 + compound offset (relative to the reference compound)
                 + compound degradation x tyre age
                 + fuel coefficient x fraction of tank remaining

    Driver dummies rather than a single intercept: the alternative is to compare
    a Red Bull on hards against a Sauber on softs and call the difference a tyre
    property. The driver term absorbs car and driver pace, so the compound terms
    are estimated within a driver.

    Crucially, fuel enters as its own regressor instead of being subtracted
    beforehand. Within a single stint, tyre age and fuel load are perfectly
    collinear - the tyre ages exactly as the tank empties - so nothing about
    fuel can be learned from one stint in isolation. Identification comes from
    *across* stints: the same driver starts stint 2 on a fresh tyre with a much
    lighter car, and it is that contrast which separates the two effects.
    """
    # Driver *and season*: the same name in 2022 and 2024 is a different car, a
    # different power unit and a different aero regulation. Pooling those into
    # one baseline would push several seasons of car development into the
    # compound terms, which is the very confusion this model exists to avoid.
    entrant = sub["Season"].astype(str) + "|" + sub["Driver"].astype(str)
    entrants = sorted(entrant.unique())
    non_ref = [c for c in compounds if c != reference]

    cols, names = [], []
    for d in entrants:
        cols.append((entrant == d).to_numpy(dtype=float))
        names.append(f"driver[{d}]")
    for c in non_ref:
        cols.append((sub["Compound"] == c).to_numpy(dtype=float))
        names.append(f"offset[{c}]")

    tyre_life = sub["TyreLife"].to_numpy(dtype=float)
    for c in compounds:
        cols.append(tyre_life * (sub["Compound"] == c).to_numpy(dtype=float))
        names.append(f"deg[{c}]")

    cols.append(_fuel_fraction(sub))
    names.append("fuel")

    return np.column_stack(cols), names


def _fuel_overlap(sub: pd.DataFrame, compound: str, reference: str) -> float:
    """How much two compounds' fuel-load ranges overlap, as a Jaccard index.

    This is the identifiability check, and it is the most important diagnostic
    in the file. If a circuit ran softs only in the opening stint and hards only
    at the end, then "soft tyre" and "heavy car" are the same column of the
    design matrix, and no estimator - however robust - can separate a soft-tyre
    pace penalty from a fuel-load penalty. The regression will still return a
    number. That number will be meaningless, and the only defence is to say so.

    Computed on interquartile ranges rather than full ranges, so one unusual
    stint cannot manufacture overlap the bulk of the data does not have. 0 means
    the compounds were completely separated in race phase; 1 means they were run
    at indistinguishable fuel loads.
    """
    f = _fuel_fraction(sub)
    a_mask = (sub["Compound"] == compound).to_numpy()
    b_mask = (sub["Compound"] == reference).to_numpy()
    if a_mask.sum() < 4 or b_mask.sum() < 4:
        return 0.0

    a_lo, a_hi = np.percentile(f[a_mask], [25, 75])
    b_lo, b_hi = np.percentile(f[b_mask], [25, 75])

    inter = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
    union = max(a_hi, b_hi) - min(a_lo, b_lo)
    return float(inter / union) if union > 1e-9 else 1.0


def fit_joint_model(
    laps: pd.DataFrame, a: Assumptions = ASSUMPTIONS, bootstrap: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the joint fuel/degradation/offset model, one circuit at a time.

    Returns (per-compound estimates, per-circuit fuel estimates). Confidence
    intervals come from a cluster bootstrap that resamples whole stints, because
    laps within a stint are anything but independent and resampling them
    individually would produce intervals that are narrow and wrong.

    `bootstrap=False` returns the point estimates with the intervals left as NaN.
    The bootstrap is the whole cost of this function - several hundred refits of
    a robust regression - and the assumption sweeps call it dozens of times while
    reading nothing but the point estimates. Skipping it there turns a sweep from
    minutes into seconds and changes no number that the sweep reports.
    """
    rng = np.random.default_rng(a.random_seed)
    compound_rows, fuel_rows = [], []

    for circuit, sub in laps.groupby("Circuit"):
        sub = sub.reset_index(drop=True)
        compounds = sorted(sub["Compound"].unique())
        if len(compounds) < 2 or len(sub) < 50:
            continue

        # Reference is the best-measured compound, so the baseline everything
        # else is quoted against rests on the most data.
        reference = sub["Compound"].value_counts().idxmax()

        X, names = _build_design(sub, compounds, reference)
        y = sub["LapTimeSeconds"].to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X, y, sub_ok = X[ok], y[ok], sub[ok]
        if len(y) < 50:
            continue

        beta, rank = _huber_irls(X, y, a.huber_delta_s)
        full_rank = rank >= X.shape[1]
        cond = float(np.linalg.cond(X))

        # Cluster bootstrap over whole stints.
        stint_ids = sub_ok.groupby(STINT_KEYS, sort=False).ngroup().to_numpy()
        groups = [np.flatnonzero(stint_ids == g) for g in np.unique(stint_ids)]
        draws = np.full((a.n_bootstrap_joint if bootstrap else 0, X.shape[1]), np.nan)
        for b in range(draws.shape[0]):
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[p] for p in pick])
            try:
                draws[b], _ = _huber_irls(X[idx], y[idx], a.huber_delta_s, n_iter=6)
            except np.linalg.LinAlgError:
                continue

        def ci(j: int) -> tuple[float, float]:
            if draws.shape[0] == 0:
                return (np.nan, np.nan)
            col = draws[:, j]
            col = col[np.isfinite(col)]
            if len(col) < 20:
                return (np.nan, np.nan)
            return tuple(np.percentile(col, [2.5, 97.5]))

        index = {n: i for i, n in enumerate(names)}

        fuel_j = index["fuel"]
        fuel_lo, fuel_hi = ci(fuel_j)
        fuel_rows.append(
            {
                "Circuit": circuit,
                "ReferenceCompound": reference,
                "NLaps": int(len(y)),
                "NStints": int(len(groups)),
                # Seconds per full tank, plus the same number divided by the
                # assumed start mass so it can be read against the 0.03-0.04
                # s/kg figure the literature quotes.
                "FuelSecPerFullTank": float(beta[fuel_j]),
                "FuelCILow": fuel_lo,
                "FuelCIHigh": fuel_hi,
                "ImpliedSecPerKg": float(beta[fuel_j]) / a.fuel_start_kg,
                "AssumedSecPerKg": a.fuel_effect_s_per_kg,
                "DesignFullRank": bool(full_rank),
                "ConditionNumber": cond,
            }
        )

        for c in compounds:
            raw_offset = 0.0 if c == reference else float(beta[index[f"offset[{c}]"]])
            off_lo, off_hi = (0.0, 0.0) if c == reference else ci(index[f"offset[{c}]"])
            deg_j = index[f"deg[{c}]"]
            deg_lo, deg_hi = ci(deg_j)
            overlap = 1.0 if c == reference else _fuel_overlap(sub_ok, c, reference)

            compound_rows.append(
                {
                    "Circuit": circuit,
                    "Compound": c,
                    "IsReference": c == reference,
                    "NLaps": int((sub_ok["Compound"] == c).sum()),
                    "RawOffsetSec": raw_offset,
                    "OffsetCILow": off_lo,
                    "OffsetCIHigh": off_hi,
                    "JointDegSecPerLap": float(beta[deg_j]),
                    "JointDegCILow": deg_lo,
                    "JointDegCIHigh": deg_hi,
                    "FuelOverlap": overlap,
                    "OffsetIdentified": bool(overlap >= a.min_fuel_overlap),
                }
            )

    compounds_df = pd.DataFrame(compound_rows)
    fuel_df = pd.DataFrame(fuel_rows)

    if not compounds_df.empty:
        # Re-express relative to the fastest compound at each circuit, matching
        # the convention used everywhere else in the study.
        compounds_df["OffsetSec"] = compounds_df["RawOffsetSec"] - compounds_df.groupby(
            "Circuit"
        )["RawOffsetSec"].transform("min")
        compounds_df = compounds_df.sort_values(["Circuit", "OffsetSec"]).reset_index(drop=True)

    return compounds_df, fuel_df


def identifiability_report(
    joint_compounds: pd.DataFrame, a: Assumptions = ASSUMPTIONS
) -> pd.DataFrame:
    """One row per circuit saying whether its compound offsets mean anything."""
    if joint_compounds.empty:
        return pd.DataFrame()

    rows = []
    for circuit, group in joint_compounds.groupby("Circuit"):
        non_ref = group[~group["IsReference"]]
        identified = bool(non_ref["OffsetIdentified"].all()) if len(non_ref) else False
        rows.append(
            {
                "Circuit": circuit,
                "ReferenceCompound": group[group["IsReference"]]["Compound"].iloc[0],
                "MinFuelOverlap": float(non_ref["FuelOverlap"].min()) if len(non_ref) else np.nan,
                "OffsetsIdentified": identified,
                "Verdict": (
                    "offsets usable"
                    if identified
                    else "compound confounded with fuel load - offsets not identified"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("Circuit").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Is degradation actually linear?
# ---------------------------------------------------------------------------

def compare_linear_quadratic(laps: pd.DataFrame, min_laps: int = 10) -> pd.DataFrame:
    """Test the linearity assumption per stint with out-of-sample error.

    A straight line is the standard assumption and it is convenient, but tyres
    are known to fall off a cliff once the surface is gone. Rather than assert
    linearity, each stint is fitted both ways and scored by leave-one-out
    cross-validated RMSE: if the quadratic genuinely predicts held-out laps
    better, the linear model is hiding something.
    """
    rows = []
    for keys, stint in laps.groupby(STINT_KEYS):
        x = stint["TyreLife"].to_numpy(dtype=float)
        y = stint["LapTimeFuelCorrected"].to_numpy(dtype=float)
        n = len(x)
        if n < min_laps:
            continue

        errs = {1: [], 2: []}
        for degree in (1, 2):
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                if np.ptp(x[mask]) == 0:
                    continue
                coeffs = np.polyfit(x[mask], y[mask], degree)
                errs[degree].append(y[i] - np.polyval(coeffs, x[i]))

        if not errs[1] or not errs[2]:
            continue

        rmse_lin = float(np.sqrt(np.mean(np.square(errs[1]))))
        rmse_quad = float(np.sqrt(np.mean(np.square(errs[2]))))
        rows.append(
            {
                **dict(zip(STINT_KEYS, keys)),
                "Compound": stint["Compound"].iloc[0],
                "NLaps": n,
                "LooRmseLinear": rmse_lin,
                "LooRmseQuadratic": rmse_quad,
                "QuadraticWins": rmse_quad < rmse_lin,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compound pace offsets (naive), stint limits and pit loss
# ---------------------------------------------------------------------------

def estimate_compound_offsets(stint_fits: pd.DataFrame) -> pd.DataFrame:
    """Naive compound offsets: centre stint intercepts within each driver.

    Retained as a comparison exhibit rather than as the study's answer. It
    cancels car pace correctly, but it takes the fuel correction as given, so
    any error in the assumed fuel coefficient reappears here as a fake compound
    difference - and because teams run softs early and hards late, that error is
    almost perfectly aligned with compound. `fit_joint_model` is what feeds the
    strategy optimiser; this function survives so the write-up can put the two
    side by side and show what assuming the fuel effect costs.
    """
    df = stint_fits.copy()
    df["DriverMean"] = df.groupby(["Circuit", "Driver"])["InterceptSec"].transform("mean")
    df["Centred"] = df["InterceptSec"] - df["DriverMean"]

    rows = []
    for (circuit, compound), group in df.groupby(["Circuit", "Compound"]):
        rows.append(
            {
                "Circuit": circuit,
                "Compound": compound,
                "NStints": len(group),
                "CentredPace": float(group["Centred"].mean()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["OffsetSec"] = out["CentredPace"] - out.groupby("Circuit")["CentredPace"].transform("min")
    return out.sort_values(["Circuit", "OffsetSec"]).reset_index(drop=True)


def observed_stint_limits(laps: pd.DataFrame, a: Assumptions = ASSUMPTIONS) -> pd.DataFrame:
    """The longest stint actually run on each compound, per circuit.

    The strategy model assumes degradation is linear, which means a long stint
    costs time smoothly and forever. Real tyres do not work that way: past some
    age the surface is gone and the loss becomes a cliff, and an optimiser fed a
    straight line will happily recommend a stint no engineer would sanction -
    which is exactly how one circuit here ended up being told to run 65 of its
    70 laps on a single set of softs.

    Teams know where the limit is and their behaviour reveals it, so the longest
    stint the field actually completed is the best public evidence available. A
    small margin goes on top, because the observed maximum is a sample rather
    than the physical ceiling.
    """
    rows = []
    for (circuit, compound), group in laps.groupby(["Circuit", "Compound"]):
        per_stint = group.groupby(STINT_KEYS)["TyreLife"].max()
        if per_stint.empty:
            continue
        rows.append(
            {
                "Circuit": circuit,
                "Compound": compound,
                "MaxObservedStintLaps": int(per_stint.max()),
                "MedianStintLaps": float(per_stint.median()),
                "MaxAllowedStintLaps": int(per_stint.max()) + a.max_stint_margin_laps,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Circuit", "Compound"]).reset_index(drop=True)


def estimate_pit_loss(raw_laps: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Estimate total pit-lane time loss per circuit, from the laps themselves.

    A stop costs time on the in-lap (slowing for the pit entry), in the pit lane
    itself, and on the out-lap (a cold tyre and the exit). Rather than take a
    published figure, each stop is measured against the driver's own reference
    pace: the median of their nearby green-flag racing laps.

        loss = (in-lap + out-lap) - 2 x reference pace

    This is computed on raw laps, before cleaning removes exactly the in- and
    out-laps it needs. Safety-car stops are excluded, because stopping under a
    safety car is cheap and including those would understate the real cost of a
    green-flag stop - which is the number the strategy model needs.
    """
    rows = []
    for (season, circuit, driver), group in raw_laps.groupby(["Season", "Circuit", "Driver"]):
        group = group.sort_values("LapNumber")
        green = group[(group["TrackStatus"] == "1") & ~group["IsPitInLap"] & ~group["IsPitOutLap"]]
        if len(green) < 5:
            continue

        for _, in_lap in group[group["IsPitInLap"]].iterrows():
            lap_no = in_lap["LapNumber"]
            out_lap = group[group["IsPitOutLap"] & (group["LapNumber"] == lap_no + 1)]
            if out_lap.empty:
                continue
            if in_lap["TrackStatus"] != "1" or out_lap.iloc[0]["TrackStatus"] != "1":
                continue

            nearby = green[green["LapNumber"].between(lap_no - window, lap_no + window + 1)]
            if len(nearby) < 3:
                continue

            reference = nearby["LapTimeSeconds"].median()
            observed = in_lap["LapTimeSeconds"] + out_lap.iloc[0]["LapTimeSeconds"]
            if not np.isfinite(observed) or not np.isfinite(reference):
                continue

            rows.append(
                {
                    "Season": season,
                    "Circuit": circuit,
                    "Driver": driver,
                    "StopOnLap": lap_no,
                    "PitLossSeconds": float(observed - 2 * reference),
                }
            )

    stops = pd.DataFrame(rows)
    if stops.empty:
        return stops

    # Median across stops: robust to the slow stop, the double-stack and the
    # unsafe release, none of which represent a normally executed stop.
    summary = (
        stops.groupby(["Season", "Circuit"])["PitLossSeconds"]
        .agg(
            PitLossMedian="median",
            PitLossIQR=lambda s: s.quantile(0.75) - s.quantile(0.25),
            NStops="size",
        )
        .reset_index()
    )
    return summary


def late_stint_penalty(laps: pd.DataFrame, min_laps: int = 14, train_fraction: float = 0.67) -> pd.DataFrame:
    """How much more does a tyre lose late in a stint than a straight line predicts?

    The strategy model prices a stint with `deg * n(n+1)/2`, which assumes the
    loss per lap is constant forever. If real tyres fall away faster once the
    surface goes, that formula under-costs long stints - and an optimiser that
    under-costs long stints will recommend stopping too late. That is a specific,
    testable explanation for the study's largest disagreement with reality, and
    this function is the test.

    Each long stint is split by tyre age: a line is fitted to the first
    two-thirds and used to predict the last third. The mean residual over the
    predicted laps is the extra time the linear model failed to charge. A
    positive number means degradation accelerates, the straight line is
    optimistic about old tyres, and the recommended stop lap is biased late.
    """
    rows = []
    for keys, stint in laps.groupby(STINT_KEYS):
        stint = stint.sort_values("TyreLife")
        n = len(stint)
        if n < min_laps:
            continue

        x = stint["TyreLife"].to_numpy(dtype=float)
        y = stint["LapTimeFuelCorrected"].to_numpy(dtype=float)
        cut = int(n * train_fraction)
        if cut < 5 or n - cut < 3 or np.ptp(x[:cut]) == 0:
            continue

        slope, intercept, _, _ = stats.theilslopes(y[:cut], x[:cut], alpha=0.95)
        predicted = intercept + slope * x[cut:]
        residual = float(np.mean(y[cut:] - predicted))

        rows.append(
            {
                **dict(zip(STINT_KEYS, keys)),
                "Compound": stint["Compound"].iloc[0],
                "NLaps": n,
                "TrainLaps": cut,
                "LinearSlope": float(slope),
                "LateStintExtraSeconds": residual,
                "DegradationAccelerates": residual > 0,
            }
        )

    return pd.DataFrame(rows)


def summarise_late_stint_penalty(penalty: pd.DataFrame) -> pd.DataFrame:
    """Per-circuit summary of the late-stint penalty, with a sign test.

    The share of stints whose loss accelerates is reported alongside the median
    size, because a small median over a large majority of stints is a systematic
    bias, while a large median over half of them is just noise.
    """
    if penalty.empty:
        return pd.DataFrame()

    return (
        penalty.groupby("Circuit")
        .agg(
            NStints=("LateStintExtraSeconds", "size"),
            MedianExtraSeconds=("LateStintExtraSeconds", "median"),
            ShareAccelerating=("DegradationAccelerates", "mean"),
        )
        .reset_index()
        .sort_values("MedianExtraSeconds", ascending=False)
    )


# ---------------------------------------------------------------------------
# Curvature: pricing the cliff instead of pretending it is a straight line
# ---------------------------------------------------------------------------

def fit_stint_curvature(laps: pd.DataFrame, a: Assumptions = ASSUMPTIONS) -> pd.DataFrame:
    """Fit a robust quadratic to each stint and keep the curvature term.

    `late_stint_penalty` establishes *that* wear accelerates; this measures the
    acceleration in a form the strategy model can price. Fitting

        lap time = c0 + c1 * age + c2 * age^2

    makes c2 the rate at which the loss per lap itself grows. A positive c2 is a
    tyre falling away; zero is the straight line the optimiser used to assume.

    Huber weighting rather than plain least squares, for the same reason the
    linear fits use Theil-Sen: a quadratic is more flexible than a line and so
    even more willing to bend itself around a handful of traffic laps at the end
    of a stint, which is precisely where the curvature is being read from.
    """
    rows = []
    for keys, stint in laps.groupby(STINT_KEYS):
        x = stint["TyreLife"].to_numpy(dtype=float)
        y = stint["LapTimeFuelCorrected"].to_numpy(dtype=float)
        if len(x) < a.min_laps_for_curvature or np.ptp(x) == 0:
            continue

        X = np.column_stack([np.ones_like(x), x, x**2])
        beta, rank = _huber_irls(X, y, a.huber_delta_s)
        if rank < 3:
            continue

        rows.append(
            {
                **dict(zip(STINT_KEYS, keys)),
                "Compound": stint["Compound"].iloc[0],
                "NLaps": len(x),
                "LinearTerm": float(beta[1]),
                "CurvatureTerm": float(beta[2]),
            }
        )

    return pd.DataFrame(rows)


def pool_curvature(
    curvature_fits: pd.DataFrame, a: Assumptions = ASSUMPTIONS
) -> pd.DataFrame:
    """Pool per-stint curvature into one value per circuit and compound.

    Curvature is a second derivative read off ~20 noisy laps, so it is far less
    stable than a slope and is treated with matching suspicion. A pooled value
    reaches the optimiser only when it clears three hurdles: enough stints, a
    bootstrap interval that excludes zero, and a positive sign.

    Where it does not clear them the curvature is set to exactly zero and the
    model falls back to the straight line. That is a deliberate asymmetry. A
    curvature term that is wrong in the positive direction invents a cliff and
    stops the car far too early; falling back to linear merely returns the model
    to the bias it already had and has already reported.
    """
    if curvature_fits.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(a.random_seed)
    rows = []
    for (circuit, compound), group in curvature_fits.groupby(["Circuit", "Compound"]):
        values = group["CurvatureTerm"].to_numpy(dtype=float)
        weights = group["NLaps"].to_numpy(dtype=float)
        mask = np.isfinite(values) & np.isfinite(weights)
        values, weights = values[mask], weights[mask]
        if len(values) == 0:
            continue

        mean = float(np.average(values, weights=weights))

        lo = hi = np.nan
        if len(values) >= 2:
            draws = np.empty(a.n_bootstrap)
            for i in range(a.n_bootstrap):
                idx = rng.integers(0, len(values), len(values))
                w = weights[idx]
                draws[i] = np.average(values[idx], weights=w) if w.sum() > 0 else np.nan
            lo, hi = np.nanpercentile(draws, [2.5, 97.5])

        n_stints = int(len(values))
        reason = ""
        if n_stints < a.min_stints_for_curvature:
            reason = f"only {n_stints} stint(s), need {a.min_stints_for_curvature}"
        elif not np.isfinite(lo):
            reason = "no finite confidence interval"
        elif lo <= 0:
            reason = "interval includes zero, indistinguishable from linear"
        elif mean <= 0:
            reason = "curvature not positive"

        rows.append(
            {
                "Circuit": circuit,
                "Compound": compound,
                "NStints": n_stints,
                "CurvatureSecPerLap2": mean,
                "CurvatureCILow": lo,
                "CurvatureCIHigh": hi,
                "Usable": reason == "",
                # What the optimiser is actually handed: the measured value when
                # it is trustworthy, and an honest zero when it is not.
                "AppliedCurvature": mean if reason == "" else 0.0,
                "ExcludedReason": reason,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Circuit", "Compound"]).reset_index(drop=True)


def season_heterogeneity(
    stint_fits: pd.DataFrame, a: Assumptions = ASSUMPTIONS
) -> pd.DataFrame:
    """Do the seasons agree about a circuit and compound, or are they different tyres?

    Pooling three seasons is what finally makes compound offsets identifiable,
    but it buys that at a price: "SOFT" is a label, not a rubber. Pirelli
    allocates C1-C5 per event and the mapping moves between years, so the soft at
    Barcelona in 2022 may be a materially different tyre from the soft in 2024 -
    and pooling two different tyres into one number produces an average of
    something that never existed.

    The honest defence is to look. Each season's degradation is estimated
    separately and the bootstrap intervals compared; where two seasons' intervals
    are disjoint the pooled figure is flagged as covering compounds that do not
    behave alike. This does not decide anything on its own - it tells the reader
    which pooled cells to distrust.
    """
    if stint_fits.empty:
        return pd.DataFrame()

    per_season = []
    for (season, circuit, compound), group in stint_fits.groupby(
        ["Season", "Circuit", "Compound"]
    ):
        values = group["SlopeSecPerLap"].to_numpy(dtype=float)
        weights = group["NLaps"].to_numpy(dtype=float)
        std_errors = (
            group["SlopeCIHigh"].to_numpy(dtype=float)
            - group["SlopeCILow"].to_numpy(dtype=float)
        ) / (2 * 1.96)
        mask = np.isfinite(values) & np.isfinite(weights)
        values, weights, std_errors = values[mask], weights[mask], std_errors[mask]
        if len(values) < 2:
            continue
        lo, hi = _bootstrap_mean(values, weights, std_errors, a)
        per_season.append(
            {
                "Season": season,
                "Circuit": circuit,
                "Compound": compound,
                "NStints": int(len(values)),
                "DegSecPerLap": float(np.average(values, weights=weights)),
                "CILow": lo,
                "CIHigh": hi,
            }
        )

    seasons_df = pd.DataFrame(per_season)
    if seasons_df.empty:
        return pd.DataFrame()

    rows = []
    for (circuit, compound), group in seasons_df.groupby(["Circuit", "Compound"]):
        group = group.dropna(subset=["CILow", "CIHigh"])
        if len(group) < 2:
            continue

        disjoint = False
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a_row, b_row = group.iloc[i], group.iloc[j]
                if a_row["CIHigh"] < b_row["CILow"] or b_row["CIHigh"] < a_row["CILow"]:
                    disjoint = True

        rows.append(
            {
                "Circuit": circuit,
                "Compound": compound,
                "NSeasons": int(len(group)),
                "Seasons": ",".join(str(int(s)) for s in sorted(group["Season"])),
                "MinDeg": float(group["DegSecPerLap"].min()),
                "MaxDeg": float(group["DegSecPerLap"].max()),
                "SpreadSecPerLap": float(group["DegSecPerLap"].max() - group["DegSecPerLap"].min()),
                "SeasonsDisagree": disjoint,
            }
        )

    if not rows:
        # One season, or no compound measured twice: there is nothing to compare,
        # which is a legitimate state rather than an error. Returning an empty
        # frame keeps callers uniform.
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("SpreadSecPerLap", ascending=False).reset_index(drop=True)


def wet_race_report(raw_laps: pd.DataFrame, a: Assumptions = ASSUMPTIONS) -> pd.DataFrame:
    """Which races were weather-affected, measured rather than remembered.

    Determined from the tyres the field actually fitted: a race where a quarter
    of all laps ran on intermediates was not a dry race, whatever the calendar
    says. Using the compound data rather than a weather feed keeps the rule
    inside the dataset and independent of anything the model predicted.
    """
    df = raw_laps.copy()
    df["IsWetTyre"] = ~df["Compound"].isin(DRY_COMPOUNDS)
    out = (
        df.groupby(["Season", "Circuit"])["IsWetTyre"]
        .agg(WetLapShare="mean", NLaps="size")
        .reset_index()
    )
    out["WetAffected"] = out["WetLapShare"] > a.max_wet_lap_share
    return out.sort_values(["Season", "Circuit"]).reset_index(drop=True)

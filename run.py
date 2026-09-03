"""Run the whole study end to end.

    python run.py fetch     # pull races from the F1 API into data/ (needs network)
    python run.py analyse   # clean, model, optimise, plot (works offline)
    python run.py all       # both
    python run.py demo      # run the pipeline on simulated races, no network

`demo` exists so the pipeline can be exercised and reviewed without the API,
and because a simulated race has a known answer the estimates can be checked
against.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

from src import plots
from src.clean import clean_laps
from src.config import ASSUMPTIONS, CIRCUITS, DATA_DIR, OUTPUT_DIR, SEASONS, TARGET_SEASON
from src.data import load_races, load_raw, save_raw
from src.backtest import rolling_backtest, summarise_backtest
from src.model import (
    compare_linear_quadratic,
    estimate_compound_offsets,
    estimate_pit_loss,
    fit_joint_model,
    fit_stint_curvature,
    fit_stint_slopes,
    identifiability_report,
    late_stint_penalty,
    observed_stint_limits,
    pool_curvature,
    pool_degradation,
    season_heterogeneity,
    summarise_late_stint_penalty,
)
from src.strategy import (
    actual_stop_laps,
    compare_to_actual,
    optimal_summary,
    optimise_one_stop,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("run")


def fetch() -> pd.DataFrame:
    raw = load_races(CIRCUITS, seasons=SEASONS)
    path = save_raw(raw)
    log.info("Saved %d laps to %s", len(raw), path)
    return raw


def analyse(raw: pd.DataFrame, outdir: str = OUTPUT_DIR, a=ASSUMPTIONS) -> dict:
    os.makedirs(outdir, exist_ok=True)

    laps, audit = clean_laps(raw, a)
    log.info("Cleaning kept %d of %d laps (%.0f%%)", len(laps), len(raw), 100 * len(laps) / len(raw))
    audit.to_csv(f"{outdir}/cleaning_audit.csv", index=False)

    # ---- Estimation ------------------------------------------------------
    stint_fits = fit_stint_slopes(laps)
    pooled = pool_degradation(stint_fits, a)
    naive_offsets = estimate_compound_offsets(stint_fits)
    joint_compounds, joint_fuel = fit_joint_model(laps, a)
    identifiability = identifiability_report(joint_compounds, a)
    curvature = pool_curvature(fit_stint_curvature(laps, a), a)
    heterogeneity = season_heterogeneity(stint_fits, a)
    stint_limits = observed_stint_limits(laps, a)
    pit_loss_df = estimate_pit_loss(raw)
    linearity = compare_linear_quadratic(laps)
    late_penalty = late_stint_penalty(laps)
    late_summary = summarise_late_stint_penalty(late_penalty)

    for name, frame in [
        ("stint_fits", stint_fits),
        ("degradation_by_circuit", pooled),
        ("compound_offsets_naive", naive_offsets),
        ("compound_offsets_joint", joint_compounds),
        ("fuel_estimates", joint_fuel),
        ("identifiability", identifiability),
        ("curvature", curvature),
        ("season_heterogeneity", heterogeneity),
        ("stint_limits", stint_limits),
        ("pit_loss", pit_loss_df),
        ("linearity_check", linearity),
        ("late_stint_penalty", late_penalty),
        ("late_stint_penalty_summary", late_summary),
    ]:
        frame.to_csv(f"{outdir}/{name}.csv", index=False)

    if not linearity.empty:
        log.info(
            "Quadratic beat linear on %.0f%% of stints (leave-one-out RMSE)",
            100 * linearity["QuadraticWins"].mean(),
        )

    # The estimated fuel coefficient is a headline result, not a footnote: every
    # compound offset in the study is a residual left over after fuel is removed,
    # so a circuit where the fitted coefficient disagrees with the assumed one is
    # a circuit where the assumed one was quietly doing the talking.
    if not joint_fuel.empty:
        log.info("Fuel effect estimated from the laps (assumed %.3f s/kg):", a.fuel_effect_s_per_kg)
        for _, row in joint_fuel.iterrows():
            log.info(
                "  %-14s %.4f s/kg   [%.4f, %.4f]%s",
                row["Circuit"], row["ImpliedSecPerKg"],
                row["FuelCILow"] / a.fuel_start_kg, row["FuelCIHigh"] / a.fuel_start_kg,
                "" if row["DesignFullRank"] else "   (design rank-deficient)",
            )

    # Does the tyre lose more late in a stint than the straight line charges for?
    # If so the optimiser under-costs long stints and will recommend stopping too
    # late, which is exactly the direction of this study's largest disagreement
    # with what teams actually did.
    if not late_summary.empty:
        log.info("Late-stint loss beyond the linear model (final third of long stints):")
        for _, row in late_summary.iterrows():
            log.info(
                "  %-14s %+.3f s/lap median over %d stints, %.0f%% accelerating",
                row["Circuit"], row["MedianExtraSeconds"], row["NStints"],
                100 * row["ShareAccelerating"],
            )

    if not curvature.empty:
        applied = curvature[curvature["Usable"]]
        log.info(
            "Curvature measured on %d of %d circuit/compound cells (applied to "
            "strategy: %s - see Assumptions.apply_curvature):",
            len(applied), len(curvature), "yes" if a.apply_curvature else "no",
        )
        for _, row in applied.iterrows():
            log.info(
                "  %-14s %-7s %+.5f s/lap^2  [%.5f, %.5f]",
                row["Circuit"], row["Compound"], row["CurvatureSecPerLap2"],
                row["CurvatureCILow"], row["CurvatureCIHigh"],
            )

    disagreeing = (
        heterogeneity[heterogeneity["SeasonsDisagree"]] if not heterogeneity.empty else pd.DataFrame()
    )
    if not disagreeing.empty:
        log.warning("Cells where the seasons disagree - pooling may mix different rubber:")
        for _, row in disagreeing.iterrows():
            log.warning(
                "  %-14s %-7s %s: %.3f to %.3f s/lap (spread %.3f)",
                row["Circuit"], row["Compound"], row["Seasons"],
                row["MinDeg"], row["MaxDeg"], row["SpreadSecPerLap"],
            )

    excluded = pooled[~pooled["Usable"]] if not pooled.empty else pd.DataFrame()
    if not excluded.empty:
        log.warning("Degradation cells excluded from the optimiser:")
        for _, row in excluded.iterrows():
            log.warning("  %-14s %-7s %s", row["Circuit"], row["Compound"], row["ExcludedReason"])

    unidentified = (
        identifiability[~identifiability["OffsetsIdentified"]]
        if not identifiability.empty
        else pd.DataFrame()
    )
    if not unidentified.empty:
        log.warning("Circuits where compound offsets are not identified by the data:")
        for _, row in unidentified.iterrows():
            log.warning(
                "  %-14s minimum fuel-load overlap %.2f - offsets reported but not trustworthy",
                row["Circuit"], row["MinFuelOverlap"],
            )

    # ---- Optimisation ----------------------------------------------------
    usable = pooled[pooled["Usable"]] if not pooled.empty else pd.DataFrame()
    total_laps = raw.groupby("Circuit")["TotalLaps"].max().to_dict()
    # Several seasons are pooled for tyre behaviour, but the pit lane belongs to
    # one race: use the target season's measurement, falling back to the median
    # across seasons where that race is missing.
    if not pit_loss_df.empty:
        target = pit_loss_df[pit_loss_df["Season"] == TARGET_SEASON]
        pit_loss = pit_loss_df.groupby("Circuit")["PitLossMedian"].median().to_dict()
        pit_loss.update(target.set_index("Circuit")["PitLossMedian"].to_dict())
    else:
        pit_loss = {}
    identified = (
        identifiability.set_index("Circuit")["OffsetsIdentified"].to_dict()
        if not identifiability.empty
        else {}
    )
    total_laps = raw[raw["Season"] == TARGET_SEASON].groupby("Circuit")["TotalLaps"].max().to_dict()
    seasons = {c: TARGET_SEASON for c in total_laps}

    summaries, all_plans = [], {}
    for circuit in sorted(usable["Circuit"].unique()) if not usable.empty else []:
        deg = usable[usable["Circuit"] == circuit].set_index("Compound")["DegSecPerLap"].to_dict()

        # Offsets come from the joint model, which estimates fuel rather than
        # assuming it. The naive offsets are written to disk for comparison but
        # are never fed to the optimiser.
        offs = (
            joint_compounds[joint_compounds["Circuit"] == circuit]
            .set_index("Compound")["OffsetSec"]
            .to_dict()
        )
        shared = {c: deg[c] for c in deg if c in offs}
        if len(shared) < 2:
            log.warning("%s: fewer than two usable compounds, skipping", circuit)
            continue

        if circuit not in total_laps:
            log.warning("%s: not raced in the target season, skipping", circuit)
            continue

        loss = pit_loss.get(circuit)
        if loss is None:
            log.warning("%s: no pit-loss estimate, skipping", circuit)
            continue

        caps = (
            stint_limits[stint_limits["Circuit"] == circuit]
            .set_index("Compound")["MaxAllowedStintLaps"]
            .to_dict()
        )

        curve = (
            curvature[curvature["Circuit"] == circuit]
            .set_index("Compound")["AppliedCurvature"]
            .to_dict()
            if not curvature.empty and a.apply_curvature
            else {}
        )

        laps_total = int(total_laps[circuit])
        summary = optimal_summary(
            circuit, laps_total, shared, offs, float(loss),
            max_stint=caps, curvature=curve, a=a,
        )
        summary["Season"] = seasons.get(circuit)
        summary["OffsetsIdentified"] = bool(identified.get(circuit, False))
        summaries.append(summary)
        all_plans[circuit] = optimise_one_stop(
            laps_total, shared, offs, float(loss), max_stint=caps,
            undercut_gain_s=a.undercut_gain_s, curvature=curve,
        )

    summary_df = pd.DataFrame(summaries)
    actual = actual_stop_laps(raw)
    comparison = compare_to_actual(summary_df, actual) if not summary_df.empty else pd.DataFrame()

    summary_df.to_csv(f"{outdir}/strategy_summary.csv", index=False)
    if not comparison.empty:
        comparison.to_csv(f"{outdir}/model_vs_actual.csv", index=False)

    # ---- Out-of-sample validation ---------------------------------------
    # The only check in the project that is neither synthetic nor in-sample.
    seasons_present = sorted(raw["Season"].unique())
    backtest = rolling_backtest(raw, seasons_present, a) if len(seasons_present) > 1 else pd.DataFrame()
    backtest_summary = summarise_backtest(backtest)
    backtest.to_csv(f"{outdir}/backtest.csv", index=False)
    backtest_summary.to_csv(f"{outdir}/backtest_summary.csv", index=False)

    if not backtest_summary.empty:
        log.info("Out-of-sample backtest (model never sees the season it predicts):")
        for _, row in backtest_summary.iterrows():
            log.info(
                "  train %-9s -> %d%s: first-stop MAE %.1f laps  (baselines: "
                "last season %.1f, half distance %.1f)  bias %+.1f  stop count %.0f%% correct",
                row["TrainSeasons"], row["TestSeason"],
                " dry only" if row["DryRacesOnly"] else "         ", row["ModelMAELaps"],
                row["PersistenceMAELaps"], row["BaselineMAELaps"],
                row["MeanSignedErrorLaps"], 100 * row["StopCountAccuracy"],
            )

    # ---- Figures ---------------------------------------------------------
    if not usable.empty:
        plots.plot_degradation_by_circuit(usable, outdir)
    if not joint_fuel.empty:
        plots.plot_fuel_estimates(joint_fuel, a, outdir)
    if not backtest_summary.empty:
        plots.plot_backtest(backtest_summary, outdir)
    if not stint_fits.empty and not usable.empty:
        headline = usable.groupby("Circuit")["NLaps"].sum().idxmax()
        plots.plot_stint_evidence(laps, stint_fits, headline, outdir)
        if headline in all_plans and not all_plans[headline].empty:
            actual_stop = None
            if not actual.empty and headline in set(actual["Circuit"]):
                actual_stop = float(
                    actual[actual["Circuit"] == headline]["MedianFirstStopLap"].iloc[0]
                )
            plots.plot_stop_lap_curve(all_plans[headline], headline, actual_stop, outdir)
    if not comparison.empty:
        plots.plot_model_vs_actual(comparison, outdir)

    log.info("Wrote tables and figures to %s/", outdir)
    return {
        "laps": laps,
        "audit": audit,
        "stint_fits": stint_fits,
        "degradation": pooled,
        "offsets_naive": naive_offsets,
        "offsets_joint": joint_compounds,
        "fuel": joint_fuel,
        "identifiability": identifiability,
        "stint_limits": stint_limits,
        "pit_loss": pit_loss_df,
        "linearity": linearity,
        "late_stint_penalty": late_penalty,
        "late_stint_summary": late_summary,
        "summary": summary_df,
        "comparison": comparison,
        "curvature": curvature,
        "heterogeneity": heterogeneity,
        "backtest": backtest,
        "backtest_summary": backtest_summary,
    }


def demo(outdir: str = f"{OUTPUT_DIR}/demo") -> dict:
    """Run the pipeline on simulated races with known ground truth."""
    from src.synthetic import TRUE_DEG, TRUE_FUEL_EFFECT, simulate_race

    frames = [
        simulate_race(circuit=name, total_laps=laps, seed=seed)
        for name, laps, seed in [
            ("Synthetic Park", 57, 7),
            ("Synthetic Bay", 63, 11),
            ("Synthetic Ring", 44, 23),
        ]
    ]
    raw = pd.concat(frames, ignore_index=True)
    results = analyse(raw, outdir=outdir)

    log.info("Ground-truth check (simulated degradation, s/lap):")
    for _, row in results["degradation"].iterrows():
        truth = TRUE_DEG[row["Compound"]]
        log.info(
            "  %-16s %-7s estimated %.4f  true %.4f  error %+.4f  CI [%.4f, %.4f]%s",
            row["Circuit"], row["Compound"], row["DegSecPerLap"], truth,
            row["DegSecPerLap"] - truth, row["DegCILow"], row["DegCIHigh"],
            "" if row["Usable"] else f"  EXCLUDED: {row['ExcludedReason']}",
        )

    if not results["fuel"].empty:
        log.info("Ground-truth check (fuel effect, true %.4f s/kg):", TRUE_FUEL_EFFECT)
        for _, row in results["fuel"].iterrows():
            log.info(
                "  %-16s estimated %.4f  error %+.4f",
                row["Circuit"], row["ImpliedSecPerKg"],
                row["ImpliedSecPerKg"] - TRUE_FUEL_EFFECT,
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["fetch", "analyse", "all", "demo"])
    args = parser.parse_args()

    if args.stage == "demo":
        demo()
        return 0

    if args.stage in ("fetch", "all"):
        raw = fetch()
    else:
        raw = load_raw()
        log.info("Loaded %d cached laps from %s/", len(raw), DATA_DIR)

    if args.stage in ("analyse", "all"):
        analyse(raw)

    return 0


if __name__ == "__main__":
    sys.exit(main())

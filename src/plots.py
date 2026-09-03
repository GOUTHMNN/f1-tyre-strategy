"""Figures for the write-up.

Every chart is rendered twice, once for a light background and once for a dark
one, so the README stays readable in either GitHub theme. Charts are static PNGs
because that is what renders inside a README; the interaction layer an HTML
chart would carry is replaced here by direct labelling, so no value has to be
recovered by squinting at an axis.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import DRY_COMPOUNDS, OUTPUT_DIR, THEMES  # noqa: E402

FIG_DPI = 160


def _style(ax, theme: dict, xlabel: str = "", ylabel: str = "", title: str = "", subtitle: str = ""):
    """Recessive axes, horizontal-only grid, no chartjunk."""
    ax.set_facecolor(theme["surface"])
    ax.figure.set_facecolor(theme["surface"])

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
        ax.spines[side].set_linewidth(1.0)

    ax.grid(axis="y", color=theme["grid"], linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)

    if xlabel:
        ax.set_xlabel(xlabel, color=theme["muted"], fontsize=10, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=theme["muted"], fontsize=10, labelpad=8)
    if title:
        ax.set_title("", pad=0)
        ax.figure.text(
            0.012, 0.965, title, ha="left", va="top",
            color=theme["text"], fontsize=13, fontweight="bold",
        )
    if subtitle:
        ax.figure.text(
            0.012, 0.905, subtitle, ha="left", va="top",
            color=theme["muted"], fontsize=9.5,
        )


def _save(fig, name: str, mode: str, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}-{mode}.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _both_modes(draw_fn, name: str, outdir: str = OUTPUT_DIR) -> list[str]:
    return [_save(draw_fn(THEMES[m]), name, m, outdir) for m in ("light", "dark")]


# ---------------------------------------------------------------------------

def plot_degradation_by_circuit(pooled: pd.DataFrame, outdir: str = OUTPUT_DIR) -> list[str]:
    """Dot plot of degradation rate with confidence intervals.

    A dot plot rather than bars: these are estimates with uncertainty, not
    counts, and bars imply a meaningful zero baseline and invite the eye to
    compare areas. The interval is the point of the chart, so it gets drawn
    first and the marker sits on top of it.
    """

    def draw(theme: dict):
        circuits = sorted(pooled["Circuit"].unique())
        compounds = [c for c in DRY_COMPOUNDS if c in set(pooled["Compound"])]

        fig, ax = plt.subplots(figsize=(9, 0.34 * len(circuits) * len(compounds) + 2.0))
        offsets = np.linspace(-0.3, 0.3, max(len(compounds), 1))

        for ci, compound in enumerate(compounds):
            sub = pooled[pooled["Compound"] == compound].set_index("Circuit")
            ys, xs, los, his = [], [], [], []
            for i, circuit in enumerate(circuits):
                if circuit not in sub.index:
                    continue
                row = sub.loc[circuit]
                ys.append(i + offsets[ci])
                xs.append(row["DegSecPerLap"])
                los.append(row["DegCILow"])
                his.append(row["DegCIHigh"])

            if not ys:
                continue

            for y, lo, hi in zip(ys, los, his):
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.plot([lo, hi], [y, y], color=theme[compound], linewidth=2.0,
                            alpha=0.45, solid_capstyle="round", zorder=2)

            ax.scatter(xs, ys, s=64, color=theme[compound], label=compound,
                       zorder=3, edgecolor=theme["surface"], linewidth=1.6)

            # Direct labels: colour is never the only way to read a value.
            for x, y in zip(xs, ys):
                ax.annotate(f"{x:.3f}", (x, y), xytext=(9, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=theme["muted"])

        ax.set_yticks(range(len(circuits)))
        ax.set_yticklabels(circuits, color=theme["text"], fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=theme["grid"], linewidth=0.9)
        ax.axvline(0, color=theme["grid"], linewidth=1.0)

        _style(
            ax, theme,
            xlabel="Degradation (seconds lost per lap of tyre life, fuel-corrected)",
            title="How fast each compound falls away",
            subtitle="Point estimate with 95% bootstrap interval. Wider bars mean the circuit's stints disagreed more.",
        )
        leg = ax.legend(frameon=False, fontsize=9.5, ncols=len(compounds), loc="lower left",
                        bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0)
        for text in leg.get_texts():
            text.set_color(theme["text"])
        fig.subplots_adjust(top=0.82)
        return fig

    return _both_modes(draw, "degradation-by-circuit", outdir)


def plot_stint_evidence(
    laps: pd.DataFrame, stint_fits: pd.DataFrame, circuit: str, outdir: str = OUTPUT_DIR
) -> list[str]:
    """The raw laps behind one circuit's numbers, with the fitted lines on top.

    Showing the scatter is a matter of honesty: a reader can see for themselves
    how noisy a race lap time is, and judge whether a straight line through it
    is a fair summary or wishful thinking.
    """

    def draw(theme: dict):
        sub = laps[laps["Circuit"] == circuit]
        fits = stint_fits[stint_fits["Circuit"] == circuit]

        fig, ax = plt.subplots(figsize=(9, 5.2))
        for compound in DRY_COMPOUNDS:
            comp_laps = sub[sub["Compound"] == compound]
            if comp_laps.empty:
                continue

            # Centre each stint on its own baseline so cars of different pace can
            # share one axis: the question is the slope, not who is quicker.
            centred = []
            for _, stint in comp_laps.groupby(["Driver", "Stint"]):
                y = stint["LapTimeFuelCorrected"]
                centred.append(pd.DataFrame({"TyreLife": stint["TyreLife"], "Delta": y - y.min()}))
            if not centred:
                continue
            pooled_pts = pd.concat(centred)

            ax.scatter(pooled_pts["TyreLife"], pooled_pts["Delta"], s=13, alpha=0.30,
                       color=theme[compound], linewidth=0, zorder=2)

            comp_fits = fits[fits["Compound"] == compound]
            if comp_fits.empty:
                continue
            slope = np.average(comp_fits["SlopeSecPerLap"], weights=comp_fits["NLaps"])
            xs = np.linspace(pooled_pts["TyreLife"].min(), pooled_pts["TyreLife"].max(), 50)
            ax.plot(xs, slope * (xs - xs.min()), color=theme[compound], linewidth=2.4,
                    zorder=3, label=f"{compound}  {slope:.3f} s/lap")

        _style(
            ax, theme,
            xlabel="Tyre age (laps)",
            ylabel="Lap time lost vs the stint's best lap (s)",
            title=f"The evidence behind the fit — {circuit}",
            subtitle="Every green-flag racing lap, each stint centred on its own best. Lines are the pooled degradation slope.",
        )
        leg = ax.legend(frameon=False, loc="upper left", fontsize=9.5)
        for text in leg.get_texts():
            text.set_color(theme["text"])
        fig.subplots_adjust(top=0.85)
        return fig

    return _both_modes(draw, f"stint-evidence-{circuit.lower().replace(' ', '-')}", outdir)


def plot_stop_lap_curve(
    plans: pd.DataFrame, circuit: str, actual_stop: float | None = None, outdir: str = OUTPUT_DIR
) -> list[str]:
    """The decision chart: race time against stop lap, per compound plan.

    This is the figure the whole project exists to produce. The flat bottom of
    each curve matters more than its minimum: it shows how much freedom the team
    actually has, and a curve that is flat for fifteen laps means the precise
    stop lap was never the real question.
    """

    def draw(theme: dict):
        fig, ax = plt.subplots(figsize=(9, 5.2))
        best_overall = plans["RelativeRaceTime"].min()

        top_plans = (
            plans.groupby("Plan")["RelativeRaceTime"].min().nsmallest(3).index.tolist()
        )

        # The winning plan is the finding, so it gets the only strong colour and
        # the runners-up recede. Compound hues are deliberately not reused here:
        # a curve represents a two-compound plan, so colouring it as though it
        # were a single compound would misread at a glance.
        emphasis = [theme["SOFT"], theme["muted"], theme["grid"]]
        widths = [2.6, 1.8, 1.8]

        # Cap the y-axis: once a plan is 20s off, its exact cost is irrelevant
        # and letting it set the scale would flatten the region that matters.
        y_cap = 20.0

        for rank, (plan, colour, lw) in enumerate(zip(top_plans, emphasis, widths)):
            curve = plans[plans["Plan"] == plan].sort_values("StopLap")
            y = curve["RelativeRaceTime"] - best_overall
            ax.plot(curve["StopLap"], y, color=colour, linewidth=lw, label=plan,
                    zorder=4 - rank)

            best = curve.loc[curve["RelativeRaceTime"].idxmin()]
            by = best["RelativeRaceTime"] - best_overall
            if by > y_cap:
                continue
            ax.scatter([best["StopLap"]], [by], s=80, color=colour,
                       edgecolor=theme["surface"], linewidth=1.8, zorder=5)
            # Stagger the labels so adjacent optima do not overprint.
            ax.annotate(
                f"{plan}\nlap {int(best['StopLap'])}",
                (best["StopLap"], by),
                xytext=(0, 14 + 26 * rank), textcoords="offset points", ha="center",
                fontsize=9, color=theme["text"] if rank == 0 else theme["muted"],
                fontweight="bold" if rank == 0 else "normal",
            )

        # The window in which the call barely matters.
        winner = plans[plans["Plan"] == top_plans[0]]
        window = winner[winner["RelativeRaceTime"] - best_overall <= 1.0]["StopLap"]
        if len(window):
            ax.axvspan(window.min(), window.max(), color=theme["grid"], alpha=0.5, zorder=0)
            ax.annotate(
                f"within 1s of optimal:\nlaps {int(window.min())}–{int(window.max())}",
                (float(np.mean([window.min(), window.max()])), y_cap * 0.9),
                ha="center", va="top", fontsize=9, color=theme["muted"],
            )

        if actual_stop is not None and np.isfinite(actual_stop):
            ax.axvline(actual_stop, color=theme["accent"], linewidth=1.4,
                       linestyle=(0, (4, 3)), zorder=2)
            ax.annotate(
                f"teams actually\nstopped: lap {actual_stop:.0f}",
                (actual_stop, y_cap * 0.42), xytext=(8, 0),
                textcoords="offset points", fontsize=9, color=theme["text"],
            )

        ax.set_ylim(-0.8, y_cap)
        _style(
            ax, theme,
            xlabel="Lap of the pit stop",
            ylabel="Race time lost vs the best plan (s)",
            title=f"When to stop — {circuit}",
            subtitle="Modelled one-stop race time. Lower is better; the shaded band is where the choice costs under a second.",
        )
        leg = ax.legend(frameon=False, fontsize=9.5, ncols=3, loc="lower left",
                        bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0)
        for text in leg.get_texts():
            text.set_color(theme["text"])
        fig.subplots_adjust(top=0.80)
        return fig

    return _both_modes(draw, f"stop-lap-{circuit.lower().replace(' ', '-')}", outdir)


def plot_model_vs_actual(comparison: pd.DataFrame, outdir: str = OUTPUT_DIR) -> list[str]:
    """Model recommendation against what the teams did, per circuit.

    A dumbbell chart, because the quantity of interest is the gap between two
    paired values. Where the gap is large, the model is missing something real -
    traffic, the undercut, or a tyre allocation it knows nothing about.
    """

    def draw(theme: dict):
        df = comparison.dropna(subset=["MedianFirstStopLap"]).sort_values("StopLapError")
        fig, ax = plt.subplots(figsize=(9, 0.46 * len(df) + 2.2))

        for i, (_, row) in enumerate(df.iterrows()):
            ax.plot([row["MedianFirstStopLap"], row["BestOneStopLap"]], [i, i],
                    color=theme["grid"], linewidth=2.6, solid_capstyle="round", zorder=2)
            ax.scatter([row["MedianFirstStopLap"]], [i], s=76, color=theme["MEDIUM"],
                       edgecolor=theme["surface"], linewidth=1.6, zorder=3,
                       label="Teams (median)" if i == 0 else None)
            ax.scatter([row["BestOneStopLap"]], [i], s=76, color=theme["SOFT"],
                       edgecolor=theme["surface"], linewidth=1.6, zorder=3,
                       label="Model optimum" if i == 0 else None)
            ax.annotate(f"{row['StopLapError']:+.0f} laps",
                        (max(row["MedianFirstStopLap"], row["BestOneStopLap"]), i),
                        xytext=(11, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=theme["muted"])

        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["Circuit"], color=theme["text"], fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=theme["grid"], linewidth=0.9)
        # Room on the right so the delta labels are not clipped.
        ax.margins(x=0.14, y=0.22)

        _style(
            ax, theme,
            xlabel="Lap of first stop",
            title="Model against reality",
            subtitle="Where the two disagree, the model is missing something — traffic, the undercut, or tyre allocation.",
        )
        leg = ax.legend(frameon=False, fontsize=9.5, ncols=2, loc="lower left",
                        bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0)
        for text in leg.get_texts():
            text.set_color(theme["text"])
        fig.subplots_adjust(top=0.78)
        return fig

    return _both_modes(draw, "model-vs-actual", outdir)


def plot_fuel_estimates(fuel: pd.DataFrame, a, outdir: str = OUTPUT_DIR) -> list[str]:
    """Fitted fuel coefficient per circuit against the value the study assumed.

    This chart is the audit of an assumption. Every compound offset in the
    project is what remains after the fuel effect has been taken out, so if the
    assumed coefficient is wrong, the error does not vanish - it reappears
    wearing a tyre compound's name, because teams run softs on a heavy car and
    hards on a light one.

    The assumed value is drawn as a reference line with the commonly quoted
    0.030-0.040 s/kg band behind it. A circuit whose interval clears that band
    is telling you the correction applied to it was not the right size.
    """

    def draw(theme: dict):
        df = fuel.sort_values("ImpliedSecPerKg")
        fig, ax = plt.subplots(figsize=(9, 0.46 * len(df) + 2.4))

        ax.axvspan(0.030, 0.040, color=theme["grid"], alpha=0.7, zorder=0)
        ax.axvline(a.fuel_effect_s_per_kg, color=theme["accent"], linewidth=1.4,
                   linestyle=(0, (4, 3)), zorder=2)

        for i, (_, row) in enumerate(df.iterrows()):
            lo = row["FuelCILow"] / a.fuel_start_kg
            hi = row["FuelCIHigh"] / a.fuel_start_kg
            inside = np.isfinite(lo) and np.isfinite(hi) and lo <= a.fuel_effect_s_per_kg <= hi
            colour = theme["HARD"] if inside else theme["SOFT"]

            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [i, i], color=colour, linewidth=2.2, alpha=0.45,
                        solid_capstyle="round", zorder=2)
            ax.scatter([row["ImpliedSecPerKg"]], [i], s=70, color=colour,
                       edgecolor=theme["surface"], linewidth=1.6, zorder=3)
            ax.annotate(f"{row['ImpliedSecPerKg']:.4f}", (row["ImpliedSecPerKg"], i),
                        xytext=(10, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=theme["muted"])

        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["Circuit"], color=theme["text"], fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=theme["grid"], linewidth=0.9)
        ax.margins(x=0.16, y=0.20)

        ax.annotate(
            f"assumed {a.fuel_effect_s_per_kg:.3f}",
            (a.fuel_effect_s_per_kg, -0.62), ha="center", fontsize=9, color=theme["text"],
        )

        _style(
            ax, theme,
            xlabel="Fuel effect (seconds per kg), estimated jointly with tyre degradation",
            title="Auditing the assumption the offsets rest on",
            subtitle="Green intervals contain the assumed value; orange ones do not, and their circuit's compound offsets absorb the difference.",
        )
        fig.subplots_adjust(top=0.80)
        return fig

    return _both_modes(draw, "fuel-effect-estimated", outdir)

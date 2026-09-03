"""Central configuration and modelling assumptions.

Every assumption that could change a conclusion lives here, with a source or a
justification, so it can be found, argued with, and sensitivity-tested. Numbers
buried in code are numbers nobody checks.
"""

from dataclasses import dataclass

SEASON = 2024  # default single-season entry point

# Races are identified by their exact official event name, never by a loose one.
#
# FastF1 resolves session names with a fuzzy string match. Asking it for
# "Great Britain" in 2024 returns the *Austrian* Grand Prix - a different circuit,
# a different tyre allocation, 71 laps instead of 52 - and says so only at INFO
# level, so the mislabelled data flows downstream unchallenged. "United States"
# is worse still: three 2024 rounds carry that country (Miami, Austin, Las Vegas).
#
# Round numbers are unambiguous within a season but move between them (the
# British Grand Prix was round 10 in 2022 and round 12 in 2024), so the round is
# resolved from the schedule by *exact* name match at load time and the returned
# event is verified again before a single lap is used.
@dataclass(frozen=True)
class Circuit:
    label: str        # short name used in tables and figures
    event_name: str   # exact official EventName, matched exactly against the schedule


# Six circuits chosen to span the degradation spectrum. The original rationale
# read "two high-deg abrasive tracks (Bahrain, Barcelona), two medium
# (Silverstone, Austin), one low-deg but traffic-limited (Hungary), one
# short-lap high-lap-count (Zandvoort)" - and the measurements only partly bear
# that out. Bahrain is indeed the most severe and Austin the mildest, but
# Hungary degrades slightly *harder* than Barcelona rather than being the
# low-deg outlier the selection assumed. The spread is what the selection
# needed, and it delivered that; the prior about individual circuits is left
# here as a prior the data corrected.
CIRCUITS = [
    Circuit("Bahrain",       "Bahrain Grand Prix"),
    Circuit("Spain",         "Spanish Grand Prix"),
    Circuit("Great Britain", "British Grand Prix"),
    Circuit("Hungary",       "Hungarian Grand Prix"),
    Circuit("Netherlands",   "Dutch Grand Prix"),
    Circuit("United States", "United States Grand Prix"),
]

# Three seasons of the same six circuits.
#
# One season cannot measure compound pace. Within a single race a team runs each
# compound in one phase of the race, so compound and fuel load are the same
# variable and the offset is not identified - five of six circuits failed that
# test on 2024 alone. Different years bring different strategies to the same
# track, and it is that variation across seasons which finally separates the two.
#
# The cost is that a "SOFT" is not the same rubber every year: Pirelli allocates
# C1-C5 per event and the mapping moves. `model.season_heterogeneity` reports
# where the per-season estimates disagree enough that pooling them is unsafe.
SEASONS = [2022, 2023, 2024]

# The season strategy is reported for, and the one the backtest predicts.
TARGET_SEASON = 2024  # default single-season entry point

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


@dataclass(frozen=True)
class Assumptions:
    """Physical and procedural assumptions of the model.

    Defaults are public-domain regulation figures and widely published estimates,
    not team data. They are approximations: `sensitivity.py` re-runs the whole
    pipeline across plausible ranges so the reader can see which conclusions
    survive and which do not.
    """

    # ---- Fuel -------------------------------------------------------------
    # FIA technical regulations cap the race fuel load at 110 kg. Teams start
    # below the cap, but the exact number is not public.
    fuel_start_kg: float = 100.0

    # Time cost of carrying one extra kg. Commonly quoted at 0.03-0.04 s/kg for
    # modern ground-effect cars; the midpoint is used here.
    #
    # This is only a *prior*. The joint model in `model.fit_joint_model`
    # estimates the coefficient from the laps themselves, because assuming it
    # and then reading compound offsets out of the residual makes those offsets
    # a measurement of this guess rather than of the tyres.
    fuel_effect_s_per_kg: float = 0.035

    # ---- Cleaning ---------------------------------------------------------
    # Laps discarded at the start of a stint *after* the out-lap. Tyre life 1 is
    # the out-lap (or the standing-start lap) and is always dropped; a new tyre
    # then needs a lap or so more to reach working temperature.
    stint_warmup_laps: int = 1

    # A stint shorter than this cannot support a slope estimate worth having.
    min_stint_laps: int = 6

    # Outlier rejection within a stint, in median absolute deviations. Traffic,
    # lock-ups and lifts produce slow laps that are not degradation.
    outlier_mad_threshold: float = 3.0

    # ---- Pooling guards ---------------------------------------------------
    # A circuit/compound cell needs this much evidence before its degradation
    # rate is allowed to reach the optimiser. Hungary 2024 ran exactly one
    # 12-lap soft stint; it fitted a *negative* slope, and the optimiser
    # cheerfully recommended 65 laps on softs to harvest the free time.
    min_stints_for_pooling: int = 3
    min_laps_for_pooling: int = 30

    # A tyre that gets faster with age is a fitting artefact, not a finding.
    reject_negative_degradation: bool = True

    # ---- Curvature --------------------------------------------------------
    # A stint needs this many laps before a quadratic is worth fitting to it,
    # and a circuit/compound cell this many stints before the pooled curvature
    # is allowed to price a strategy. Curvature is a second derivative taken
    # from ~20 noisy laps and deserves more evidence than a slope, not less.
    min_laps_for_curvature: int = 12
    min_stints_for_curvature: int = 8

    # Whether the measured curvature is allowed to price strategies. It is not,
    # and that decision was made by the backtest rather than by taste.
    #
    # The curvature is real: seven circuit/compound cells show it with bootstrap
    # intervals clear of zero, and `late_stint_penalty` finds accelerating wear
    # independently. It is also well motivated - pricing a stint linearly says a
    # tyre's twentieth lap costs what its second did, which is false, and
    # under-charging long stints is exactly what makes an optimiser stop late.
    #
    # It nonetheless makes held-out predictions worse: 8.0 laps of error against
    # 6.0 without it on 2024, and 9.4 against 7.0 on dry races only. A second
    # derivative estimated from twenty noisy laps is fitted to the tail of each
    # stint, and the tail is where traffic and fuel-saving live, so it appears to
    # be learning the end of a stint rather than the tyre.
    #
    # Kept, measured and reported; not applied. An elaboration that improves the
    # story and worsens the predictions is exactly the kind a model should be
    # made to earn its way past, and this one did not.
    apply_curvature: bool = False

    # ---- Wet races --------------------------------------------------------
    # Share of a race's laps on wet or intermediate tyres above which it is
    # treated as weather-affected. This is a dry-tyre strategy model, and a race
    # decided by a rain shower is not evidence against it.
    #
    # The threshold is set from the shape of the data rather than tuned: across
    # eighteen races the wet share is either 0.0% or 24.4%, nothing in between,
    # so any cutoff in that gap gives the same answer and none of them can be
    # nudged to flatter a result. Backtest scores are reported both ways
    # regardless, because dropping inconvenient races on the quiet is how an
    # honest evaluation becomes a dishonest one.
    max_wet_lap_share: float = 0.05

    # ---- Strategy ---------------------------------------------------------
    # Cap stint length at the longest actually observed for that compound and
    # circuit, plus this margin. Linear degradation extrapolated over 60 laps is
    # a fantasy: real tyres reach a cliff and teams know where it is, so the
    # observed maximum is the best public evidence of the limit.
    max_stint_margin_laps: int = 3

    # Track-position value of completing a stop before a closely matched rival.
    # Zero by default because this model cannot see the cars around it; exposed
    # so the sensitivity sweep can show how much of the model-vs-reality gap an
    # undercut of a given size would explain.
    undercut_gain_s: float = 0.0

    # ---- Inference --------------------------------------------------------
    # Bootstrap resamples for confidence intervals on degradation slopes.
    n_bootstrap: int = 2000

    # Cluster-bootstrap resamples for the joint model. Fewer, because each draw
    # refits a whole robust regression rather than averaging scalars.
    n_bootstrap_joint: int = 400

    # Huber threshold for the joint fit, in seconds. Beyond roughly this far
    # from the fit a lap is treated as traffic and down-weighted rather than
    # allowed to pull the line.
    huber_delta_s: float = 0.6

    # A compound offset is only reported when the compound's fuel-load range
    # overlaps the reference compound's by at least this fraction. Below it,
    # compound and fuel load are collinear and the offset is not identified by
    # the data at all - see `model.identifiability_report`.
    min_fuel_overlap: float = 0.15

    random_seed: int = 42


ASSUMPTIONS = Assumptions()

# Plot styling kept in one place so every figure in the repo looks like it came
# from the same study.
#
# The obvious choice would be the sidewall colours - red, yellow, white - but
# those fail an accessibility check: red against yellow is one of the hardest
# pairs for red-green colour blindness, and white or grey carries no chroma at
# all on a pale background. These three hues were checked for colour-vision
# separation and contrast instead, and compound identity is additionally carried
# by the legend and by direct labels, so colour is never the only cue.
THEMES = {
    "light": {
        "SOFT": "#eb6834",
        "MEDIUM": "#2a78d6",
        "HARD": "#1baf7a",
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e3e2df",
        "accent": "#0b0b0b",
    },
    "dark": {
        "SOFT": "#d95926",
        "MEDIUM": "#3987e5",
        "HARD": "#199e70",
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3a38",
        "accent": "#ffffff",
    },
}

CACHE_DIR = "cache"
OUTPUT_DIR = "outputs"
DATA_DIR = "data"

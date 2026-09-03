"""Central configuration and modelling assumptions.

Every assumption that could change a conclusion lives here, with a source or a
justification, so it can be found, argued with, and sensitivity-tested. Numbers
buried in code are numbers nobody checks.
"""

from dataclasses import dataclass

SEASON = 2024

# Races are identified by round number, never by name.
#
# FastF1 resolves session names with a fuzzy string match. Asking it for
# "Great Britain" in 2024 returns the *Austrian* Grand Prix - a different circuit,
# a different tyre allocation, 71 laps instead of 52 - and says so only at INFO
# level, so the mislabelled data flows downstream unchallenged. "United States"
# is worse still: three 2024 rounds carry that country (Miami, Austin, Las Vegas).
#
# Round numbers are unambiguous, and `data.load_race` asserts the event that
# comes back matches `expect_event` before a single lap is used.
@dataclass(frozen=True)
class Race:
    round_number: int
    label: str          # short name used in tables and figures
    expect_event: str   # official EventName, verified at load time


# Six circuits chosen to span the degradation spectrum. The original rationale
# read "two high-deg abrasive tracks (Bahrain, Barcelona), two medium
# (Silverstone, Austin), one low-deg but traffic-limited (Hungary), one
# short-lap high-lap-count (Zandvoort)" - and the measurements only partly bear
# that out. Bahrain is indeed the most severe (0.100 s/lap on hards) and Austin
# the mildest (0.021), but Hungary (0.080) degrades slightly *harder* than
# Barcelona (0.071) rather than being the low-deg outlier the selection assumed.
# The spread is what the selection needed, and it delivered that; the prior
# about individual circuits is left here as a prior that the data corrected.
RACES = [
    Race(1,  "Bahrain",       "Bahrain Grand Prix"),
    Race(10, "Spain",         "Spanish Grand Prix"),
    Race(12, "Great Britain", "British Grand Prix"),
    Race(13, "Hungary",       "Hungarian Grand Prix"),
    Race(15, "Netherlands",   "Dutch Grand Prix"),
    Race(19, "United States", "United States Grand Prix"),
]

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

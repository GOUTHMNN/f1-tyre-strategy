"""A simulated race with known ground truth, for testing the pipeline.

The F1 API is not always reachable, and more importantly a pipeline validated
only on real data can never be checked for correctness - nobody knows the true
degradation rate of a real tyre. Here the truth is set by hand, noise and
traffic and safety cars are injected on purpose, and the tests assert that the
estimator recovers the numbers it was given. If it cannot recover a known
answer from clean-ish synthetic data, it certainly cannot be trusted on Bahrain.

Two properties of this generator matter more than the rest, because both were
originally absent and their absence let a real bug pass every test:

* `fuel_effect` is a free parameter. The first version hard-coded it to exactly
  the value `config.Assumptions` assumes, which made the fuel correction perfect
  by construction. No test could then detect a wrong fuel coefficient - and a
  wrong fuel coefficient is precisely what corrupts compound offsets on real
  data.
* `strategy_mix` controls whether compounds are spread across race phases or
  perfectly aligned with them. The first version ran every driver's first stint
  on a soft or medium and every second stint on a hard, which is the fully
  confounded design. It scored well only because the fuel model was exact.

Together those two made the synthetic suite agree with the real pipeline about
everything except the one thing that was wrong with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRUE_DEG = {"SOFT": 0.085, "MEDIUM": 0.045, "HARD": 0.022}
TRUE_OFFSET = {"SOFT": 0.0, "MEDIUM": 0.45, "HARD": 0.95}
TRUE_PIT_LOSS = 21.0
TRUE_FUEL_EFFECT = 0.035
BASE_LAP = 92.0

# Compound sequences dealt round-robin to the field.
#
# "varied" deliberately runs every compound both early and late, so fuel load
# and compound identity are not the same variable and an estimator has some
# chance of telling them apart. Real races are rarely this generous, which is
# what `model.identifiability_report` exists to detect.
STRATEGY_MIXES = {
    "varied": [
        ("SOFT", "HARD"),
        ("HARD", "SOFT"),
        ("MEDIUM", "HARD"),
        ("HARD", "MEDIUM"),
        ("SOFT", "MEDIUM", "HARD"),
        ("HARD", "MEDIUM", "SOFT"),
        ("MEDIUM", "SOFT"),
        ("SOFT", "HARD", "MEDIUM"),
    ],
    # Every soft lap on a heavy car, every hard lap on a light one. The compound
    # column and the fuel column carry the same information, and no amount of
    # statistical care can separate them.
    "confounded": [
        ("SOFT", "HARD"),
    ],
}


def _plan_stints(
    compounds: tuple[str, ...], total_laps: int, rng: np.random.Generator
) -> list[tuple[str, int, int]]:
    """Split the race into stints for a given compound sequence."""
    n_stops = len(compounds) - 1
    if n_stops == 1:
        cuts = [int(rng.integers(int(total_laps * 0.30), int(total_laps * 0.62)))]
    else:
        first = int(rng.integers(int(total_laps * 0.18), int(total_laps * 0.36)))
        second = int(rng.integers(first + 8, int(total_laps * 0.78)))
        cuts = [first, second]

    plan, start = [], 1
    for compound, end in zip(compounds, cuts + [total_laps]):
        plan.append((compound, start, end))
        start = end + 1
    return plan


def simulate_race(
    circuit: str = "SyntheticPark",
    season: int = 2024,
    total_laps: int = 57,
    n_drivers: int = 16,
    fuel_start_kg: float = 100.0,
    fuel_effect: float = TRUE_FUEL_EFFECT,
    noise_sd: float = 0.28,
    traffic_rate: float = 0.06,
    strategy_mix: str = "varied",
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a raw-format lap table matching what `data.load_race` returns.

    `fuel_effect` is the *true* seconds-per-kg used to build the lap times. Pass
    something other than `TRUE_FUEL_EFFECT` to simulate a race whose fuel
    behaviour the pipeline's assumed coefficient gets wrong, and watch which
    estimators survive it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    sequences = STRATEGY_MIXES[strategy_mix]

    # A safety car window every race, to exercise the track-status filtering.
    sc_start = int(rng.integers(10, total_laps - 15))
    sc_laps = set(range(sc_start, sc_start + 4))

    for d in range(n_drivers):
        driver = f"D{d:02d}"
        team = f"Team{d // 2}"
        car_pace = rng.normal(0.0, 0.45)  # each car has its own baseline pace

        plan = _plan_stints(sequences[d % len(sequences)], total_laps, rng)

        for stint_no, (compound, start, end) in enumerate(plan, start=1):
            for tyre_age, lap in enumerate(range(start, end + 1), start=1):
                fuel_kg = fuel_start_kg * (1 - (lap - 1) / total_laps)
                lap_time = (
                    BASE_LAP
                    + car_pace
                    + TRUE_OFFSET[compound]
                    + TRUE_DEG[compound] * tyre_age
                    + fuel_kg * fuel_effect
                    + rng.normal(0.0, noise_sd)
                )

                # Traffic: occasional slow laps that must not bend the slope.
                if rng.random() < traffic_rate:
                    lap_time += rng.uniform(0.8, 3.5)

                is_in = lap == end and stint_no < len(plan)
                is_out = tyre_age == 1 and stint_no > 1

                if is_in:
                    lap_time += TRUE_PIT_LOSS * 0.45
                if is_out:
                    lap_time += TRUE_PIT_LOSS * 0.55

                status = "1"
                if lap in sc_laps:
                    status = "4"
                    lap_time += rng.uniform(18.0, 26.0)

                rows.append(
                    {
                        "Driver": driver,
                        "Team": team,
                        "LapNumber": float(lap),
                        "LapTimeSeconds": float(lap_time),
                        "Stint": float(stint_no),
                        "Compound": compound,
                        "TyreLife": float(tyre_age),
                        "FreshTyre": True,
                        "TrackStatus": status,
                        "IsAccurate": True,
                        "IsPitInLap": bool(is_in),
                        "IsPitOutLap": bool(is_out),
                        "Position": float(d + 1),
                        "Circuit": circuit,
                        "EventName": f"{circuit} Grand Prix",
                        "RoundNumber": 0,
                        "Season": season,
                        "TotalLaps": float(total_laps),
                    }
                )

    return pd.DataFrame(rows)

"""Loading race data from FastF1 and reducing it to a tidy lap table.

FastF1 caches raw API responses on disk, so the expensive network pull happens
once per session. This module turns each race into one flat DataFrame with the
columns the rest of the pipeline needs, and nothing else.
"""

from __future__ import annotations

import logging
import os
import warnings

import pandas as pd

from .config import CACHE_DIR, DATA_DIR, SEASON, SEASONS, Circuit

log = logging.getLogger(__name__)

LAP_COLUMNS = [
    "Driver",
    "Team",
    "LapNumber",
    "LapTimeSeconds",
    "Stint",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "TrackStatus",
    "IsAccurate",
    "IsPitInLap",
    "IsPitOutLap",
    "Position",
    "Circuit",
    "EventName",
    "RoundNumber",
    "Season",
    "TotalLaps",
]


class EventMismatch(RuntimeError):
    """The session that came back is not the session that was asked for."""


def _enable_cache(cache_dir: str = CACHE_DIR) -> None:
    import fastf1

    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def resolve_round(circuit: Circuit, season: int, cache_dir: str = CACHE_DIR) -> int:
    """Find the round number for a circuit in a season by exact name match.

    Exact, not fuzzy, and it insists on exactly one hit. FastF1's own matcher
    will cheerfully accept a near miss and hand back a different Grand Prix;
    requiring a unique exact match is what makes the substitution impossible
    rather than merely unlikely.
    """
    import fastf1

    _enable_cache(cache_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        schedule = fastf1.get_event_schedule(season)

    hits = schedule[schedule["EventName"] == circuit.event_name]
    if len(hits) != 1:
        raise EventMismatch(
            f"'{circuit.event_name}' matched {len(hits)} events in {season}; "
            "expected exactly one."
        )
    return int(hits["RoundNumber"].iloc[0])


def load_race(circuit: Circuit, season: int = SEASON, cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Load one race and return a tidy lap table.

    The round is resolved from the schedule by exact name, and the event that
    comes back is verified against the name that was asked for before any lap is
    returned. A silent substitution here is the most expensive kind of bug in
    the project: nothing downstream errors, every table fills in, and the entire
    analysis describes a circuit nobody meant to study.

    Raises EventMismatch if the wrong event comes back, and RuntimeError if the
    session loads but contains no laps - which is how a blocked network or an
    unavailable session usually presents itself.
    """
    import fastf1

    round_number = resolve_round(circuit, season, cache_dir)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        event = fastf1.get_event(season, round_number)
        actual_name = str(event["EventName"])

        if actual_name != circuit.event_name:
            raise EventMismatch(
                f"{season} round {round_number} is '{actual_name}', but the study "
                f"expects '{circuit.event_name}'."
            )

        session = event.get_session("R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)

    laps = session.laps
    if laps is None or len(laps) == 0:
        raise RuntimeError(
            f"No laps returned for {season} {actual_name}. If every session fails "
            "this way, the F1 timing API is unreachable from this machine."
        )

    df = pd.DataFrame(
        {
            "Driver": laps["Driver"].astype(str),
            "Team": laps["Team"].astype(str),
            "LapNumber": laps["LapNumber"].astype(float),
            "LapTimeSeconds": laps["LapTime"].dt.total_seconds(),
            "Stint": laps["Stint"].astype(float),
            "Compound": laps["Compound"].astype(str).str.upper(),
            "TyreLife": laps["TyreLife"].astype(float),
            "FreshTyre": laps["FreshTyre"],
            "TrackStatus": laps["TrackStatus"].astype(str),
            "IsAccurate": laps["IsAccurate"],
            "IsPitInLap": laps["PitInTime"].notna(),
            "IsPitOutLap": laps["PitOutTime"].notna(),
            "Position": laps["Position"].astype(float),
        }
    )

    # `Circuit` is the study's short label; `EventName` is what the API actually
    # returned. Keeping both means a mislabelling can never again hide, because
    # the two can be compared in the saved data long after the run.
    df["Circuit"] = circuit.label
    df["EventName"] = actual_name
    df["RoundNumber"] = round_number
    df["Season"] = season
    total_laps = getattr(session, "total_laps", None) or int(df["LapNumber"].max())
    df["TotalLaps"] = float(total_laps)

    return df[LAP_COLUMNS].reset_index(drop=True)


def load_races(
    circuits: list[Circuit],
    seasons: list[int] | None = None,
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """Load every circuit for every season, skipping races that fail to download.

    An EventMismatch is *not* skipped. A race that fails to download leaves the
    study with less data; a race that downloads the wrong circuit leaves it with
    a confident wrong answer, so that one stops everything.
    """
    seasons = list(seasons if seasons is not None else SEASONS)
    frames = []
    for season in seasons:
        for circuit in circuits:
            try:
                frame = load_race(circuit, season=season, cache_dir=cache_dir)
            except EventMismatch:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad race must not kill the study
                log.warning("Skipping %s %s: %s", season, circuit.label, exc)
                continue
            log.info(
                "Loaded %s %-14s -> %-24s %d laps",
                season, circuit.label, frame["EventName"].iloc[0], len(frame),
            )
            frames.append(frame)

    if not frames:
        raise RuntimeError("No races loaded successfully.")

    return pd.concat(frames, ignore_index=True)


def save_raw(df: pd.DataFrame, path: str = f"{DATA_DIR}/laps_raw.parquet") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:  # pyarrow not installed
        path = path.replace(".parquet", ".csv")
        df.to_csv(path, index=False)
    return path


def load_raw(path: str = f"{DATA_DIR}/laps_raw.parquet") -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    csv = path.replace(".parquet", ".csv")
    if os.path.exists(csv):
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No cached lap table at {path} or {csv}. Run the fetch step first.")

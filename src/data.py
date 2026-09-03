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

from .config import CACHE_DIR, DATA_DIR, SEASON, Race

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


def load_race(race: Race, season: int = SEASON, cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Load one race by round number and return a tidy lap table.

    The round number is resolved to an event and the event's official name is
    checked against `race.expect_event` before any lap is returned. A silent
    substitution here is the most expensive kind of bug in the project: nothing
    downstream errors, every table fills in, and the entire analysis describes a
    circuit nobody meant to study.

    Raises EventMismatch if the wrong event comes back, and RuntimeError if the
    session loads but contains no laps - which is how a blocked network or an
    unavailable session usually presents itself.
    """
    import fastf1

    _enable_cache(cache_dir)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        event = fastf1.get_event(season, race.round_number)
        actual_name = str(event["EventName"])

        if actual_name != race.expect_event:
            raise EventMismatch(
                f"{season} round {race.round_number} is '{actual_name}', but the "
                f"study expects '{race.expect_event}'. Either the round number is "
                f"wrong or the season's calendar differs from the one configured."
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
    df["Circuit"] = race.label
    df["EventName"] = actual_name
    df["RoundNumber"] = int(race.round_number)
    df["Season"] = season
    total_laps = getattr(session, "total_laps", None) or int(df["LapNumber"].max())
    df["TotalLaps"] = float(total_laps)

    return df[LAP_COLUMNS].reset_index(drop=True)


def load_races(
    races: list[Race], season: int = SEASON, cache_dir: str = CACHE_DIR
) -> pd.DataFrame:
    """Load several races, skipping any that fail rather than aborting the run.

    An EventMismatch is *not* skipped. A race that fails to download leaves the
    study with less data; a race that downloads the wrong circuit leaves it with
    a confident wrong answer, so that one stops everything.
    """
    frames = []
    for race in races:
        try:
            frame = load_race(race, season=season, cache_dir=cache_dir)
        except EventMismatch:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad race must not kill the study
            log.warning("Skipping %s round %d (%s): %s", season, race.round_number, race.label, exc)
            continue
        log.info(
            "Loaded %s round %-2d %-14s -> %-24s %d laps",
            season, race.round_number, race.label, frame["EventName"].iloc[0], len(frame),
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

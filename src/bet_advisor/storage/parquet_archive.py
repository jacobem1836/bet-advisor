"""
Parquet archive for raw odds snapshots.

Partitioned by sport key and date so historical queries can prune efficiently.
Path pattern: data/parquet/odds/sport=<key>/date=<yyyy-mm-dd>/<hh-mm>.parquet
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path("data/parquet")


def archive_snapshot(
    snapshot_df: pd.DataFrame,
    sport_key: str,
    captured_at: datetime | None = None,
    root: str | Path = _DEFAULT_ROOT,
) -> Path:
    """Write a snapshot DataFrame to the partitioned Parquet archive.

    Parameters
    ----------
    snapshot_df:
        DataFrame with at least columns: event_id, bookmaker, market, runner,
        price, point, captured_at, commence_time, source.
    sport_key:
        The Odds API sport key (e.g. 'aussierules_afl'). Used as the first
        partition level.
    captured_at:
        Timestamp for the snapshot. Defaults to UTC now. Determines the
        date and time components of the partition path.
    root:
        Root directory for the archive tree. Defaults to 'data/parquet'.

    Returns
    -------
    Path
        The path of the written Parquet file.
    """
    if snapshot_df.empty:
        raise ValueError("snapshot_df must not be empty")

    ts = captured_at or datetime.now(timezone.utc)
    date_str = ts.strftime("%Y-%m-%d")
    time_str = ts.strftime("%H-%M")

    out_dir = Path(root) / "odds" / f"sport={sport_key}" / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{time_str}.parquet"
    snapshot_df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(
        "Archived %d odds rows to %s", len(snapshot_df), out_path
    )
    return out_path


def read_snapshots(
    sport_key: str,
    date: str,
    root: str | Path = _DEFAULT_ROOT,
) -> pd.DataFrame:
    """Read all Parquet files for a given sport and date partition.

    Parameters
    ----------
    sport_key:
        The Odds API sport key.
    date:
        Date string in 'YYYY-MM-DD' format.
    root:
        Root directory for the archive tree.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame for all snapshots on that date, or an empty
        DataFrame if no files exist.
    """
    partition = Path(root) / "odds" / f"sport={sport_key}" / f"date={date}"
    files = list(partition.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in sorted(files)]
    return pd.concat(frames, ignore_index=True)

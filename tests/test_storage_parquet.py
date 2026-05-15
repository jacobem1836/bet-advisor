"""Tests for ParquetArchive -- partition path correctness and roundtrip."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from bet_advisor.storage.parquet_archive import archive_snapshot, read_snapshots


def _sample_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": f"EVT{i}",
                "bookmaker": "sportsbet",
                "market": "h2h",
                "runner": "Richmond",
                "price": 1.85,
                "point": None,
                "captured_at": "2026-04-01T08:00:00Z",
                "commence_time": "2026-04-01T10:00:00Z",
                "source": "odds_api",
            }
            for i in range(n)
        ]
    )


class TestArchiveSnapshot:
    def test_writes_parquet_file(self, tmp_path: Path) -> None:
        df = _sample_df(3)
        ts = datetime(2026, 4, 1, 8, 30, tzinfo=timezone.utc)
        out = archive_snapshot(df, "aussierules_afl", captured_at=ts, root=tmp_path)
        assert out.exists()
        assert out.suffix == ".parquet"

    def test_partition_path_structure(self, tmp_path: Path) -> None:
        df = _sample_df(2)
        ts = datetime(2026, 4, 1, 14, 45, tzinfo=timezone.utc)
        out = archive_snapshot(df, "aussierules_afl", captured_at=ts, root=tmp_path)
        # Expected: <root>/odds/sport=aussierules_afl/date=2026-04-01/14-45.parquet
        assert "sport=aussierules_afl" in str(out)
        assert "date=2026-04-01" in str(out)
        assert out.name == "14-45.parquet"

    def test_roundtrip_read_snapshots(self, tmp_path: Path) -> None:
        df = _sample_df(4)
        ts = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
        archive_snapshot(df, "aussierules_afl", captured_at=ts, root=tmp_path)

        result = read_snapshots("aussierules_afl", "2026-04-01", root=tmp_path)
        assert len(result) == 4
        assert set(result["market"].unique()) == {"h2h"}

    def test_empty_df_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            archive_snapshot(pd.DataFrame(), "aussierules_afl", root=tmp_path)

    def test_read_missing_partition_returns_empty(self, tmp_path: Path) -> None:
        result = read_snapshots("aussierules_afl", "2020-01-01", root=tmp_path)
        assert result.empty

    def test_multiple_snapshots_same_date(self, tmp_path: Path) -> None:
        for hour in [8, 9, 10]:
            ts = datetime(2026, 4, 1, hour, 0, tzinfo=timezone.utc)
            archive_snapshot(_sample_df(2), "aussierules_afl", captured_at=ts, root=tmp_path)

        result = read_snapshots("aussierules_afl", "2026-04-01", root=tmp_path)
        assert len(result) == 6  # 3 snapshots * 2 rows each

    def test_different_sport_keys_isolated(self, tmp_path: Path) -> None:
        ts = datetime(2026, 4, 1, 8, 0, tzinfo=timezone.utc)
        archive_snapshot(_sample_df(3), "aussierules_afl", captured_at=ts, root=tmp_path)
        archive_snapshot(_sample_df(5), "nrl", captured_at=ts, root=tmp_path)

        afl = read_snapshots("aussierules_afl", "2026-04-01", root=tmp_path)
        nrl = read_snapshots("nrl", "2026-04-01", root=tmp_path)
        assert len(afl) == 3
        assert len(nrl) == 5

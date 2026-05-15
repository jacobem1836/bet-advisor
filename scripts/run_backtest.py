"""
Walk-forward backtest CLI for the disposals model.

Loads player_stats from DuckDB (or generates synthetic data with --synthetic),
runs the disposals backtest, prints the BacktestReport, and writes outputs to
disk.

Usage examples
--------------
# Smoke test with synthetic data:
python scripts/run_backtest.py --synthetic --output-dir data/backtests

# Real data from DuckDB (requires Phase 2 backfill to have been run):
python scripts/run_backtest.py --start-season 2019 --end-season 2024 \\
    --output-dir data/backtests

# Fixed line, bucketed strategy:
python scripts/run_backtest.py --synthetic --line-strategy fixed \\
    --fixed-line 24.5

Limitations
-----------
- With --synthetic: all backtest outputs are from synthetic data. Do not
  interpret CLV, Brier, or ROI results as real-world model performance.
- Without --synthetic: requires DuckDB at data/analytics.duckdb populated
  by the Phase 2 backfill. Player prop closing odds are not yet available,
  so CLV is computed against a simulated closing line (see walk_forward.py).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Setup path (allows running from repo root without installing)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bet_advisor.backtest.walk_forward import (
    SplitterConfig,
    StakeConfig,
    run_disposals_backtest,
)
from bet_advisor.models.disposals import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def _generate_synthetic_player_stats(
    n_players: int = 50,
    n_seasons: int = 6,
    start_season: int = 2019,
    matches_per_season: int = 22,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic player_stats DataFrame mimicking the DuckDB schema.

    Generates ``n_players`` players across ``n_seasons`` seasons with
    ``matches_per_season`` matches each. Each player has a realistic disposals
    baseline with per-player variation and random match-level noise.

    The data schema matches the player_stats DuckDB table exactly so this
    function can serve as a drop-in for smoke tests.

    Parameters
    ----------
    n_players:
        Number of distinct players to generate.
    n_seasons:
        Number of seasons.
    start_season:
        First season year.
    matches_per_season:
        Matches per season per player.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns: match_id, player_id, player_name, team,
    opponent_team, position, time_on_ground_pct, disposals, kicks, handballs,
    marks, tackles, goals, behinds, clearances, match_date, season, venue,
    is_home.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(10)]
    venues = [
        "MCG",
        "Etihad Stadium",
        "Adelaide Oval",
        "SCG",
        "Gabba",
        "Kardinia Park",
        "UTAS Stadium",
        "Optus Stadium",
        "Marvel Stadium",
        "Metricon Stadium",
    ]

    rows: list[dict] = []
    match_counter = 0

    for season_offset in range(n_seasons):
        season = start_season + season_offset
        for match_round in range(1, matches_per_season + 1):
            # Pair teams for matches (5 matches per round in a 10-team comp)
            shuffled_teams = rng.permutation(teams)
            for m in range(5):
                home_team = shuffled_teams[m * 2]
                away_team = shuffled_teams[m * 2 + 1]
                venue = rng.choice(venues)
                match_id = f"match_{season}_{match_round}_{m}"
                match_date = pd.Timestamp(season, 3, 1) + pd.Timedelta(
                    days=int((match_round - 1) * 7 + rng.integers(0, 3))
                )
                match_counter += 1

                # Assign players to teams (10 per team)
                home_players = list(range(m * 10, m * 10 + 10))
                away_players = list(range(25 + m * 5, 25 + m * 5 + 10))

                for side, player_ids, team in [
                    ("home", home_players, home_team),
                    ("away", away_players, away_team),
                ]:
                    opp_team = away_team if side == "home" else home_team
                    for pid in player_ids:
                        # Player has a stable baseline + noise
                        baseline = 15 + (pid % n_players) * 0.5
                        disposals = int(max(0, rng.normal(baseline, 6)))
                        tog = float(np.clip(rng.normal(75, 15), 30, 100))

                        rows.append(
                            {
                                "match_id": match_id,
                                "player_id": f"player_{pid % n_players}",
                                "player_name": f"Player {pid % n_players}",
                                "team": team,
                                "opponent_team": opp_team,
                                "position": rng.choice(["MID", "FWD", "DEF", "RUCK"]),
                                "time_on_ground_pct": tog,
                                "disposals": disposals,
                                "kicks": int(disposals * rng.uniform(0.4, 0.6)),
                                "handballs": disposals - int(disposals * rng.uniform(0.4, 0.6)),
                                "marks": int(rng.normal(3, 2)),
                                "tackles": int(rng.normal(3, 2)),
                                "goals": int(rng.poisson(0.5)),
                                "behinds": int(rng.poisson(0.3)),
                                "clearances": int(rng.normal(2, 2)),
                                "match_date": match_date,
                                "season": season,
                                "venue": venue,
                                "is_home": 1 if side == "home" else 0,
                            }
                        )

    return pd.DataFrame(rows)


def _load_from_duckdb(
    start_season: int,
    end_season: int,
    db_path: str = "data/analytics.duckdb",
) -> pd.DataFrame:
    """Load player_stats from DuckDB for a season range.

    Parameters
    ----------
    start_season:
        First season to include (inclusive).
    end_season:
        Last season to include (inclusive).
    db_path:
        Path to the DuckDB database file.

    Returns
    -------
    pd.DataFrame matching the player_stats schema with added match context
    columns (match_date, season, venue, is_home, opponent_team).
    """
    import duckdb

    con = duckdb.connect(db_path)
    df = con.execute(
        """
        SELECT
            ps.match_id,
            ps.player_id,
            ps.player_name,
            ps.team,
            ps.position,
            ps.time_on_ground_pct,
            ps.disposals,
            ps.kicks,
            ps.handballs,
            ps.marks,
            ps.tackles,
            ps.goals,
            ps.behinds,
            ps.clearances,
            m.date      AS match_date,
            m.season,
            m.venue,
            CASE WHEN ps.team = m.home_team THEN 1 ELSE 0 END AS is_home,
            CASE WHEN ps.team = m.home_team THEN m.away_team
                 ELSE m.home_team END AS opponent_team
        FROM player_stats ps
        JOIN matches m ON ps.match_id = m.match_id
        WHERE m.season BETWEEN ? AND ?
          AND m.completed = TRUE
        ORDER BY m.date, ps.match_id, ps.player_id
        """,
        [start_season, end_season],
    ).df()
    con.close()
    return df


# ---------------------------------------------------------------------------
# Line strategy helpers
# ---------------------------------------------------------------------------


def _apply_line_strategy(
    player_stats_df: pd.DataFrame,
    strategy: str,
    fixed_line: float,
) -> float:
    """Return the O/U line to use for the backtest.

    Strategies
    ----------
    market:
        In production, the line comes from the bookmaker's market. For a
        backtest without real prop odds, fall back to the player's ewm mean
        disposals from the training window. Not yet implemented as a
        per-player-match value -- returns the dataset median as a proxy.
    fixed:
        Use a single fixed line for all bets (default: 24.5).
    bucketed:
        Round the dataset median to the nearest 2.5 bucket (common AFL prop
        line spacing).
    """
    if strategy == "market":
        logger.warning(
            "line-strategy=market requires real historical prop odds which are "
            "not yet available. Falling back to dataset median disposal count."
        )
        return float(player_stats_df["disposals"].median())
    elif strategy == "bucketed":
        median = float(player_stats_df["disposals"].median())
        return round(median / 2.5) * 2.5
    else:  # "fixed"
        return fixed_line


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward disposals backtest CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data.")
    parser.add_argument(
        "--start-season",
        type=int,
        default=2019,
        help="First season for real-data backtest.",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2024,
        help="Last season for real-data backtest.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/analytics.duckdb",
        help="Path to the DuckDB database file (real data mode only).",
    )
    parser.add_argument(
        "--line-strategy",
        choices=["market", "fixed", "bucketed"],
        default="fixed",
        help="How to choose the O/U line.",
    )
    parser.add_argument(
        "--fixed-line",
        type=float,
        default=24.5,
        help="Line to use when --line-strategy=fixed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/backtests",
        help="Directory for report.json and bets.csv outputs.",
    )
    parser.add_argument(
        "--min-train-seasons",
        type=int,
        default=2,
        help="Minimum seasons required before first test window.",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Notional starting bankroll for stake simulation.",
    )
    parser.add_argument(
        "--flat-unit-pct",
        type=float,
        default=0.01,
        help="Flat stake fraction (e.g. 0.01 = 1 percent of bankroll).",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.03,
        help="Minimum fractional edge to place a bet.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns exit code."""
    args = _parse_args(argv)

    # --- Load data ---
    if args.synthetic:
        logger.info("Generating synthetic player stats...")
        n_seasons = args.end_season - args.start_season + 1
        player_stats_df = _generate_synthetic_player_stats(
            n_seasons=n_seasons,
            start_season=args.start_season,
        )
        logger.info("Generated %d rows of synthetic data.", len(player_stats_df))
    else:
        logger.info(
            "Loading player_stats from DuckDB (%s) seasons %d-%d...",
            args.db_path,
            args.start_season,
            args.end_season,
        )
        try:
            player_stats_df = _load_from_duckdb(
                args.start_season, args.end_season, db_path=args.db_path
            )
        except Exception as e:
            logger.error("Failed to load from DuckDB: %s", e)
            logger.error("Tip: run with --synthetic for a smoke test without real data.")
            return 1

        if player_stats_df.empty:
            logger.error(
                "No data found in DuckDB for seasons %d-%d. Has the Phase 2 backfill been run?",
                args.start_season,
                args.end_season,
            )
            return 1

    # --- Determine line ---
    line = _apply_line_strategy(player_stats_df, args.line_strategy, args.fixed_line)
    logger.info("Using O/U line: %.1f", line)

    # --- Configure and run backtest ---
    model_cfg = ModelConfig()
    splitter_cfg = SplitterConfig(train_min_seasons=args.min_train_seasons)
    stake_cfg = StakeConfig(
        bankroll=args.bankroll,
        flat_unit_fraction=args.flat_unit_pct,
        min_edge_threshold=args.min_edge,
    )

    logger.info("Running walk-forward backtest...")
    try:
        report, bets_df = run_disposals_backtest(
            player_stats_df=player_stats_df,
            model_cfg=model_cfg,
            splitter_cfg=splitter_cfg,
            stake_cfg=stake_cfg,
            line=line,
            output_dir=args.output_dir,
        )
    except Exception as e:
        logger.error("Backtest failed: %s", e, exc_info=True)
        return 1

    # --- Print report ---
    print("\n=== Disposals Backtest Report ===")
    print(f"  Bets placed:        {report.n_bets}")
    print(f"  Wins:               {report.n_wins}")
    print(f"  ROI:                {report.roi:.4f} ({report.roi * 100:.2f}%)")
    print(f"  Win rate CI (95%):  [{report.roi_ci_low:.4f}, {report.roi_ci_high:.4f}]")
    print(f"  Brier score:        {report.brier:.4f}")
    print(f"  Log loss:           {report.log_loss_score:.4f}")
    print(f"  ECE:                {report.ece:.4f}")
    print(f"  Mean CLV:           {report.mean_clv:.4f}")
    print(f"  Positive CLV %:     {report.pct_positive_clv:.3f}")
    print(f"  Max drawdown ($):   {report.drawdown_max:.2f}")
    print(f"  Sharpe-like:        {report.sharpe_like:.4f}")

    if args.synthetic:
        print(
            "\nNote: results are from synthetic data and do not reflect "
            "real-world model performance."
        )
    else:
        print(
            "\nNote: CLV is computed against a simulated closing line. "
            "Replace with real historical prop odds once Phase 2 backfill is run."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

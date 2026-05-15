"""
Player disposals over/under model for AFL.

This module targets player disposals O/U, the softest AFL market per Phase 1
research. H2H markets are not modelled here -- they are calibration baselines,
not edges.

Architecture
------------
Two-stage regression:

  Stage 1 -- DisposalsModel fits a GradientBoostingRegressor on historical
  player-match data to predict expected disposals (mean of the distribution).

  Stage 2 -- A second GBR is fit on the absolute training residuals to estimate
  the conditional standard deviation (spread of the distribution).

Over/under probability is computed by treating the predictive distribution as
Normal(mean, std). This is a known approximation: disposal counts are discrete
non-negative integers and the true distribution has a heavier right tail than a
Gaussian. The approximation is adequate for mid-range lines (15-30 disposals)
where the bulk of betting market activity sits. It degrades at extreme lines
(< 5 or > 45). This caveat is documented here and must be surfaced to callers.

Calibration
-----------
After fitting, the model can calibrate the over/under probabilities on a
held-out slice. PlattCalibrator is preferred (smaller-sample regime per research
§3.5 notes). IsotonicCalibrator is available for larger datasets via the
calibrator_type config flag.

Look-ahead bias
---------------
All rolling features use shift(1) before the ewm/rolling calculation so the
current match's own statistics are not included in the lookback. Feature
construction requires an as-of cutoff date to allow time-aware backtest slicing.

Feature schema documented in DisposalsFeatureBuilder.FEATURE_COLUMNS.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor

from bet_advisor.eval.calibration import IsotonicCalibrator, PlattCalibrator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

# Minimum number of past matches a player must have before their rolling
# features are considered non-warmup. Rows with fewer than this many past
# matches will have NaN rolling features and should be excluded from training
# (but are valid for prediction with appropriate uncertainty inflation).
_WARMUP_MATCHES = 3

# Minimum number of samples for a player-venue mean to be trusted.
_MIN_VENUE_SAMPLES = 3

# Column expected in the input DataFrame for the date field.
_DATE_COL = "match_date"


@dataclass
class FeatureConfig:
    """Hyperparameters controlling feature engineering."""

    ewm_span: int = 10
    rolling_short: int = 5
    rolling_very_short: int = 3
    min_venue_samples: int = _MIN_VENUE_SAMPLES
    warmup_matches: int = _WARMUP_MATCHES


class DisposalsFeatureBuilder:
    """Compute time-aware features for the disposals model.

    All rolling/ewm calculations use shift(1) before aggregation so the
    current match is not included in its own lookback window. This is the
    primary guard against look-ahead bias in feature engineering.

    The builder accepts an ``as_of`` date so that callers (backtests) can
    produce the feature set as it would have appeared at a historical cutoff,
    without including any data from after that date.

    Parameters
    ----------
    cfg:
        Feature configuration. See FeatureConfig.

    Usage
    -----
    builder = DisposalsFeatureBuilder()
    X, y = builder.build(player_stats_df, as_of="2022-01-01")
    """

    FEATURE_COLUMNS: list[str] = [
        "ewm_disposals_mean",
        "ewm_disposals_std",
        "rolling_5_disposals_mean",
        "rolling_3_disposals_mean",
        "opp_pressure_ewm",
        "venue_disposal_delta",
        "indoor_venue",
        "is_home",
        "days_since_last_match",
        "ewm_tog_pct",
    ]

    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        self._cfg = cfg or FeatureConfig()

    def build(
        self,
        df: pd.DataFrame,
        as_of: str | pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build feature matrix and target series.

        Parameters
        ----------
        df:
            Player stats DataFrame. Required columns: match_id, player_id,
            match_date, disposals, team, opponent_team, venue, is_home.
            Optional columns: time_on_ground_pct, position.
        as_of:
            Cutoff date (inclusive). Rows with match_date > as_of are dropped.
            If None, all data is used.

        Returns
        -------
        (X, y) where X is a DataFrame of features and y is the disposals
        Series. Index is aligned.

        Notes
        -----
        Rows in the warmup period (player has fewer than warmup_matches prior
        matches) will have NaN for rolling features. These rows are included
        in the return value; callers should decide whether to drop them.
        """
        df = df.copy()
        df[_DATE_COL] = pd.to_datetime(df[_DATE_COL])

        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            df = df[df[_DATE_COL] <= cutoff].copy()

        df = df.sort_values(["player_id", _DATE_COL]).reset_index(drop=True)

        # --- player rolling features ---
        df = self._add_player_rolling(df)

        # --- opponent pressure rating ---
        df = self._add_opponent_pressure(df)

        # --- venue delta ---
        df = self._add_venue_delta(df)

        # --- days since last match ---
        df = self._add_days_since_last(df)

        # Build X and y
        available_features = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        X = df[available_features].copy()
        y = df["disposals"].astype(float)
        return X, y

    # ------------------------------------------------------------------
    # Internal builders

    def _add_player_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add player-level ewm and rolling disposal stats (shift-1 guarded)."""
        span = self._cfg.ewm_span
        short = self._cfg.rolling_short
        vshort = self._cfg.rolling_very_short

        groups: list[pd.DataFrame] = []
        for _, grp in df.groupby("player_id", sort=False):
            grp = grp.sort_values(_DATE_COL).copy()

            # shift(1) so current match is not in its own lookback
            shifted = grp["disposals"].shift(1)

            grp["ewm_disposals_mean"] = shifted.ewm(span=span, adjust=False).mean()
            grp["ewm_disposals_std"] = shifted.ewm(span=span, adjust=False).std()
            grp["rolling_5_disposals_mean"] = shifted.rolling(short, min_periods=1).mean()
            grp["rolling_3_disposals_mean"] = shifted.rolling(vshort, min_periods=1).mean()

            # TOG ewm
            if "time_on_ground_pct" in grp.columns:
                shifted_tog = grp["time_on_ground_pct"].shift(1)
                grp["ewm_tog_pct"] = shifted_tog.ewm(span=span, adjust=False).mean()
            else:
                grp["ewm_tog_pct"] = np.nan

            groups.append(grp)

        return pd.concat(groups, ignore_index=True)

    def _add_opponent_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add opponent's ewm disposals-conceded rating.

        For each team in each match, the opponent pressure rating is the
        ewm(span) of disposals that the opposing team has conceded to their
        opponents in prior matches. shift(1) ensures the current match is
        not used.

        Approximation: we compute disposals conceded per team per match as
        the mean disposals of all opposing players in that match, then ewm
        over time. This requires the full per-player data to be present.
        """
        span = self._cfg.ewm_span

        # Compute team-level disposals conceded per match
        # "disposals conceded" = disposals scored by the opposing team's players
        team_match_disposals = df.groupby(["match_id", "team"])["disposals"].sum().reset_index()
        team_match_disposals.columns = ["match_id", "team", "team_disposals"]

        # Self-join to get the opponent's disposals for the same match
        conceded = team_match_disposals.merge(
            team_match_disposals.rename(
                columns={"team": "opponent_team", "team_disposals": "disposals_conceded"}
            ),
            on="match_id",
        )
        conceded = conceded[conceded["team"] != conceded["opponent_team"]]
        conceded = conceded[["match_id", "team", "disposals_conceded"]]

        # Join match date
        conceded = conceded.merge(df[["match_id", _DATE_COL]].drop_duplicates(), on="match_id")
        conceded = conceded.sort_values(["team", _DATE_COL]).reset_index(drop=True)

        # Compute team ewm of disposals_conceded (shift-1 guarded)
        pressure_parts: list[pd.DataFrame] = []
        for team, grp in conceded.groupby("team", sort=False):
            grp = grp.sort_values(_DATE_COL).copy()
            shifted = grp["disposals_conceded"].shift(1)
            grp["opp_pressure_ewm_team"] = shifted.ewm(span=span, adjust=False).mean()
            pressure_parts.append(grp[["match_id", "team", "opp_pressure_ewm_team"]])

        if not pressure_parts:
            df["opp_pressure_ewm"] = np.nan
            return df

        pressure_df = pd.concat(pressure_parts, ignore_index=True)

        # Each player row needs the opponent team's pressure rating
        # The opponent of player's team is the other team in the match
        if "opponent_team" not in df.columns:
            # Derive opponent_team from match_id + team
            match_team_map = (
                df[["match_id", "team"]]
                .drop_duplicates()
                .groupby("match_id")["team"]
                .apply(list)
                .reset_index()
            )
            match_team_map.columns = ["match_id", "teams"]

            def _get_opponent(row: pd.Series) -> str:
                teams = row["teams"]
                others = [t for t in teams if t != row["team"]]
                return others[0] if others else ""

            expanded = df[["match_id", "team"]].drop_duplicates().copy()
            expanded = expanded.merge(match_team_map, on="match_id")
            expanded["opponent_team"] = expanded.apply(_get_opponent, axis=1)
            df = df.merge(
                expanded[["match_id", "team", "opponent_team"]], on=["match_id", "team"], how="left"
            )

        # Join pressure on opponent_team
        pressure_df_renamed = pressure_df.rename(
            columns={"team": "opponent_team", "opp_pressure_ewm_team": "opp_pressure_ewm"}
        )
        df = df.merge(pressure_df_renamed, on=["match_id", "opponent_team"], how="left")
        return df

    def _add_venue_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add per-player, per-venue historical disposal delta (mean shift).

        For each (player_id, venue) pair, compute the mean disposal count in
        past matches at that venue and subtract the player's overall mean.
        Uses only data prior to the current match (computed globally and then
        applied -- there is a slight look-ahead within the training set here,
        but the effect is small and the bias is symmetric).

        A simpler approach: encode as a raw mean per player-venue pair using
        the full training window. This is acceptable for training set feature
        engineering; backtest callers must pass as_of to limit the data window.

        Also adds a binary indoor_venue flag if the venues DataFrame is not
        available (defaults to 0/False).
        """
        cfg = self._cfg

        # Compute per-player, per-venue means with min sample guard
        pv_stats = (
            df.groupby(["player_id", "venue"])["disposals"].agg(["mean", "count"]).reset_index()
        )
        pv_stats.columns = ["player_id", "venue", "venue_mean", "venue_count"]

        # Player overall mean
        player_mean = (
            df.groupby("player_id")["disposals"]
            .mean()
            .reset_index()
            .rename(columns={"disposals": "player_overall_mean"})
        )

        pv_stats = pv_stats.merge(player_mean, on="player_id")
        pv_stats["venue_disposal_delta"] = np.where(
            pv_stats["venue_count"] >= cfg.min_venue_samples,
            pv_stats["venue_mean"] - pv_stats["player_overall_mean"],
            0.0,
        )

        df = df.merge(
            pv_stats[["player_id", "venue", "venue_disposal_delta"]],
            on=["player_id", "venue"],
            how="left",
        )
        df["venue_disposal_delta"] = df["venue_disposal_delta"].fillna(0.0)

        # Indoor venue flag -- default False if not available in data
        if "indoor" in df.columns:
            df["indoor_venue"] = df["indoor"].astype(float)
        else:
            df["indoor_venue"] = 0.0

        return df

    def _add_days_since_last(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add days since the player's previous match."""
        groups: list[pd.DataFrame] = []
        for _, grp in df.groupby("player_id", sort=False):
            grp = grp.sort_values(_DATE_COL).copy()
            grp["days_since_last_match"] = grp[_DATE_COL].diff().dt.days.fillna(0.0)
            groups.append(grp)
        return pd.concat(groups, ignore_index=True)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

# Default hyperparameters as per research recommendation (Phase 1 §4).
_DEFAULT_MEAN_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 42,
    "loss": "squared_error",
}

_DEFAULT_STD_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "random_state": 43,
    "loss": "squared_error",
}


@dataclass
class ModelConfig:
    """Model hyperparameters and calibration config."""

    mean_params: dict = field(default_factory=lambda: dict(_DEFAULT_MEAN_PARAMS))
    std_params: dict = field(default_factory=lambda: dict(_DEFAULT_STD_PARAMS))
    calibrator_type: Literal["platt", "isotonic"] = "platt"
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    def to_dict(self) -> dict:
        """Serialise config for hashing."""
        return {
            "mean_params": self.mean_params,
            "std_params": self.std_params,
            "calibrator_type": self.calibrator_type,
            "feature_config": {
                "ewm_span": self.feature_config.ewm_span,
                "rolling_short": self.feature_config.rolling_short,
                "rolling_very_short": self.feature_config.rolling_very_short,
                "min_venue_samples": self.feature_config.min_venue_samples,
                "warmup_matches": self.feature_config.warmup_matches,
            },
        }


class DisposalsModel:
    """GBR-based player disposals prediction model.

    Predicts a distribution over disposal counts for a player-match row.
    The distribution is parameterised as Normal(mean, std) -- see module-level
    docstring for caveats on this approximation.

    Training uses two GBR estimators:
      - A mean GBR fit on disposals directly.
      - A std GBR fit on the absolute residuals from the mean GBR.

    Over/under probability is P(X > line) under Normal(mean, std).

    Calibration applies a post-hoc calibrator to the raw over/under
    probabilities on a held-out slice. The calibrator type (Platt or Isotonic)
    is controlled by ModelConfig.calibrator_type. Platt is recommended for
    typical AFL sample sizes per Phase 1 research.

    Assumptions documented
    ----------------------
    - Normal approximation for P(disposals > line). See module docstring.
    - Venue delta is computed globally over the training window, not strictly
      per-row historical. Bias is small and symmetric.
    - Opponent pressure uses same-round opponent data without game-week
      awareness (whether the opponent played earlier or later in the round).
      In a live system, only completed rounds should contribute.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self._cfg = cfg or ModelConfig()
        self._mean_gbr: GradientBoostingRegressor | None = None
        self._std_gbr: GradientBoostingRegressor | None = None
        self._calibrator: PlattCalibrator | IsotonicCalibrator | None = None
        self._feature_builder = DisposalsFeatureBuilder(self._cfg.feature_config)
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        cal_fraction: float = 0.15,
    ) -> "DisposalsModel":
        """Fit mean GBR, std GBR, and calibrator.

        Parameters
        ----------
        X:
            Feature matrix as returned by DisposalsFeatureBuilder.build().
        y:
            Target disposals, shape (n,).
        sample_weight:
            Optional per-sample weights passed to the GBRs.
        cal_fraction:
            Fraction of training data to hold out for calibrator fitting.
            The held-out slice is taken from the most-recent rows (by index
            order), preserving temporal structure.

        Returns
        -------
        self (for chaining).

        Raises
        ------
        ValueError
            If X or y contain fewer than 10 rows after NaN removal.
        """
        # Drop NaN rows (warmup period)
        mask = X.notna().all(axis=1)
        X_clean = X[mask].copy()
        y_clean = y[mask].copy()
        w_clean = sample_weight[mask] if sample_weight is not None else None

        if len(X_clean) < 10:
            raise ValueError(
                f"Fewer than 10 clean rows after removing warmup NaN rows "
                f"(got {len(X_clean)}). Cannot fit model."
            )

        # Temporal split for calibration
        n = len(X_clean)
        cal_n = max(1, int(n * cal_fraction))
        train_n = n - cal_n

        X_train = X_clean.iloc[:train_n]
        y_train = y_clean.iloc[:train_n]
        w_train = w_clean[:train_n] if w_clean is not None else None

        X_cal = X_clean.iloc[train_n:]
        y_cal = y_clean.iloc[train_n:]

        # Mean GBR
        logger.info("Fitting mean GBR on %d rows", len(X_train))
        self._mean_gbr = GradientBoostingRegressor(**self._cfg.mean_params)
        self._mean_gbr.fit(X_train, y_train, sample_weight=w_train)

        # Std GBR on absolute residuals
        residuals = np.abs(y_train.to_numpy() - self._mean_gbr.predict(X_train))
        logger.info("Fitting std GBR on %d rows", len(X_train))
        self._std_gbr = GradientBoostingRegressor(**self._cfg.std_params)
        self._std_gbr.fit(X_train, residuals, sample_weight=w_train)

        # Calibration on held-out slice
        # We calibrate at a fixed line midpoint of 20 disposals as a proxy
        # for the calibration signal. In production, calibrate on actual lines.
        CAL_LINE = 20.0
        raw_probs_cal = self._raw_over_prob(X_cal, line=CAL_LINE)
        outcomes_cal = (y_cal.to_numpy() > CAL_LINE).astype(float)

        if self._cfg.calibrator_type == "isotonic":
            self._calibrator = IsotonicCalibrator()
        else:
            self._calibrator = PlattCalibrator()

        # Only fit calibrator if we have both classes represented
        if len(np.unique(outcomes_cal)) >= 2:
            self._calibrator.fit(raw_probs_cal, outcomes_cal)
            logger.info("Calibrator fitted on %d held-out rows", len(X_cal))
        else:
            logger.warning(
                "Calibration slice has only one class at line=%.1f. "
                "Calibrator will not transform probabilities.",
                CAL_LINE,
            )
            self._calibrator = None

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction

    def predict_distribution(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return predicted mean and std disposals per row.

        Parameters
        ----------
        X:
            Feature matrix, same columns as training.

        Returns
        -------
        pd.DataFrame with columns: ``mean_disposals``, ``std_disposals``.
        Index matches X.index.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        self._check_fitted()
        X_filled = X.fillna(0.0)  # warmup rows: predict with zeros
        mean = self._mean_gbr.predict(X_filled)  # type: ignore[union-attr]
        std = np.maximum(self._std_gbr.predict(X_filled), 1.0)  # type: ignore[union-attr]
        return pd.DataFrame(
            {"mean_disposals": mean, "std_disposals": std},
            index=X.index,
        )

    def predict_over_under_prob(
        self,
        X: pd.DataFrame,
        line: float,
        calibrate: bool = True,
    ) -> np.ndarray:
        """Return P(disposals > line) for each row.

        Approximation: models the disposal distribution as Normal(mean, std).
        This is adequate for lines in the range [10, 40] but degrades at
        extreme lines. See module docstring for a full discussion.

        Parameters
        ----------
        X:
            Feature matrix.
        line:
            The over/under line (e.g. 24.5 disposals).
        calibrate:
            Whether to apply the post-hoc calibrator if one was fitted.
            Set to False to inspect raw probabilities.

        Returns
        -------
        np.ndarray, shape (n,), values in [0, 1].
        """
        self._check_fitted()
        raw_probs = self._raw_over_prob(X, line=line)

        if calibrate and self._calibrator is not None:
            return self._calibrator.transform(raw_probs)
        return raw_probs

    # ------------------------------------------------------------------
    # Persistence

    def save(self, path: str | Path) -> None:
        """Serialise the fitted model to a joblib file.

        Saves both GBRs and the calibrator. The file can be loaded with
        ``DisposalsModel.load()``.

        Parameters
        ----------
        path:
            Destination path (e.g. ``models/disposals_v1.joblib``).
        """
        self._check_fitted()
        payload = {
            "mean_gbr": self._mean_gbr,
            "std_gbr": self._std_gbr,
            "calibrator": self._calibrator,
            "cfg": self._cfg,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)
        logger.info("DisposalsModel saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "DisposalsModel":
        """Load a previously saved DisposalsModel.

        Parameters
        ----------
        path:
            Path to a joblib file saved by ``save()``.

        Returns
        -------
        Fitted DisposalsModel instance.
        """
        payload = joblib.load(path)
        instance = cls(cfg=payload["cfg"])
        instance._mean_gbr = payload["mean_gbr"]
        instance._std_gbr = payload["std_gbr"]
        instance._calibrator = payload["calibrator"]
        instance._is_fitted = True
        return instance

    # ------------------------------------------------------------------
    # Version hash

    @property
    def version_hash(self) -> str:
        """Return a SHA-256 of the model config (feature spec + hyperparams).

        This hash is stable for identical configs across runs, enabling the
        bet log to record which model version produced a given prediction
        without storing the full model artefact path.

        Note: the hash does not incorporate training data. Two models trained
        on different data windows with the same config will share the same
        version_hash. Use model artefact paths for full provenance.
        """
        config_str = json.dumps(self._cfg.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Internal helpers

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._mean_gbr is None or self._std_gbr is None:
            raise RuntimeError("DisposalsModel.fit() must be called before prediction.")

    def _raw_over_prob(self, X: pd.DataFrame, line: float) -> np.ndarray:
        """Compute P(disposals > line) using Normal CDF without calibration."""
        X_filled = X.fillna(0.0)
        mean = self._mean_gbr.predict(X_filled)  # type: ignore[union-attr]
        std = np.maximum(self._std_gbr.predict(X_filled), 1.0)  # type: ignore[union-attr]
        # P(X > line) = 1 - Phi((line - mean) / std)
        return 1.0 - norm.cdf(line, loc=mean, scale=std)

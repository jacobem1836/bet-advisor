"""
Recommendation package.

Re-exports the core engine types and helpers.
"""

from bet_advisor.recommend.engine import (
    Recommendation,
    RecommendationConfig,
    RecommendationEngine,
    UntrainedModelError,
)
from bet_advisor.recommend.model_health import (
    compute_model_health,
    ensure_model_health_table,
    record_model_health,
)
from bet_advisor.recommend.pnl import compute_pnl_snapshot

__all__ = [
    "Recommendation",
    "RecommendationConfig",
    "RecommendationEngine",
    "UntrainedModelError",
    "compute_model_health",
    "compute_pnl_snapshot",
    "ensure_model_health_table",
    "record_model_health",
]

"""Phase 6 -- H2H Elo baseline (calibration reference, not edge). Not yet implemented.

H2H markets are the sharpest AFL markets (Brier ~0.201 per research §6). This
model is a calibration baseline only -- not a primary betting target. See
RESEARCH.md §1 and §8 (Phase 6) for context and the roadmap entry.
"""

from __future__ import annotations


class H2HEloModel:
    """H2H Elo rating model. Phase 6 placeholder.

    Serves as a calibration reference against which disposals and totals model
    CLV is benchmarked. Full implementation deferred to Phase 6.
    """

    def fit(self, *args: object, **kwargs: object) -> "H2HEloModel":
        raise NotImplementedError("H2HEloModel is a Phase 6 stub. Not yet implemented.")

    def predict(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError("H2HEloModel is a Phase 6 stub. Not yet implemented.")

# HybridRouter - Blend several complexity signals for balanced expert routing
# Part of the Kapildhinwa2010-dev Toolkit.

"""HybridRouter - multi-signal blended routing.

Lineage: DNA_DYNIMO routers. Instead of one scalar score, four signals are
computed per input and combined with per-expert weights: L2 norm
(magnitude), normalized entropy of |v| (distribution shape), sparsity
(fraction of near-zero entries) and dimensionality. Each expert declares
a capacity in blended-score units; the router picks the cheapest
sufficient expert.

The blend weights are a modelling choice, not a learned one: routing
depends on the weights you pick. See README limitations.
"""

from __future__ import annotations

import numpy as np

__all__ = ["signal_scores", "blend_score", "DEFAULT_WEIGHTS", "HybridExpert", "HybridRouter"]


DEFAULT_WEIGHTS = {"norm": 1.0, "entropy": 1.0, "sparsity": 0.5, "dimension": 0.001}


def signal_scores(vector: np.ndarray) -> dict:
    """Compute norm, entropy, sparsity and dimension signals for a vector."""
    v = np.asarray(vector, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("vector must be non-empty")
    norm = float(np.linalg.norm(v))
    magnitudes = np.abs(v)
    total = magnitudes.sum()
    entropy = 0.0
    if total > 0:
        probs = magnitudes[magnitudes > 0] / total
        entropy = float(-(probs * np.log(probs)).sum() / np.log(v.size))
    sparsity = float((magnitudes < 1e-12).sum() / v.size)
    return {
        "norm": norm,
        "entropy": entropy,
        "sparsity": sparsity,
        "dimension": float(v.size),
    }


def blend_score(signals: dict, weights: dict = None) -> float:
    """Weighted sum of signals; missing weights fall back to defaults."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    return (w["norm"] * signals["norm"]
            + w["entropy"] * signals["entropy"]
            + w["sparsity"] * signals["sparsity"]
            + w["dimension"] * signals["dimension"])


class HybridExpert:
    """One routing target with a blended-score capacity and its own weights."""

    def __init__(self, name: str, capacity: float, cost: float = 1.0,
                 weights: dict = None) -> None:
        if not name:
            raise ValueError("name must be non-empty")
        if capacity <= 0 or cost < 0:
            raise ValueError("capacity must be > 0 and cost >= 0")
        self.name = name
        self.capacity = float(capacity)
        self.cost = float(cost)
        self.weights = dict(weights) if weights else {}
        self.calls = 0

    def handle(self, score: float) -> str:
        if score > self.capacity:
            raise ValueError("score exceeds capacity")
        self.calls += 1
        return "handled by {}".format(self.name)


class HybridRouter:
    """Route each input to the cheapest expert whose blended capacity suffices."""

    def __init__(self) -> None:
        self.experts: list = []

    def add_expert(self, name: str, capacity: float, cost: float = 1.0,
                   weights: dict = None) -> None:
        self.experts.append(HybridExpert(name, capacity, cost, weights))

    def route(self, vector: np.ndarray) -> HybridExpert:
        if not self.experts:
            raise ValueError("no experts registered")
        signals = signal_scores(vector)
        sufficient = [e for e in self.experts
                      if blend_score(signals, e.weights) <= e.capacity]
        if not sufficient:
            raise ValueError("no expert capacity covers the blended score")
        return min(sufficient, key=lambda e: (e.cost, e.capacity))

    def signal_breakdown(self, vector: np.ndarray) -> dict:
        """Per-expert blended scores for one input, for debugging routes."""
        signals = signal_scores(vector)
        return {e.name: blend_score(signals, e.weights) for e in self.experts}

    def route_and_handle(self, vector: np.ndarray) -> str:
        expert = self.route(vector)
        score = blend_score(signal_scores(vector), expert.weights)
        return expert.handle(score)

    def usage_report(self) -> dict:
        return {e.name: e.calls for e in self.experts}

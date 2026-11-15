"""Tests for HybridRouter: multi-signal blended routing."""

import numpy as np
import pytest

from hybridrouter import HybridRouter, blend_score, signal_scores, DEFAULT_WEIGHTS


def test_signal_scores_keys_and_ranges():
    signals = signal_scores(np.array([1.0, 1.0, 1.0, 0.0]))
    assert set(signals) == {"norm", "entropy", "sparsity", "dimension"}
    assert 0.0 <= signals["entropy"] <= 1.0
    assert 0.0 <= signals["sparsity"] <= 1.0
    assert signals["dimension"] == 4.0


def test_signal_scores_empty_raises():
    with pytest.raises(ValueError):
        signal_scores(np.array([]))


def test_blend_score_uses_defaults_and_overrides():
    signals = {"norm": 2.0, "entropy": 1.0, "sparsity": 0.0, "dimension": 100.0}
    base = blend_score(signals)
    assert base == pytest.approx(2.0 + 1.0 + 0.0 + 0.1)
    custom = blend_score(signals, {"norm": 10.0})
    assert custom == pytest.approx(20.0 + 1.0 + 0.0 + 0.1)
    assert DEFAULT_WEIGHTS["entropy"] == 1.0


def test_route_uses_expert_specific_weights():
    router = HybridRouter()
    # sparse-happy expert: sparsity lowers its blended score
    router.add_expert("sparse", capacity=5.0, cost=1.0, weights={"sparsity": -1.0})
    router.add_expert("dense", capacity=5.0, cost=1.0, weights={"sparsity": 2.0})
    vector = np.array([0.0, 0.0, 0.0, 1.0])
    breakdown = router.signal_breakdown(vector)
    assert breakdown["sparse"] < breakdown["dense"]
    assert router.route(vector).name == "sparse"


def test_route_no_capacity_raises():
    router = HybridRouter()
    router.add_expert("tiny", capacity=0.001)
    with pytest.raises(ValueError):
        router.route(np.array([100.0, 100.0]))


def test_route_empty_router_raises():
    with pytest.raises(ValueError):
        HybridRouter().route(np.array([1.0]))


def test_route_and_handle_and_report():
    router = HybridRouter()
    router.add_expert("e", capacity=100.0)
    out = router.route_and_handle(np.array([1.0, 2.0]))
    assert "handled by e" in out
    assert router.usage_report() == {"e": 1}

# HybridRouter

[![CI](https://github.com/kapildhinwa2010-dev/HybridRouter/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/HybridRouter/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/HybridRouter/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/HybridRouter/actions/workflows/codeql.yml)

Blend several complexity signals (norm, entropy, sparsity, dimension) for balanced expert routing.

## What it does

Instead of one scalar score, four signals are computed per input and combined
with per-expert weights. Each expert declares a capacity in blended-score
units and its own weights; the router picks the cheapest expert whose
blended score fits the capacity.

    from hybridrouter import HybridRouter

    router = HybridRouter()
    router.add_expert("sparse", capacity=5.0, weights={"sparsity": -1.0})
    router.add_expert("dense", capacity=5.0, weights={"sparsity": 2.0})
    print(router.route_and_handle([0.0, 0.0, 1.0]))

## API

- signal_scores(vector) - norm, entropy, sparsity, dimension signals
- blend_score(signals, weights) - weighted sum, defaults in DEFAULT_WEIGHTS
- HybridExpert(name, capacity, cost, weights)
- HybridRouter.add_expert / route / signal_breakdown / route_and_handle / usage_report

## Honest results and limitations

- The blend weights are a modelling choice, not a learned one: routing
  behaviour depends entirely on the weights and capacities you declare.
- Signals are on different scales (norm can be large, entropy is 0-1), so
  default weights do not make the terms comparable; tune them per use.
- No training or calibration is involved.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.

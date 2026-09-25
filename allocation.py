"""Budgeted replay sampling from the calibrated ExSpectra score."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Callable

import numpy as np


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    shifted = (values - np.max(values)) / temperature
    weights = np.exp(shifted)
    return weights / weights.sum()


def sequential_mixture_sample(
    candidates: Sequence[Any],
    scores: Sequence[float] | np.ndarray,
    *,
    batch_size: int,
    temperature: float,
    uniform_mix: float,
    rng: np.random.Generator | None = None,
    key: Callable[[Any], Any] | None = None,
) -> list[Any]:
    """Sample exactly ``batch_size`` candidates without replacement.

    At every draw the distribution is the convex mixture of a score softmax
    and the uniform distribution over the remaining candidates.  The returned
    list is in draw order; callers should apply the learner's fixed
    intervention ordering before constructing the replay batch.
    """

    items = list(candidates)
    values = np.asarray(scores, dtype=float)
    if len(items) != len(values):
        raise ValueError("candidates and scores must have the same length")
    if not items or batch_size < 0 or batch_size > len(items):
        raise ValueError("batch_size must lie between zero and the candidate count")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("uniform_mix must lie in [0, 1]")
    rng = np.random.default_rng() if rng is None else rng
    remaining = list(range(len(items)))
    chosen: list[Any] = []
    for _ in range(batch_size):
        logits = values[remaining]
        prob = (1.0 - uniform_mix) * _softmax(logits, temperature)
        prob += uniform_mix / len(remaining)
        picked = int(rng.choice(len(remaining), p=prob))
        idx = remaining.pop(picked)
        chosen.append(items[idx])
    if key is not None:
        # The method samples first and leaves fixed learner ordering explicit.
        chosen = sorted(chosen, key=key)
    return chosen

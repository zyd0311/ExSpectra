"""Probe-design and matched branch label bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class ProbeDraw:
    candidate_index: int
    propensity: float


@dataclass(frozen=True)
class AuditLabel:
    candidate_index: int
    co_adaptive: float
    frozen: float
    scripted: float
    complementarity: float
    response: float


def probe_mixture(
    audit_probabilities: Sequence[float],
    *,
    uniform_mix: float,
    rng: np.random.Generator | None = None,
) -> ProbeDraw:
    """Draw one positive-support audit candidate and record its propensity."""

    audit = np.asarray(audit_probabilities, dtype=float)
    if audit.ndim != 1 or len(audit) == 0 or np.any(audit < 0):
        raise ValueError("audit_probabilities must be a non-empty non-negative vector")
    total = audit.sum()
    if total <= 0:
        raise ValueError("audit_probabilities must have positive mass")
    audit = audit / total
    if not 0.0 < uniform_mix <= 1.0:
        raise ValueError("uniform_mix must lie in (0, 1]")
    rng = np.random.default_rng() if rng is None else rng
    probability = uniform_mix / len(audit) + (1.0 - uniform_mix) * audit
    index = int(rng.choice(len(audit), p=probability))
    return ProbeDraw(index, float(probability[index]))


def caud_labels(
    candidate_index: int,
    *,
    co_adaptive: float,
    frozen: float,
    scripted: float,
) -> AuditLabel:
    """Form total, complementarity and response labels from matched branches."""

    return AuditLabel(
        candidate_index=candidate_index,
        co_adaptive=co_adaptive,
        frozen=frozen,
        scripted=scripted,
        complementarity=scripted - frozen,
        response=co_adaptive - scripted,
    )

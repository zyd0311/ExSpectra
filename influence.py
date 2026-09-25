"""Shared-derivative coefficient assembly for ExSpectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class SharedCoefficients:
    """Candidate-shared coefficient vectors from the three CAUD channels."""

    frozen: Any
    complementarity: Any
    response: Any

    @property
    def total(self) -> Any:
        return self.frozen + self.complementarity + self.response


def build_shared_coefficients(
    g_focal: Any,
    curvature_focal_focal: Any,
    curvature_focal_teammate: Any,
    sensitivity_own: Any,
    sensitivity_response: Any,
) -> SharedCoefficients:
    """Assemble the paper's ``b_frz``, ``b_comp`` and ``b_resp``.

    Inputs are vectors in the focal parameter block.  Curvature and sensitivity
    terms are already contracted with fixed continuation directions.  A host
    learner can compute those products with its autodiff backend.
    """

    return SharedCoefficients(
        frozen=g_focal - sensitivity_own - curvature_focal_focal,
        complementarity=-curvature_focal_teammate,
        response=-sensitivity_response,
    )


def score_function_gradient(returns: Sequence[float], log_prob_gradients: Sequence[Any]) -> Any:
    """Estimate ``E[R * score]`` from fixed evaluation trajectories."""

    if len(returns) != len(log_prob_gradients) or not returns:
        raise ValueError("returns and score gradients must be non-empty and aligned")
    result = None
    for value, gradient in zip(returns, log_prob_gradients):
        term = value * gradient
        result = term if result is None else result + term
    return result / len(returns)


def cross_agent_curvature_product(
    returns: Sequence[float],
    focal_scores: Sequence[Any],
    teammate_scores: Sequence[Any],
    teammate_direction: Any,
) -> Any:
    """Compute the non-shared-policy cross-agent curvature product."""

    if not (len(returns) == len(focal_scores) == len(teammate_scores)) or not returns:
        raise ValueError("trajectory inputs must be non-empty and aligned")
    result = None
    for value, focal, teammate in zip(returns, focal_scores, teammate_scores):
        try:
            projection = teammate @ teammate_direction
        except TypeError:
            projection = np.dot(np.asarray(teammate), np.asarray(teammate_direction))
        term = value * focal * projection
        result = term if result is None else result + term
    return result / len(returns)

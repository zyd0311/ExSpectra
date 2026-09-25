"""Candidate replay displacements and the post-reference anchor."""

from __future__ import annotations

from typing import Any


def candidate_displacement(candidate_gradient: Any, reference_gradient: Any, *, step_size: float, batch_size: int) -> Any:
    """Compute the focal-block displacement from the paper's SGD surrogate.

    The host learner supplies candidate and reference focal gradients.  No
    optimizer state, clipping rule, or numeric default is embedded here.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return -(step_size / batch_size) * (candidate_gradient - reference_gradient)


def reference_anchor(parameters: Any, reference_gradient: Any, *, step_size: float) -> Any:
    """Return the focal-only post-reference SGD anchor ``z``."""

    return parameters - step_size * reference_gradient

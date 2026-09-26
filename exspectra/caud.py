"""Exact CAUD bookkeeping and score assembly.

CAUD separates the co-adaptive utility shift into the change caused by
reference teammate learning and the response induced by the candidate update:

    u_co = u_frz + (u_scr - u_frz) + (u_co - u_scr).

The functions here deliberately accept branch outputs supplied by a host
learner. They do not run an environment or make assumptions about a backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class CAUDComponents:
    """Matched utility components for one candidate or a vector of candidates."""

    frozen: Any
    complementarity: Any
    response: Any

    @property
    def structural(self) -> Any:
        return self.frozen + self.complementarity + self.response

    @property
    def co_adaptive(self) -> Any:
        return self.structural


def caud_decompose(frozen: Any, scripted: Any, co_adaptive: Any) -> CAUDComponents:
    """Return u_frz, u_scr-u_frz and u_co-u_scr exactly."""

    return CAUDComponents(
        frozen=frozen,
        complementarity=scripted - frozen,
        response=co_adaptive - scripted,
    )


def structural_score(displacement: Any, coefficients: Any, residual: Any = 0.0) -> Any:
    """Assemble the structural score plus an already-fitted residual term."""

    if hasattr(coefficients, "total"):
        total = coefficients.total
    elif all(hasattr(coefficients, name) for name in ("frozen", "complementarity", "response")):
        total = coefficients.frozen + coefficients.complementarity + coefficients.response
    else:
        total = coefficients
    try:
        return displacement @ total + residual
    except TypeError:
        return np.asarray(displacement) @ np.asarray(total) + residual


def slot_mean(values: Iterable[Any]) -> Any:
    """Average slot-level utility values without imposing a learner type."""

    values = list(values)
    if not values:
        raise ValueError("slot_mean requires at least one value")
    return sum(values) / len(values)

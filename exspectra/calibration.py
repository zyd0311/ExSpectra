"""Temporal, propensity-weighted residual calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CalibrationRecord:
    """A completed probe frozen at its source checkpoint."""

    source_step: int
    structural_prediction: float
    features: np.ndarray
    co_adaptive_utility: float
    sampling_propensity: float
    target_propensity: float


class RidgeCalibrator:
    """Fit the residual model using only records from earlier checkpoints."""

    def __init__(self, *, ridge: float, dimension: int) -> None:
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        if dimension < 0:
            raise ValueError("dimension must be non-negative")
        self.ridge = float(ridge)
        self.dimension = int(dimension)
        self.coefficients = np.zeros(self.dimension, dtype=float)
        self.fitted_through: int | None = None

    def fit(self, records: Iterable[CalibrationRecord], *, current_step: int) -> np.ndarray:
        """Fit weighted ridge on records with source_step < current_step."""

        usable = [r for r in records if r.source_step < current_step]
        if not usable:
            self.coefficients = np.zeros(self.dimension, dtype=float)
            self.fitted_through = current_step
            return self.coefficients.copy()
        X = np.asarray([np.asarray(r.features, dtype=float) for r in usable])
        if X.ndim != 2 or X.shape[1] != self.dimension:
            raise ValueError("record feature dimension does not match calibrator")
        y = np.asarray([r.co_adaptive_utility - r.structural_prediction for r in usable])
        propensity = np.asarray([r.sampling_propensity for r in usable])
        target = np.asarray([r.target_propensity for r in usable])
        if np.any(propensity <= 0) or np.any(target < 0):
            raise ValueError("propensities must be non-negative and sampling support positive")
        weights = target / propensity
        normal = X.T @ (weights[:, None] * X)
        rhs = X.T @ (weights * y)
        self.coefficients = np.linalg.solve(normal + self.ridge * np.eye(self.dimension), rhs)
        self.fitted_through = current_step
        return self.coefficients.copy()

    def correction(self, features: np.ndarray) -> np.ndarray | float:
        """Evaluate beta^T phi for one or many feature vectors."""

        return np.asarray(features) @ self.coefficients

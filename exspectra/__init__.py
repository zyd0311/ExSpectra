"""Importable ExSpectra building blocks.

The package exposes method components only.  Learner, environment, experiment,
and baseline implementations remain the responsibility of the host project.
"""

from .allocation import sequential_mixture_sample
from .audit import AuditLabel, ProbeDraw, caud_labels, probe_mixture
from .calibration import CalibrationRecord, RidgeCalibrator
from .caud import CAUDComponents, caud_decompose, structural_score
from .displacement import candidate_displacement, reference_anchor
from .influence import SharedCoefficients, build_shared_coefficients

__all__ = [
    "AuditLabel",
    "CAUDComponents",
    "CalibrationRecord",
    "ProbeDraw",
    "RidgeCalibrator",
    "SharedCoefficients",
    "build_shared_coefficients",
    "candidate_displacement",
    "caud_decompose",
    "caud_labels",
    "probe_mixture",
    "reference_anchor",
    "sequential_mixture_sample",
    "structural_score",
]

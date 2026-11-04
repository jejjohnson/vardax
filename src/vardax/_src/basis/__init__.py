"""Reduced-basis control vectors: δx = Φ z with the background term ½ zᵀ Σ_z⁻¹ z.

``LinearBasis`` covers every linear family through its constructors, ``rbf_basis``,
``fourier_basis``, ``wavelet_basis``, ``eof_basis`` and ``linear_basis``;
``CompositeBasis`` stacks bases into one control and ``ModelErrorBasis`` is the
separable space-time basis of a weak-constraint model error.
"""

from .composite import CompositeBasis, composite_basis
from .constructors import eof_basis, fourier_basis, rbf_basis, wavelet_basis
from .linear import LinearBasis, linear_basis
from .model_error import ModelErrorBasis

__all__ = [
    "CompositeBasis",
    "LinearBasis",
    "ModelErrorBasis",
    "composite_basis",
    "eof_basis",
    "fourier_basis",
    "linear_basis",
    "rbf_basis",
    "wavelet_basis",
]

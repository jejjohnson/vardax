r"""Reduced-basis control-vector operators.

Map a coefficient vector $X$ of length ``nbasis`` to a state increment
$\Phi X$ on a grid, with a separable background term
$\tfrac{1}{2} X^\top Q^{-1} X$. Used to parameterise a variational
analysis increment in a low-dimensional space.

One generic module, `LinearBasis`, covers every linear family; the
family lives in the constructor that builds $\Phi$ (delegating the
matrix math to ``geonnax.basis``): :func:`rbf_basis` (Gaussian /
Wendland atoms), :func:`fourier_basis` (Laplacian eigenfunctions /
HSGP), :func:`wavelet_basis` (Haar / Daubechies), :func:`eof_basis`
(data-driven), and the :func:`linear_basis` escape hatch for any
precomputed matrix. `CompositeBasis` stacks several bases into one
control vector with a block-diagonal prior.

Reference: MASSH `mapping/src/basis.py` (`set_basis`, `operg`,
families `GAUSS*` / `BM` / `MIOST` / `WAVELET3D`).
"""

from .composite import CompositeBasis, composite_basis
from .constructors import eof_basis, fourier_basis, rbf_basis, wavelet_basis
from .linear import LinearBasis, linear_basis

__all__ = [
    "CompositeBasis",
    "LinearBasis",
    "composite_basis",
    "eof_basis",
    "fourier_basis",
    "linear_basis",
    "rbf_basis",
    "wavelet_basis",
]

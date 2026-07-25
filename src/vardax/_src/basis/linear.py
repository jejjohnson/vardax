r"""Generic linear reduced basis.

`LinearBasis` wraps *any* precomputed design matrix $\Phi \in
\mathbb{R}^{N \times M}$ together with a diagonal prior
$Q = \mathrm{diag}(\sigma^2)$: ``operg`` maps a coefficient vector
$X$ to the state increment $(\Phi X)$ reshaped onto the grid, and
``prior_inv`` applies $Q^{-1}$ for the separable background term
$\tfrac{1}{2} X^\top Q^{-1} X$.

The basis *family* (RBF, Fourier, wavelet, EOF, …) lives entirely in
the constructor that builds $\Phi$ — see `vardax._src.basis.constructors`,
which delegates the matrix math to ``geonnax.basis``.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as _np


def _as_variance(variance: float | Float[Array, " M"], nbasis: int) -> _np.ndarray:
    """Broadcast ``variance`` to ``(nbasis,)`` and require strict positivity."""
    var_arr = _np.broadcast_to(_np.asarray(variance, dtype=float), (nbasis,)).astype(
        _np.float32
    )
    if not _np.all(_np.isfinite(var_arr)):
        raise ValueError("variance must be finite.")
    if _np.any(var_arr <= 0):
        raise ValueError(
            f"variance must be strictly positive; got min={float(var_arr.min())}."
        )
    return var_arr


class LinearBasis(eqx.Module):
    r"""Generic linear reduced basis with a diagonal-$Q$ prior.

    Build via :func:`linear_basis` (or a family constructor:
    :func:`vardax.rbf_basis`, :func:`vardax.fourier_basis`,
    :func:`vardax.wavelet_basis`, :func:`vardax.eof_basis`); direct
    instantiation expects the precomputed ``phi`` ``(N, M)`` matrix,
    ``variance`` ``(M,)``, and the grid ``output_shape`` with
    ``prod(output_shape) == N``.

    Satisfies `vardax.protocols.ReducedBasis` (`operg` / `prior_inv` /
    `nbasis`). Stationary: ``operg`` ignores the ``t`` argument.
    """

    phi: Float[Array, "N M"]
    variance: Float[Array, " M"]
    output_shape: tuple[int, ...] = eqx.field(static=True)

    def operg(
        self,
        t: float,
        X: Float[Array, " M"],
        state: Float[Array, ...] | None = None,
    ) -> Float[Array, ...]:
        r"""Apply $\Phi X$ — return the increment on ``output_shape``."""
        return (self.phi @ X).reshape(self.output_shape)

    def prior_inv(self, X: Float[Array, " M"]) -> Float[Array, " M"]:
        r"""Apply $Q^{-1} X$ with diagonal $Q = \mathrm{diag}(\sigma^2)$."""
        return X / self.variance

    @property
    def nbasis(self) -> int:
        return int(self.phi.shape[1])


def linear_basis(
    phi: Float[Array, "N M"],
    *,
    variance: float | Float[Array, " M"] = 1.0,
    output_shape: tuple[int, ...] | None = None,
) -> LinearBasis:
    r"""Build a `LinearBasis` from an arbitrary precomputed design matrix.

    The escape hatch for basis families without a dedicated constructor:
    any ``(N, M)`` matrix (spherical harmonics, Slepian, Gabor frames,
    hand-rolled …) becomes a `ReducedBasis`.

    Args:
        phi: Design matrix of shape ``(N, M)``.
        variance: Scalar or per-coefficient prior variance ``(M,)``.
        output_shape: Grid shape the increment is reshaped to; must
            satisfy ``prod(output_shape) == N``. Defaults to ``(N,)``.

    Returns:
        Configured `LinearBasis`.
    """
    phi_arr = jnp.asarray(phi)
    if phi_arr.ndim != 2:
        raise ValueError(f"phi must be 2-D (N, M); got shape {phi_arr.shape}.")
    n, m = int(phi_arr.shape[0]), int(phi_arr.shape[1])
    shape = (n,) if output_shape is None else tuple(int(s) for s in output_shape)
    if math.prod(shape) != n:
        raise ValueError(
            f"prod(output_shape)={math.prod(shape)} does not match phi rows N={n}."
        )
    return LinearBasis(
        phi=phi_arr,
        variance=jnp.asarray(_as_variance(variance, m)),
        output_shape=shape,
    )

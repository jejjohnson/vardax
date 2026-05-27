"""Learned prior models for 4DVarNet.

All priors are implemented as Flax NNX modules with an ``encode`` /
``decode`` interface (bilinear autoencoder) or a simple forward pass that
returns the autoencoder reconstruction.
"""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------


class _BilinearBlock(nnx.Module):
    """Single bilinear block: relu(Ax) * tanh(Bx)."""

    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs) -> None:
        self.linear_a = nnx.Linear(in_features, out_features, rngs=rngs)
        self.linear_b = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        a = self.linear_a(x)
        b = self.linear_b(x)
        return jax.nn.relu(a) * jnp.tanh(b)


# ---------------------------------------------------------------------------
# 1-D priors
# ---------------------------------------------------------------------------


class BilinAEPrior1D(nnx.Module):
    """Bilinear autoencoder prior for 1-D data.

    The encoder maps the input to a low-dimensional latent code; the decoder
    reconstructs the original space.  The prior cost is
    ``||x - decode(encode(x))||^2``.

    Attributes:
        state_dim: Spatial size of the input (``N``).
        latent_dim: Dimensionality of the latent code.
        n_time: Number of time steps (``T``).
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        n_time: int = 1,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.n_time = n_time
        in_features = n_time * state_dim
        self._bilin = _BilinearBlock(in_features, latent_dim, rngs)
        self._decode_dense = nnx.Linear(latent_dim, n_time * state_dim, rngs=rngs)

    def __call__(self, x: Float[Array, "B T N"]) -> Float[Array, "B T N"]:
        b, t, n = x.shape
        x_flat = x.reshape(b, t * n)
        z = self._bilin(x_flat)
        out = self._decode_dense(z)
        return out.reshape(b, t, n)

    def encode(self, x: Float[Array, "B T N"]) -> Float[Array, "B Z"]:
        """Encode input to latent space."""
        b, t, n = x.shape
        x_flat = x.reshape(b, t * n)
        return self._bilin(x_flat)

    def decode(self, z: Float[Array, "B Z"]) -> Float[Array, "B T N"]:
        """Decode latent code to state space."""
        out = self._decode_dense(z)
        return out.reshape(-1, self.n_time, self.state_dim)


class MLPAEPrior1D(nnx.Module):
    """MLP autoencoder prior for 1-D data.

    Attributes:
        state_dim: Spatial size of the input (``N``).
        latent_dim: Dimensionality of the latent code.
        hidden_dim: Hidden layer width.
        n_time: Number of time steps (``T``).
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
        n_time: int = 1,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.n_time = n_time
        in_features = n_time * state_dim
        self.enc1 = nnx.Linear(in_features, hidden_dim, rngs=rngs)
        self.enc2 = nnx.Linear(hidden_dim, latent_dim, rngs=rngs)
        self.dec1 = nnx.Linear(latent_dim, hidden_dim, rngs=rngs)
        self.dec2 = nnx.Linear(hidden_dim, in_features, rngs=rngs)

    def __call__(self, x: Float[Array, "B T N"]) -> Float[Array, "B T N"]:
        b, t, n = x.shape
        x_flat = x.reshape(b, t * n)
        z = jax.nn.relu(self.enc1(x_flat))
        z = self.enc2(z)
        h = jax.nn.relu(self.dec1(z))
        out = self.dec2(h)
        return out.reshape(b, t, n)


class BilinAEPrior2D(nnx.Module):
    """Bilinear autoencoder prior for 2-D data.

    Attributes:
        latent_dim: Dimensionality of the latent code.
        n_time: Number of time steps (``T``).
        height: Spatial height ``H``.
        width: Spatial width ``W``.
    """

    def __init__(
        self,
        latent_dim: int,
        n_time: int,
        height: int,
        width: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.n_time = n_time
        self.height = height
        self.width = width
        in_features = n_time * height * width
        self._bilin = _BilinearBlock(in_features, latent_dim, rngs)
        self._decode_dense = nnx.Linear(latent_dim, in_features, rngs=rngs)

    def __call__(self, x: Float[Array, "B T H W"]) -> Float[Array, "B T H W"]:
        b, t, h, w = x.shape
        x_flat = x.reshape(b, t * h * w)
        z = self._bilin(x_flat)
        out = self._decode_dense(z)
        return out.reshape(b, t, h, w)


class BilinAEPrior2DMultivar(nnx.Module):
    """Bilinear autoencoder prior for 2-D multivariate data.

    Attributes:
        latent_dim: Dimensionality of the latent code.
        n_time: Number of time steps (``T``).
        n_channels: Number of channels ``C``.
        height: Spatial height ``H``.
        width: Spatial width ``W``.
    """

    def __init__(
        self,
        latent_dim: int,
        n_time: int,
        n_channels: int,
        height: int,
        width: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.n_time = n_time
        self.n_channels = n_channels
        self.height = height
        self.width = width
        in_features = n_time * n_channels * height * width
        self._bilin = _BilinearBlock(in_features, latent_dim, rngs)
        self._decode_dense = nnx.Linear(latent_dim, in_features, rngs=rngs)

    def __call__(self, x: Float[Array, "B T C H W"]) -> Float[Array, "B T C H W"]:
        b, t, c, h, w = x.shape
        x_flat = x.reshape(b, t * c * h * w)
        z = self._bilin(x_flat)
        out = self._decode_dense(z)
        return out.reshape(b, t, c, h, w)


# ---------------------------------------------------------------------------
# Identity prior
# ---------------------------------------------------------------------------


class IdentityPrior(nnx.Module):
    """Trivial identity prior: :math:`\\varphi(x) = x`.

    Acts as a pure observation-driven baseline where the prior term
    contributes zero cost regardless of the state, and is also useful
    as a sanity-check building block in tests.
    """

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        """Return the input unchanged.

        Args:
            x: Input array of arbitrary shape.

        Returns:
            The same array ``x``.
        """
        return x


# ---------------------------------------------------------------------------
# Lorenz-63 prior
# ---------------------------------------------------------------------------


class L63Prior(nnx.Module):
    """Learned prior for the Lorenz-63 system.

    A simple MLP autoencoder designed for the 3-dimensional Lorenz-63
    attractor.  The state is treated as a flat vector of length ``3``.

    Attributes:
        latent_dim: Dimensionality of the latent code (default ``3``).
        hidden_dim: Hidden layer width.
        state_dim: Dimensionality of the state vector (default ``3``).
    """

    def __init__(
        self,
        latent_dim: int = 3,
        hidden_dim: int = 32,
        state_dim: int = 3,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.enc1 = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.enc2 = nnx.Linear(hidden_dim, latent_dim, rngs=rngs)
        self.dec1 = nnx.Linear(latent_dim, hidden_dim, rngs=rngs)
        self.dec2 = nnx.Linear(hidden_dim, state_dim, rngs=rngs)

    def __call__(self, x: Float[Array, "B N"]) -> Float[Array, "B N"]:
        z = jnp.tanh(self.enc1(x))
        z = self.enc2(z)
        h = jnp.tanh(self.dec1(z))
        return self.dec2(h)


# ---------------------------------------------------------------------------
# Lorenz-96 priors
# ---------------------------------------------------------------------------


class L96Prior(nnx.Module):
    """Learned prior for the Lorenz-96 system.

    A simple MLP autoencoder designed for the N-dimensional Lorenz-96
    attractor.  The state is treated as a flat vector of length ``N``.

    Attributes:
        latent_dim: Dimensionality of the latent code.
        hidden_dim: Hidden layer width.
        state_dim: Dimensionality of the state vector.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        state_dim: int = 40,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.enc1 = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.enc2 = nnx.Linear(hidden_dim, latent_dim, rngs=rngs)
        self.dec1 = nnx.Linear(latent_dim, hidden_dim, rngs=rngs)
        self.dec2 = nnx.Linear(hidden_dim, state_dim, rngs=rngs)

    def __call__(self, x: Float[Array, "B N"]) -> Float[Array, "B N"]:
        z = jnp.tanh(self.enc1(x))
        z = self.enc2(z)
        h = jnp.tanh(self.dec1(z))
        return self.dec2(h)


class ConvAEPrior1D(nnx.Module):
    """Convolutional autoencoder prior for 1-D spatially-structured data.

    Uses circular (periodic) padding suitable for systems with periodic
    boundary conditions such as Lorenz-96.  Operates on inputs of shape
    ``(B, T, N)`` where ``N`` is the spatial dimension.

    Attributes:
        latent_channels: Number of channels in the latent representation.
        kernel_size: Convolution kernel size (must be a positive odd integer).
        n_time: Number of time steps ``T``; used as the input/output channels
            and validated against the runtime input shape.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        kernel_size: int = 3,
        n_time: int = 1,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}."
            )
        self.kernel_size = kernel_size
        self.n_time = n_time
        ksize = (kernel_size,)
        self._enc_conv = nnx.Conv(
            n_time, latent_channels, kernel_size=ksize, padding="VALID", rngs=rngs
        )
        self._dec_conv = nnx.Conv(
            latent_channels, n_time, kernel_size=ksize, padding="VALID", rngs=rngs
        )

    def __call__(self, x: Float[Array, "B T N"]) -> Float[Array, "B T N"]:
        t = x.shape[1]
        if t != self.n_time:
            raise ValueError(
                f"Input time dimension {t} does not match n_time={self.n_time}."
            )
        # Treat time as channels: (B, N, T)
        h = x.transpose((0, 2, 1))

        # Circular padding for periodic boundaries (pad==0 when kernel_size==1)
        pad = self.kernel_size // 2
        if pad > 0:
            h = jnp.concatenate([h[:, -pad:, :], h, h[:, :pad, :]], axis=1)
        h = self._enc_conv(h)
        h = jax.nn.relu(h)

        # Decode: circular padding + conv back to n_time channels
        if pad > 0:
            h = jnp.concatenate([h[:, -pad:, :], h, h[:, :pad, :]], axis=1)
        h = self._dec_conv(h)

        # Back to (B, T, N)
        return h.transpose((0, 2, 1))

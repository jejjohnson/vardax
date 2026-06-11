"""Learned prior models for FourDVarNet.

All priors are implemented as Equinox modules. They expose a single
``__call__`` returning the reconstructed state; bilinear / MLP /
convolutional autoencoder variants are provided for 1D and 2D spatial
data with optional time and channel axes.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------


class _BilinearBlock(eqx.Module):
    """Single bilinear block: relu(Ax) * tanh(Bx)."""

    linear_a: eqx.nn.Linear
    linear_b: eqx.nn.Linear

    def __init__(
        self, in_features: int, out_features: int, *, key: PRNGKeyArray
    ) -> None:
        key_a, key_b = jax.random.split(key)
        self.linear_a = eqx.nn.Linear(in_features, out_features, key=key_a)
        self.linear_b = eqx.nn.Linear(in_features, out_features, key=key_b)

    def __call__(self, x: Array) -> Array:
        # eqx.nn.Linear operates on a single sample, vmap over batch
        a = jax.vmap(self.linear_a)(x)
        b = jax.vmap(self.linear_b)(x)
        return jax.nn.relu(a) * jnp.tanh(b)


# ---------------------------------------------------------------------------
# 1-D priors
# ---------------------------------------------------------------------------


class BilinAEPrior1D(eqx.Module):
    """Bilinear autoencoder prior for 1-D data.

    The encoder maps the input to a low-dimensional latent code; the decoder
    reconstructs the original space. The prior cost is
    ``||x - decode(encode(x))||^2``.

    Attributes:
        state_dim: Spatial size of the input (``N``).
        latent_dim: Dimensionality of the latent code.
        n_time: Number of time steps (``T``).

    Examples:
        >>> import jax, jax.numpy as jnp
        >>> from vardax import BilinAEPrior1D
        >>> prior = BilinAEPrior1D(
        ...     state_dim=4, latent_dim=2, n_time=3, key=jax.random.PRNGKey(0)
        ... )
        >>> prior(jnp.ones((2, 3, 4))).shape
        (2, 3, 4)
    """

    state_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    n_time: int = eqx.field(static=True)
    _bilin: _BilinearBlock
    _decode_dense: eqx.nn.Linear

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        n_time: int = 1,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.n_time = n_time
        in_features = n_time * state_dim
        key_bilin, key_dec = jax.random.split(key)
        self._bilin = _BilinearBlock(in_features, latent_dim, key=key_bilin)
        self._decode_dense = eqx.nn.Linear(latent_dim, n_time * state_dim, key=key_dec)

    def __call__(self, x: Float[Array, "B T N"]) -> Float[Array, "B T N"]:
        b, t, n = x.shape
        x_flat = x.reshape(b, t * n)
        z = self._bilin(x_flat)
        out = jax.vmap(self._decode_dense)(z)
        return out.reshape(b, t, n)

    def encode(self, x: Float[Array, "B T N"]) -> Float[Array, "B Z"]:
        """Encode input to latent space."""
        b, t, n = x.shape
        x_flat = x.reshape(b, t * n)
        return self._bilin(x_flat)

    def decode(self, z: Float[Array, "B Z"]) -> Float[Array, "B T N"]:
        """Decode latent code to state space."""
        out = jax.vmap(self._decode_dense)(z)
        return out.reshape(-1, self.n_time, self.state_dim)


class MLPAEPrior1D(eqx.Module):
    """MLP autoencoder prior for 1-D data.

    Attributes:
        state_dim: Spatial size of the input (``N``).
        latent_dim: Dimensionality of the latent code.
        hidden_dim: Hidden layer width.
        n_time: Number of time steps (``T``).
    """

    n_time: int = eqx.field(static=True)
    enc1: eqx.nn.Linear
    enc2: eqx.nn.Linear
    dec1: eqx.nn.Linear
    dec2: eqx.nn.Linear

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
        n_time: int = 1,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.n_time = n_time
        in_features = n_time * state_dim
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.enc1 = eqx.nn.Linear(in_features, hidden_dim, key=k1)
        self.enc2 = eqx.nn.Linear(hidden_dim, latent_dim, key=k2)
        self.dec1 = eqx.nn.Linear(latent_dim, hidden_dim, key=k3)
        self.dec2 = eqx.nn.Linear(hidden_dim, in_features, key=k4)

    def __call__(self, x: Float[Array, "B T N"]) -> Float[Array, "B T N"]:
        b, t, n = x.shape
        x_flat = x.reshape(b, t * n)
        z = jax.nn.relu(jax.vmap(self.enc1)(x_flat))
        z = jax.vmap(self.enc2)(z)
        h = jax.nn.relu(jax.vmap(self.dec1)(z))
        out = jax.vmap(self.dec2)(h)
        return out.reshape(b, t, n)


class BilinAEPrior2D(eqx.Module):
    """Bilinear autoencoder prior for 2-D data.

    Attributes:
        latent_dim: Dimensionality of the latent code.
        n_time: Number of time steps (``T``).
        height: Spatial height ``H``.
        width: Spatial width ``W``.
    """

    n_time: int = eqx.field(static=True)
    height: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    _bilin: _BilinearBlock
    _decode_dense: eqx.nn.Linear

    def __init__(
        self,
        latent_dim: int,
        n_time: int,
        height: int,
        width: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.n_time = n_time
        self.height = height
        self.width = width
        in_features = n_time * height * width
        k_bilin, k_dec = jax.random.split(key)
        self._bilin = _BilinearBlock(in_features, latent_dim, key=k_bilin)
        self._decode_dense = eqx.nn.Linear(latent_dim, in_features, key=k_dec)

    def __call__(self, x: Float[Array, "B T H W"]) -> Float[Array, "B T H W"]:
        b, t, h, w = x.shape
        x_flat = x.reshape(b, t * h * w)
        z = self._bilin(x_flat)
        out = jax.vmap(self._decode_dense)(z)
        return out.reshape(b, t, h, w)


class BilinAEPrior2DMultivar(eqx.Module):
    """Bilinear autoencoder prior for 2-D multivariate data.

    Attributes:
        latent_dim: Dimensionality of the latent code.
        n_time: Number of time steps (``T``).
        n_channels: Number of channels ``C``.
        height: Spatial height ``H``.
        width: Spatial width ``W``.
    """

    n_time: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    height: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    _bilin: _BilinearBlock
    _decode_dense: eqx.nn.Linear

    def __init__(
        self,
        latent_dim: int,
        n_time: int,
        n_channels: int,
        height: int,
        width: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.n_time = n_time
        self.n_channels = n_channels
        self.height = height
        self.width = width
        in_features = n_time * n_channels * height * width
        k_bilin, k_dec = jax.random.split(key)
        self._bilin = _BilinearBlock(in_features, latent_dim, key=k_bilin)
        self._decode_dense = eqx.nn.Linear(latent_dim, in_features, key=k_dec)

    def __call__(self, x: Float[Array, "B T C H W"]) -> Float[Array, "B T C H W"]:
        b, t, c, h, w = x.shape
        x_flat = x.reshape(b, t * c * h * w)
        z = self._bilin(x_flat)
        out = jax.vmap(self._decode_dense)(z)
        return out.reshape(b, t, c, h, w)


# ---------------------------------------------------------------------------
# Identity prior
# ---------------------------------------------------------------------------


class IdentityPrior(eqx.Module):
    r"""Trivial identity prior: $\varphi(x) = x$.

    Zero parameters. Useful as a pure obs-driven baseline (the prior
    cost vanishes everywhere) and as a sanity-check building block in
    the linear-Gaussian agreement tests.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vardax import IdentityPrior
        >>> prior = IdentityPrior()
        >>> x = jnp.arange(6.0).reshape(1, 2, 3)
        >>> bool(jnp.all(prior(x) == x))
        True
    """

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        """Return the input unchanged."""
        return x


# ---------------------------------------------------------------------------
# Lorenz priors
# ---------------------------------------------------------------------------


class L63Prior(eqx.Module):
    """Learned prior for the Lorenz-63 system.

    A simple MLP autoencoder designed for the 3-dimensional Lorenz-63
    attractor. The state is treated as a flat vector of length ``3``.

    Attributes:
        latent_dim: Dimensionality of the latent code (default ``3``).
        hidden_dim: Hidden layer width.
        state_dim: Dimensionality of the state vector (default ``3``).
    """

    enc1: eqx.nn.Linear
    enc2: eqx.nn.Linear
    dec1: eqx.nn.Linear
    dec2: eqx.nn.Linear

    def __init__(
        self,
        latent_dim: int = 3,
        hidden_dim: int = 32,
        state_dim: int = 3,
        *,
        key: PRNGKeyArray,
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.enc1 = eqx.nn.Linear(state_dim, hidden_dim, key=k1)
        self.enc2 = eqx.nn.Linear(hidden_dim, latent_dim, key=k2)
        self.dec1 = eqx.nn.Linear(latent_dim, hidden_dim, key=k3)
        self.dec2 = eqx.nn.Linear(hidden_dim, state_dim, key=k4)

    def __call__(self, x: Float[Array, "B N"]) -> Float[Array, "B N"]:
        z = jnp.tanh(jax.vmap(self.enc1)(x))
        z = jax.vmap(self.enc2)(z)
        h = jnp.tanh(jax.vmap(self.dec1)(z))
        return jax.vmap(self.dec2)(h)


class L96Prior(eqx.Module):
    """Learned prior for the Lorenz-96 system.

    A simple MLP autoencoder designed for the N-dimensional Lorenz-96
    attractor. The state is treated as a flat vector of length ``N``.

    Attributes:
        latent_dim: Dimensionality of the latent code.
        hidden_dim: Hidden layer width.
        state_dim: Dimensionality of the state vector.
    """

    enc1: eqx.nn.Linear
    enc2: eqx.nn.Linear
    dec1: eqx.nn.Linear
    dec2: eqx.nn.Linear

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        state_dim: int = 40,
        *,
        key: PRNGKeyArray,
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.enc1 = eqx.nn.Linear(state_dim, hidden_dim, key=k1)
        self.enc2 = eqx.nn.Linear(hidden_dim, latent_dim, key=k2)
        self.dec1 = eqx.nn.Linear(latent_dim, hidden_dim, key=k3)
        self.dec2 = eqx.nn.Linear(hidden_dim, state_dim, key=k4)

    def __call__(self, x: Float[Array, "B N"]) -> Float[Array, "B N"]:
        z = jnp.tanh(jax.vmap(self.enc1)(x))
        z = jax.vmap(self.enc2)(z)
        h = jnp.tanh(jax.vmap(self.dec1)(z))
        return jax.vmap(self.dec2)(h)


class ConvAEPrior1D(eqx.Module):
    """Convolutional autoencoder prior for 1-D spatially-structured data.

    Uses circular (periodic) padding suitable for systems with periodic
    boundary conditions such as Lorenz-96. Operates on inputs of shape
    ``(B, T, N)`` where ``N`` is the spatial dimension.

    Attributes:
        latent_channels: Number of channels in the latent representation.
        kernel_size: Convolution kernel size (must be a positive odd integer).
        n_time: Number of time steps ``T``; used as the input/output channels
            and validated against the runtime input shape.
    """

    kernel_size: int = eqx.field(static=True)
    n_time: int = eqx.field(static=True)
    _enc_conv: eqx.nn.Conv1d
    _dec_conv: eqx.nn.Conv1d

    def __init__(
        self,
        latent_channels: int = 16,
        kernel_size: int = 3,
        n_time: int = 1,
        *,
        key: PRNGKeyArray,
    ) -> None:
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}."
            )
        self.kernel_size = kernel_size
        self.n_time = n_time
        k_enc, k_dec = jax.random.split(key)
        # eqx.nn.Conv1d uses channels-first: (in_channels, length)
        self._enc_conv = eqx.nn.Conv1d(
            in_channels=n_time,
            out_channels=latent_channels,
            kernel_size=kernel_size,
            padding=0,  # we apply circular padding manually
            key=k_enc,
        )
        self._dec_conv = eqx.nn.Conv1d(
            in_channels=latent_channels,
            out_channels=n_time,
            kernel_size=kernel_size,
            padding=0,
            key=k_dec,
        )

    def __call__(self, x: Float[Array, "B T N"]) -> Float[Array, "B T N"]:
        t = x.shape[1]
        if t != self.n_time:
            raise ValueError(
                f"Input time dimension {t} does not match n_time={self.n_time}."
            )
        # eqx Conv1d expects channels-first: (in_channels=T, length=N)
        # x is already (B, T, N) so vmap over the batch dim works directly.
        pad = self.kernel_size // 2

        def _forward(xi: Float[Array, "T N"]) -> Float[Array, "T N"]:
            # Circular padding along spatial axis (axis=1)
            if pad > 0:
                h = jnp.concatenate([xi[:, -pad:], xi, xi[:, :pad]], axis=1)
            else:
                h = xi
            h = self._enc_conv(h)
            h = jax.nn.relu(h)
            if pad > 0:
                h = jnp.concatenate([h[:, -pad:], h, h[:, :pad]], axis=1)
            return self._dec_conv(h)

        return jax.vmap(_forward)(x)

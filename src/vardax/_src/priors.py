"""Learned autoencoder priors x̂ = g_ae(x; θ) for FourDVarNet, as Equinox modules.

Each prior's ``__call__`` returns the reconstruction x̂ of a state window; the
variants are bilinear, MLP and convolutional autoencoders for 1-D and 2-D data.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from vardax._src.errors import ConfigurationError, ShapeError

# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------


class _BilinearBlock(eqx.Module):
    """Bilinear block relu(W_a x) ⊙ tanh(W_b x) applied to each row of a batch."""

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
        # W_a x
        relu_input = jax.vmap(self.linear_a)(x)
        # W_b x
        tanh_input = jax.vmap(self.linear_b)(x)
        # relu(W_a x) ⊙ tanh(W_b x)
        return jax.nn.relu(relu_input) * jnp.tanh(tanh_input)


# ---------------------------------------------------------------------------
# 1-D priors
# ---------------------------------------------------------------------------


class BilinAEPrior1D(eqx.Module):
    """Bilinear autoencoder prior for 1-D data.

        x̂ = g_ae(x; θ) = g_dec(g_enc(x; ν); θ)

    with encoder g_enc(x; ν) = relu(W_a x) ⊙ tanh(W_b x) and a dense decoder
    g_dec(z; θ). The prior cost is 𝒥_p = mean((x − x̂)²).

    Algorithm:
        1. Flatten each sample over time and space.
        2. Encode with the bilinear block to the latent code z.
        3. Decode with a dense layer and reshape to ``(T, N)``.

    Args:
        state_dim: Spatial size of the input (``N``).
        latent_dim: Dimensionality of the latent code.
        n_time: Number of time steps (``T``).
        key: PRNG key for the layer weights.

    Examples:
        >>> import jax, jax.numpy as jnp
        >>> from vardax import BilinAEPrior1D
        >>> prior = BilinAEPrior1D(
        ...     state_dim=4, latent_dim=2, n_time=3, key=jax.random.PRNGKey(0)
        ... )
        >>> prior(jnp.ones((2, 3, 4))).shape
        (2, 3, 4)

        The latent code is low-dimensional, and decoding it is the same as
        calling the prior:

        >>> x = jnp.ones((2, 3, 4))
        >>> prior.encode(x).shape
        (2, 2)
        >>> bool(jnp.allclose(prior.decode(prior.encode(x)), prior(x), atol=1e-6))
        True

        Used as ``prior_fn`` of a solver step, with a pass-through modulator:

        >>> import vardax as vdx
        >>> batch = vdx.Batch1D(input=x, mask=jnp.ones_like(x))
        >>> state = vdx.init_solver_state_1d(batch, hidden_dim=5)
        >>> passthrough = lambda grad, state, lstm: (grad, lstm)
        >>> vdx.solver_step_1d(state, batch, prior, passthrough).x.shape
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
        # vec(x): flatten time and space
        state_flat = einx.id("b t n -> b (t n)", x)
        # z = g_enc(x; ν)
        latent = self._bilin(state_flat)
        # x̂ = g_dec(z; θ)
        decoded = jax.vmap(self._decode_dense)(latent)
        return einx.id("b (t n) -> b t n", decoded, t=x.shape[1])

    def encode(self, x: Float[Array, "B T N"]) -> Float[Array, "B Z"]:
        """Encode a state window to its latent code: z = g_enc(x; ν).

        Args:
            x: State window to encode.

        Returns:
            Latent code z of dimension ``latent_dim`` per sample.
        """
        # vec(x): flatten time and space
        state_flat = einx.id("b t n -> b (t n)", x)
        # z = g_enc(x; ν)
        return self._bilin(state_flat)

    def decode(self, z: Float[Array, "B Z"]) -> Float[Array, "B T N"]:
        """Decode a latent code to a state window: x̂ = g_dec(z; θ).

        Args:
            z: Latent code z.

        Returns:
            Decoded state window x̂.
        """
        # x̂ = g_dec(z; θ)
        decoded = jax.vmap(self._decode_dense)(z)
        return einx.id("b (t n) -> b t n", decoded, t=self.n_time, n=self.state_dim)


class MLPAEPrior1D(eqx.Module):
    """MLP autoencoder prior for 1-D data.

        x̂ = g_ae(x; θ) = g_dec(g_enc(x; ν); θ)

    with g_enc(x; ν) = W₂ relu(W₁ x) and g_dec(z; θ) = W₄ relu(W₃ z), each with a
    bias, applied to the flattened ``(T, N)`` window.

    Algorithm:
        1. Flatten each sample over time and space.
        2. Encode with a ReLU hidden layer and a linear map to the latent code.
        3. Decode with a ReLU hidden layer and a linear map back, then reshape
           to ``(T, N)``.

    Args:
        state_dim: Spatial size of the input (``N``).
        latent_dim: Dimensionality of the latent code.
        hidden_dim: Hidden layer width.
        n_time: Number of time steps (``T``).
        key: PRNG key for the layer weights.

    Examples:
        >>> import jax, jax.numpy as jnp
        >>> from vardax import MLPAEPrior1D
        >>> prior = MLPAEPrior1D(
        ...     state_dim=4,
        ...     latent_dim=2,
        ...     hidden_dim=8,
        ...     n_time=3,
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> prior(jnp.ones((2, 3, 4))).shape
        (2, 3, 4)

        A single time step is the default:

        >>> flat = MLPAEPrior1D(state_dim=4, latent_dim=2, key=jax.random.PRNGKey(0))
        >>> flat(jnp.ones((5, 1, 4))).shape
        (5, 1, 4)
    """

    n_time: int = eqx.field(static=True)
    encoder_hidden: eqx.nn.Linear
    encoder_out: eqx.nn.Linear
    decoder_hidden: eqx.nn.Linear
    decoder_out: eqx.nn.Linear

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
        key_enc1, key_enc2, key_dec1, key_dec2 = jax.random.split(key, 4)
        self.encoder_hidden = eqx.nn.Linear(in_features, hidden_dim, key=key_enc1)
        self.encoder_out = eqx.nn.Linear(hidden_dim, latent_dim, key=key_enc2)
        self.decoder_hidden = eqx.nn.Linear(latent_dim, hidden_dim, key=key_dec1)
        self.decoder_out = eqx.nn.Linear(hidden_dim, in_features, key=key_dec2)

    def __call__(self, x: Float[Array, "B T N"]) -> Float[Array, "B T N"]:
        # vec(x): flatten time and space
        state_flat = einx.id("b t n -> b (t n)", x)
        # relu(W₁ x)
        latent = jax.nn.relu(jax.vmap(self.encoder_hidden)(state_flat))
        # z = W₂ relu(W₁ x)
        latent = jax.vmap(self.encoder_out)(latent)
        # relu(W₃ z)
        hidden = jax.nn.relu(jax.vmap(self.decoder_hidden)(latent))
        # x̂ = W₄ relu(W₃ z)
        decoded = jax.vmap(self.decoder_out)(hidden)
        return einx.id("b (t n) -> b t n", decoded, t=x.shape[1])


class BilinAEPrior2D(eqx.Module):
    """Bilinear autoencoder prior for 2-D data.

        x̂ = g_ae(x; θ) = g_dec(g_enc(x; ν); θ)

    with encoder g_enc(x; ν) = relu(W_a x) ⊙ tanh(W_b x) on the flattened window
    and a dense decoder g_dec(z; θ).

    Algorithm:
        1. Flatten each sample over time and both spatial axes.
        2. Encode with the bilinear block.
        3. Decode with a dense layer and reshape to ``(T, H, W)``.

    Args:
        latent_dim: Dimensionality of the latent code.
        n_time: Number of time steps (``T``).
        height: Spatial height ``H``.
        width: Spatial width ``W``.
        key: PRNG key for the layer weights.

    Examples:
        >>> import jax, jax.numpy as jnp
        >>> from vardax import BilinAEPrior2D
        >>> prior = BilinAEPrior2D(
        ...     latent_dim=2, n_time=3, height=4, width=5, key=jax.random.PRNGKey(0)
        ... )
        >>> prior(jnp.ones((1, 3, 4, 5))).shape
        (1, 3, 4, 5)

        It is the prior inside [`FourDVarNet2D`][vardax.FourDVarNet2D]:

        >>> import vardax as vdx
        >>> model = vdx.FourDVarNet2D(
        ...     n_time=3, height=4, width=5, key=jax.random.PRNGKey(0)
        ... )
        >>> type(model.prior).__name__
        'BilinAEPrior2D'
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
        key_bilin, key_dec = jax.random.split(key)
        self._bilin = _BilinearBlock(in_features, latent_dim, key=key_bilin)
        self._decode_dense = eqx.nn.Linear(latent_dim, in_features, key=key_dec)

    def __call__(self, x: Float[Array, "B T H W"]) -> Float[Array, "B T H W"]:
        _, n_time, height, width = x.shape
        # vec(x): flatten time and both spatial axes
        state_flat = einx.id("b t h w -> b (t h w)", x)
        # z = g_enc(x; ν)
        latent = self._bilin(state_flat)
        # x̂ = g_dec(z; θ)
        decoded = jax.vmap(self._decode_dense)(latent)
        return einx.id("b (t h w) -> b t h w", decoded, t=n_time, h=height, w=width)


class BilinAEPrior2DMultivar(eqx.Module):
    """Bilinear autoencoder prior for 2-D multivariate data.

        x̂ = g_ae(x; θ) = g_dec(g_enc(x; ν); θ)

    with encoder g_enc(x; ν) = relu(W_a x) ⊙ tanh(W_b x) on the window flattened
    over time, channels and space, and a dense decoder g_dec(z; θ).

    Algorithm:
        1. Flatten each sample over time, channel and both spatial axes.
        2. Encode with the bilinear block.
        3. Decode with a dense layer and reshape to ``(T, C, H, W)``.

    Args:
        latent_dim: Dimensionality of the latent code.
        n_time: Number of time steps (``T``).
        n_channels: Number of channels ``C``.
        height: Spatial height ``H``.
        width: Spatial width ``W``.
        key: PRNG key for the layer weights.

    Examples:
        >>> import jax, jax.numpy as jnp
        >>> from vardax import BilinAEPrior2DMultivar
        >>> prior = BilinAEPrior2DMultivar(
        ...     latent_dim=2,
        ...     n_time=3,
        ...     n_channels=2,
        ...     height=4,
        ...     width=5,
        ...     key=jax.random.PRNGKey(0),
        ... )
        >>> prior(jnp.ones((1, 3, 2, 4, 5))).shape
        (1, 3, 2, 4, 5)

        The batch axis is free, so a larger batch works unchanged:

        >>> prior(jnp.ones((4, 3, 2, 4, 5))).shape
        (4, 3, 2, 4, 5)
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
        key_bilin, key_dec = jax.random.split(key)
        self._bilin = _BilinearBlock(in_features, latent_dim, key=key_bilin)
        self._decode_dense = eqx.nn.Linear(latent_dim, in_features, key=key_dec)

    def __call__(self, x: Float[Array, "B T C H W"]) -> Float[Array, "B T C H W"]:
        _, n_time, n_channels, height, width = x.shape
        # vec(x): flatten time, channel and both spatial axes
        state_flat = einx.id("b t c h w -> b (t c h w)", x)
        # z = g_enc(x; ν)
        latent = self._bilin(state_flat)
        # x̂ = g_dec(z; θ)
        decoded = jax.vmap(self._decode_dense)(latent)
        return einx.id(
            "b (t c h w) -> b t c h w",
            decoded,
            t=n_time,
            c=n_channels,
            h=height,
            w=width,
        )


# ---------------------------------------------------------------------------
# Identity prior
# ---------------------------------------------------------------------------


class IdentityPrior(eqx.Module):
    """Trivial identity prior: x̂ = g_ae(x; θ) = x.

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

        The prior cost 𝒥_p = mean((x − x̂)²) is therefore zero, for any shape:

        >>> y = jnp.ones((4,))
        >>> float(jnp.mean((y - prior(y)) ** 2))
        0.0

        It satisfies the [`Prior`][vardax.Prior] protocol:

        >>> import vardax as vdx
        >>> isinstance(prior, vdx.Prior)
        True
    """

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        """Return the input unchanged: x̂ = x.

        Args:
            x: State to project.

        Returns:
            The same array, x̂ = x.
        """
        return x


# ---------------------------------------------------------------------------
# Lorenz priors
# ---------------------------------------------------------------------------


class L63Prior(eqx.Module):
    """Learned prior for the Lorenz-63 system.

        x̂ = g_ae(x; θ) = g_dec(g_enc(x; ν); θ)

    with g_enc(x; ν) = W₂ tanh(W₁ x) and g_dec(z; θ) = W₄ tanh(W₃ z), each with a
    bias. The state is treated as a flat vector of length ``3``.

    Algorithm:
        1. Encode with a tanh hidden layer and a linear map to the latent code.
        2. Decode with a tanh hidden layer and a linear map back to the state.

    Args:
        latent_dim: Dimensionality of the latent code.
        hidden_dim: Hidden layer width.
        state_dim: Dimensionality of the state vector.
        key: PRNG key for the layer weights.

    Examples:
        >>> import jax, jax.numpy as jnp
        >>> from vardax import L63Prior
        >>> prior = L63Prior(key=jax.random.PRNGKey(0))
        >>> prior(jnp.ones((5, 3))).shape
        (5, 3)

        The latent and hidden widths are free parameters:

        >>> small = L63Prior(latent_dim=2, hidden_dim=4, key=jax.random.PRNGKey(0))
        >>> small(jnp.zeros((1, 3))).shape
        (1, 3)
    """

    encoder_hidden: eqx.nn.Linear
    encoder_out: eqx.nn.Linear
    decoder_hidden: eqx.nn.Linear
    decoder_out: eqx.nn.Linear

    def __init__(
        self,
        latent_dim: int = 3,
        hidden_dim: int = 32,
        state_dim: int = 3,
        *,
        key: PRNGKeyArray,
    ) -> None:
        key_enc1, key_enc2, key_dec1, key_dec2 = jax.random.split(key, 4)
        self.encoder_hidden = eqx.nn.Linear(state_dim, hidden_dim, key=key_enc1)
        self.encoder_out = eqx.nn.Linear(hidden_dim, latent_dim, key=key_enc2)
        self.decoder_hidden = eqx.nn.Linear(latent_dim, hidden_dim, key=key_dec1)
        self.decoder_out = eqx.nn.Linear(hidden_dim, state_dim, key=key_dec2)

    def __call__(self, x: Float[Array, "B N"]) -> Float[Array, "B N"]:
        # tanh(W₁ x)
        latent = jnp.tanh(jax.vmap(self.encoder_hidden)(x))
        # z = W₂ tanh(W₁ x)
        latent = jax.vmap(self.encoder_out)(latent)
        # tanh(W₃ z)
        hidden = jnp.tanh(jax.vmap(self.decoder_hidden)(latent))
        # x̂ = W₄ tanh(W₃ z)
        return jax.vmap(self.decoder_out)(hidden)


class L96Prior(eqx.Module):
    """Learned prior for the Lorenz-96 system.

        x̂ = g_ae(x; θ) = g_dec(g_enc(x; ν); θ)

    with g_enc(x; ν) = W₂ tanh(W₁ x) and g_dec(z; θ) = W₄ tanh(W₃ z), each with a
    bias. The state is treated as a flat vector of length ``N``.

    Algorithm:
        1. Encode with a tanh hidden layer and a linear map to the latent code.
        2. Decode with a tanh hidden layer and a linear map back to the state.

    Args:
        latent_dim: Dimensionality of the latent code.
        hidden_dim: Hidden layer width.
        state_dim: Dimensionality of the state vector.
        key: PRNG key for the layer weights.

    Examples:
        >>> import jax, jax.numpy as jnp
        >>> from vardax import L96Prior
        >>> prior = L96Prior(state_dim=8, latent_dim=4, key=jax.random.PRNGKey(0))
        >>> prior(jnp.ones((3, 8))).shape
        (3, 8)

        The defaults target the standard 40-variable system:

        >>> L96Prior(key=jax.random.PRNGKey(0))(jnp.ones((1, 40))).shape
        (1, 40)
    """

    encoder_hidden: eqx.nn.Linear
    encoder_out: eqx.nn.Linear
    decoder_hidden: eqx.nn.Linear
    decoder_out: eqx.nn.Linear

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        state_dim: int = 40,
        *,
        key: PRNGKeyArray,
    ) -> None:
        key_enc1, key_enc2, key_dec1, key_dec2 = jax.random.split(key, 4)
        self.encoder_hidden = eqx.nn.Linear(state_dim, hidden_dim, key=key_enc1)
        self.encoder_out = eqx.nn.Linear(hidden_dim, latent_dim, key=key_enc2)
        self.decoder_hidden = eqx.nn.Linear(latent_dim, hidden_dim, key=key_dec1)
        self.decoder_out = eqx.nn.Linear(hidden_dim, state_dim, key=key_dec2)

    def __call__(self, x: Float[Array, "B N"]) -> Float[Array, "B N"]:
        # tanh(W₁ x)
        latent = jnp.tanh(jax.vmap(self.encoder_hidden)(x))
        # z = W₂ tanh(W₁ x)
        latent = jax.vmap(self.encoder_out)(latent)
        # tanh(W₃ z)
        hidden = jnp.tanh(jax.vmap(self.decoder_hidden)(latent))
        # x̂ = W₄ tanh(W₃ z)
        return jax.vmap(self.decoder_out)(hidden)


class ConvAEPrior1D(eqx.Module):
    """Convolutional autoencoder prior for 1-D spatially-structured data.

        x̂ = g_ae(x; θ) = g_dec(g_enc(x; ν); θ)

    with g_enc(x; ν) = relu(W_e ⋆ pad(x)) and g_dec(z; θ) = W_d ⋆ pad(z), where
    ⋆ is the convolution and pad(·) circular (periodic) padding of the spatial
    axis, suited to periodic systems such as Lorenz-96.

    Algorithm:
        1. Pad the spatial axis periodically and apply the encoder convolution.
        2. Apply a ReLU.
        3. Pad periodically again and apply the decoder convolution back to
           ``n_time`` channels.

    Args:
        latent_channels: Number of channels in the latent representation.
        kernel_size: Convolution kernel size (must be a positive odd integer).
        n_time: Number of time steps ``T``; used as the input/output channels
            and validated against the runtime input shape.
        key: PRNG key for the convolution weights.

    Raises:
        ConfigurationError: If ``kernel_size`` is not a positive odd integer.
        ShapeError: On call, if the input's time dimension differs from
            ``n_time``.

    Examples:
        >>> import jax, jax.numpy as jnp
        >>> from vardax import ConvAEPrior1D
        >>> prior = ConvAEPrior1D(n_time=3, key=jax.random.PRNGKey(0))
        >>> prior(jnp.ones((2, 3, 10))).shape
        (2, 3, 10)

        Periodic padding makes the prior shift-equivariant on the ring:

        >>> x = jax.random.normal(jax.random.PRNGKey(1), (1, 3, 10))
        >>> shifted = prior(jnp.roll(x, 2, axis=-1))
        >>> bool(jnp.allclose(shifted, jnp.roll(prior(x), 2, axis=-1), atol=1e-5))
        True

        An even kernel is rejected:

        >>> ConvAEPrior1D(
        ...     kernel_size=4, key=jax.random.PRNGKey(0)
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ConfigurationError: kernel_size must be a positive odd integer, got 4.
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
            raise ConfigurationError(
                f"kernel_size must be a positive odd integer, got {kernel_size}."
            )
        self.kernel_size = kernel_size
        self.n_time = n_time
        key_enc, key_dec = jax.random.split(key)
        # eqx.nn.Conv1d uses channels-first: (in_channels, length)
        self._enc_conv = eqx.nn.Conv1d(
            in_channels=n_time,
            out_channels=latent_channels,
            kernel_size=kernel_size,
            padding=0,  # we apply circular padding manually
            key=key_enc,
        )
        self._dec_conv = eqx.nn.Conv1d(
            in_channels=latent_channels,
            out_channels=n_time,
            kernel_size=kernel_size,
            padding=0,
            key=key_dec,
        )

    def __call__(self, x: Float[Array, "B T N"]) -> Float[Array, "B T N"]:
        n_time = x.shape[1]
        if n_time != self.n_time:
            raise ShapeError(
                f"Input time dimension {n_time} does not match n_time={self.n_time}."
            )
        # eqx Conv1d expects channels-first: (in_channels=T, length=N)
        # x is already (B, T, N) so vmap over the batch dim works directly.
        pad = self.kernel_size // 2

        def _forward(sample: Float[Array, "T N"]) -> Float[Array, "T N"]:
            # circular padding on the spatial axis only: the grid is periodic
            periodic_pad = ((0, 0), (pad, pad))
            # pad(x)
            hidden = jnp.pad(sample, periodic_pad, mode="wrap")
            # W_e ⋆ pad(x)
            hidden = self._enc_conv(hidden)
            # z = relu(W_e ⋆ pad(x))
            hidden = jax.nn.relu(hidden)
            # pad(z)
            hidden = jnp.pad(hidden, periodic_pad, mode="wrap")
            # x̂ = W_d ⋆ pad(z)
            return self._dec_conv(hidden)

        return jax.vmap(_forward)(x)

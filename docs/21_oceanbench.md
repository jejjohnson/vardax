# OceanBench: SSH Applications

Ported and condensed from the `mfourdvar` OceanBench notes
(`content/oceanbench/`); the flagship application context that
motivated this library.

> "OceanBench is a framework for co-designing learning-driven
> high-level experiments from ocean models, reanalysis, and
> observations. It consists of an end-to-end framework for piping data
> from its raw form to an ML-ready state and from model outputs to
> interpretable quantities."

Chapter [16](16_ssh_example.md) walks through *one* SSH
reconstruction end-to-end; this chapter is the surrounding benchmark
context — where the data come from, how the experiment ladder is
structured, and which OceanBench task each vardax component serves.

## Why a benchmark framework

The ML method is the smallest part of an operational chain. The parts
that dominate the effort are getting raw ocean data into an ML-ready
form, evaluating a proposed tool against operationally meaningful
metrics, and inserting it into an operational pipeline —
reproducibly, so that non-experts in operations can contribute a
method without rebuilding the plumbing. OceanBench standardises that
chain (raw observations / simulations / reanalyses → ML-ready patches
→ method → interpretable metrics); vardax slots in as the *method*
box, with `pipekit` supplying the pipeline substrate
([design/pipekit_composition.md](design/pipekit_composition.md)).

## Data

Three families of product, all reachable through the Copernicus
Marine Data Store:

- **Observations** — along-track satellite altimetry (NADIR
  pencil-beam tracks; SWOT wide-swath), plus in-situ profiles.
  Sparse, gappy, noisy: the `y` of every cost in this library.
- **Reanalysis** — GLORYS12V1, the CMEMS global eddy-resolving
  reanalysis at $1/12^\circ$ with 50 vertical levels (1993–2020),
  assimilating along-track altimetry, SST, sea ice, and in-situ
  T/S profiles. Reanalyses blend model and observations, which makes
  them the standard training target for learned methods.
- **Free-run simulation** — NEMO runs without assimilation; the
  ground truth generator for observing-system simulation experiments
  (OSSE), where "truth" must be known exactly.

Two pragmatic ladders recur throughout: **regions** (Gulf Stream →
Mediterranean → North Atlantic → global) and **frequency** (daily maps
before hourly), each solving a drastically simpler problem before
scaling up — with transfer learning carrying weights up the rungs.

## The experiment ladder

**OSSE** (observing-system *simulation* experiments) sample synthetic
observations from a known simulated truth, so reconstruction error is
exactly measurable. The editions add data sources incrementally —
ablate to learn which source carries the signal:

| Edition | Observations | What it tests |
|---|---|---|
| OSSE NADIR | simulated NADIR altimetry tracks (from a NEMO run) | baseline sparse-track interpolation |
| OSSE SWOT | NADIR + SWOT swaths | higher spatial resolution, lower temporal; much higher data volume |
| OSSE NADIR + SWOT + SST | + sea-surface temperature | multivariate synergy — SSH and SST are dynamically coupled, and SST is abundant with few gaps |

**OSE** (observing-system experiments) then rerun the winning
configuration on *real* NADIR altimetry, where truth is unknown and
evaluation falls back to withheld tracks and physical diagnostics.

## Tasks

### SSH interpolation

Fill the gaps between altimeter tracks to produce daily gap-free SSH
maps — the first OceanBench edition, and the task chapter
[16](16_ssh_example.md) implements three ways (OI baseline,
`IncrementalFourDVar`, learned `FourDVarNet`). The learned
configuration in brief:

The block below runs an untrained model on a toy SSH field, $\eta(t, y, x) = \sin(x - 0.2 t) \cos(y)$ on $16 \times 16$ cells, observed along diagonal ground tracks; a real configuration uses $128 \times 128$ patches, `latent_dim=32`, `hidden_dim=48` and 15 solver steps, and is trained before use (chapter [16](16_ssh_example.md)).

```python
import einx
import jax
import jax.numpy as jnp
import vardax as vdx

# Shapes: B = 1 patch; T = 5 days; H = W = 16 lat-lon cells

key_model, key_noise = jax.random.split(jax.random.PRNGKey(0))
cells = jnp.arange(16)  # (H,)
grid = jnp.linspace(0.0, 2.0 * jnp.pi, 16)  # (H,)
days = jnp.arange(5)  # (T,)

# x − 0.2 t
phase = einx.subtract("w, t -> t w", grid, 0.2 * days)  # (T, W)
# η(t, y, x) = sin(x − 0.2 t) cos(y), a toy SSH field drifting west
ssh_truth = einx.multiply(  # (T, H, W)
    "t w, h -> t h w", jnp.sin(phase), jnp.cos(grid)
)
# i + j − 3t, the along-track coordinate of a diagonal ground track
track_coord = einx.subtract(  # (T, H, W)
    "h w, t -> t h w", einx.add("h, w -> h w", cells, cells), 3 * days
)
# m = 1 where i + j − 3t ≡ 0 (mod 8), 0 in the gaps
track_mask = (track_coord % 8 == 0).astype(jnp.float32)  # (T, H, W)
# y = m ⊙ (η + 0.02 ε), zero-filled gaps
ssh_tracks = track_mask * (  # (T, H, W)
    ssh_truth + 0.02 * jax.random.normal(key_noise, (5, 16, 16))
)

# today (vardax)
model = vdx.FourDVarNet2D(
    n_time=5,  # T: days in the assimilation window
    height=16,  # H, W: regional lat-lon patch
    width=16,
    latent_dim=8,
    hidden_dim=8,
    n_solver_steps=3,
    key=key_model,
)
# today (vardax)
batch = vdx.Batch2D(
    input=einx.id("t h w -> 1 t h w", ssh_tracks),  # (T, H, W) → (B, T, H, W)
    mask=einx.id("t h w -> 1 t h w", track_mask),  # 1 on-track, 0 in gaps
    target=einx.id("t h w -> 1 t h w", ssh_truth),  # OSSE only
)
# x* = argminₓ U(x), unrolled by the learned solver
ssh_maps = model(batch)  # → (B, T, H, W)
observed_fraction = einx.mean("t h w ->", track_mask)  # → ()
print(f"observed fraction {observed_fraction:.3f}, maps {ssh_maps.shape}")
```

It prints `observed fraction 0.125, maps (1, 5, 16, 16)`.

Real altimetry products carry `NaN` in the gaps rather than zeros —
strip them with the NaN-safe observation costs
([`obs_cost_2d(..., nan_to_num=True)`](xref:api#vardax.obs_cost_2d)) or a
masking preprocessing step, per chapter [20](20_uncertainty.md).

### SSH forecasting

Propagate currently-assimilated maps forward in time (1 / 5 / 10-day
leads), training on historical reanalysis so the model captures the
blended physics-plus-observations signal. In vardax terms this is the
`ForwardModel` seam's job: a learned surrogate trained on GLORYS
drives a [`DACycle`](https://jejjohnson.github.io/vardax/reference/cycle/) forward between assimilation
windows, and the analysis methods correct it as observations arrive.

### SSH surrogates

Learn the flow map itself — emulate the simulation at 1 / 6 / 12 /
24-hour steps from free-run data. Surrogates are the component-level
answer to non-differentiable GCMs (chapter
[19](19_physical_models.md)): once trained, a surrogate is a
differentiable `ForwardModel`, usable as the forward operator of
[`StrongFourDVar`](06_strong_4dvar.md) or wrapped as a dynamical
prior.

## Status

The interpolation task is fully exercised by the code in chapter
[16](16_ssh_example.md) and the 2-D demo notebook; a dedicated
OceanBench SSH-interpolation tutorial notebook is deferred (tracked
with the Phase-3 tutorial issues of the mfourdvar migration epic).

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

```python
import jax
from vardax import Batch2D, FourDVarNet2D

model = FourDVarNet2D(
    n_time=5,                # T: days in the assimilation window
    height=128,              # H, W: regional lon-lat patch
    width=128,
    latent_dim=32,
    hidden_dim=48,
    n_solver_steps=15,
    key=jax.random.PRNGKey(0),
)

batch = Batch2D(
    input=ssh_tracks,        # (B, T, H, W), zero-filled gaps
    mask=track_mask,         # 1 on-track, 0 in gaps
    target=ssh_truth,        # OSSE only
)
ssh_maps = model(batch)      # (B, T, H, W) gap-free reconstruction
```

Real altimetry products carry `NaN` in the gaps rather than zeros —
strip them with the NaN-safe observation costs
([`obs_cost_2d(..., nan_to_num=True)`](api/costs_priors.md)) or a
masking preprocessing step, per chapter [20](20_uncertainty.md).

### SSH forecasting

Propagate currently-assimilated maps forward in time (1 / 5 / 10-day
leads), training on historical reanalysis so the model captures the
blended physics-plus-observations signal. In vardax terms this is the
`ForwardModel` seam's job: a learned surrogate trained on GLORYS
drives a [`DACycle`](api/cycle.md) forward between assimilation
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

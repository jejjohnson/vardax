# API Reference

vardax is JAX-native variational data assimilation: seven peer analysis
methods — from [Optimal Interpolation](../04_oi_blue.md) through
[learned 4DVarNet](../09_4dvarnet.md) — plus the observation operators,
priors, posterior adapters, and training loops that surround them. The
reference is organised by area rather than dumped as one flat page:

| Section | What's inside |
|---------|---------------|
| [Models](models.md) | The seven peer analysis methods — `OptimalInterpolation`, `ThreeDVar`, `StrongFourDVar`, `WeakFourDVar`, `IncrementalFourDVar`, `FourDVarNet1D`, `FourDVarNet2D` |
| [Amortized Inference](amortized.md) | `AmortizedPosterior` and its observation encoders and posterior heads (regression, conditional flow, score diffusion) |
| [Observation Operators](obs_operators.md) | `LinearObs`, `MaskedIdentity`, `AveragingKernel`, multi-instrument fusion, and the instrument registry |
| [Posterior Covariance](posterior.md) | The `Posterior` container and the Laplace / Gauss-Newton / ensemble adapters that build it |
| [Costs, Priors & Solvers](costs_priors.md) | Variational cost functions, learned and analytic priors, and the 4DVarNet inner-loop solver functions |
| [Training & Adjoints](training.md) | Loss functions and train/eval steps, adjoint strategies, and ConvLSTM gradient modulators |
| [Cycle Integration](cycle.md) | `VarDACycle` / `VarSmootherCycle` factories over pipekit-cycle |
| [Protocols & Types](protocols.md) | The structural Protocols every component satisfies, plus the batch and LSTM-state type aliases |
| [Utilities & Diagnostics](utils.md) | Lorenz-96 simulation, six-step validation gates (SBC and friends), and plotting helpers |

## Import conventions

Everything documented here is importable from the top-level namespace:

```python
import vardax as vdx

model = vdx.ThreeDVar(...)
step = model.as_analysis_step()
```

Three public submodules are also bound into the package namespace for the
access patterns used in the design docs — `vdx.amortized.AmortizedPosterior`,
`vdx.cycle.VarDACycle`, and `vdx.adjoints.OneStepAdjoint` resolve after a
plain `import vardax`. Everything under `vardax._src` is private;
import paths there may change without notice.

## Division of labour

vardax deliberately implements only the variational-DA layer:

- **Linear algebra delegates to [gaussx](https://jejjohnson.github.io/gaussx/).**
  Covariances are lazy lineax operators; inverses, diagonals, and ensemble
  covariances route through `gaussx.inv`, `gaussx.diag`, and
  `gaussx.ensemble_covariance` rather than being materialised here.
- **Orchestration delegates to [pipekit-cycle](https://github.com/jejjohnson/pipekit).**
  Every model exposes `.as_analysis_step()`, returning an object that
  satisfies the pipekit `AnalysisStep` Protocol, and the
  [Cycle Integration](cycle.md) factories wire whole assimilation cycles
  together.

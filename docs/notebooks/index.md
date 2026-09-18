# Tutorials

End-to-end, runnable walkthroughs of vardax on the Lorenz-63 and
Lorenz-96 testbeds. Notebooks 01–08 cover state estimation with 4DVar and
4DVarNet; 09–11 port the parameter-estimation, bilevel-optimisation and
gradient-learning chapters of the legacy *mfourdvar* book. Each page is a [jupytext](https://jupytext.readthedocs.io/)
percent-format `.py` source under `docs/notebooks/`, executed when the
documentation is built, so the outputs you see here are produced by the
released code.

| # | Notebook | What it shows |
|---|---|---|
| 01 | [Model-based 4DVar on L63](01_model_based_4dvar_L63.py) | Classical strong-constraint 4DVar (`StrongFourDVar`) with the Lorenz-63 ODE as forward model: background vs analysis |
| 02 | [Unrolling vs fixed-point on L63](02_unrolling_vs_fixedpoint_L63.py) | The unrolled learned solver vs the prior-only fixed-point solver, plus warm-start initialisation |
| 03 | [4DVarNet end-to-end on L63](03_4dvarnet_L63.py) | Training `FourDVarNet1D` with the demo training loop |
| 04 | [4DVarNet 2-D demo](04_4dvarnet_2d_demo.py) | `FourDVarNet2D` on synthetic spatiotemporal fields |
| 05 | [End-to-end L63 pipeline](05_end_to_end_L63.py) | Simulation, patching, masking, standardisation, training |
| 06 | [End-to-end L96 pipeline](06_end_to_end_L96.py) | The same pipeline on Lorenz-96 |
| 07 | [Prior pre-training on L63](07_prior_pretraining_L63.py) | Two-stage training: pre-train the prior, then fine-tune |
| 08 | [Classical 4DVar vs 4DVarNet](08_classical_4dvar_vs_4dvarnet_L63.py) | Gradient descent on the variational cost vs the learned solver |
| 09 | [Parameter estimation on L63](09_param_estimation_L63.py) | Learning ODE parameters (and the initial state) with `DynTrajectory` and `strong_variational_cost` |
| 10 | [Bilevel optimisation on L63](10_bilevel_opt_L63.py) | Learning the cost weights by differentiating through the inner 4DVar solve |
| 11 | [Learning the gradient update](11_gradient_learning_L63.py) | Training only the `ConvLSTMGradMod1D` against vanilla gradient descent, with a solver-steps ablation |

## Running locally

```bash
uv sync --all-extras --group docs
uv run jupytext --to notebook docs/notebooks/01_model_based_4dvar_L63.py
uv run jupyter lab docs/notebooks/01_model_based_4dvar_L63.ipynb
```

Committed `.ipynb` files are not allowed in this repository (a pre-commit
hook rejects them); edit the `.py` source and let `jupytext --sync` keep a
local `.ipynb` in step.

## See also

- [Chapter 15 — Lorenz examples](../15_lorenz_examples.md) for the
  mathematical setup behind these notebooks.
- [Chapter 19 — Physical models & ODE priors](../19_physical_models.md)
  for the dynamical priors used in the model-based examples.

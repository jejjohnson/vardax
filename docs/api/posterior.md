# Posterior Covariance

A variational analysis returns a point estimate; this layer attaches the
uncertainty around it. The design is deliberately *lazy and matrix-free*:
a posterior adapter never materialises the covariance matrix. Instead it
builds a lineax operator for the Hessian (or precision) at the analysis
point, and inverses, diagonals (pointwise variances), and ensemble
estimates are delegated to
[gaussx](https://jejjohnson.github.io/gaussx/) — `gaussx.inv`,
`gaussx.diag`, and `gaussx.ensemble_covariance` — so structured and
iterative solves come for free. See
[Posterior Covariance](https://jejjohnson.github.io/vardax/posterior-covariance/) in the Mathematical
Reference for derivations and when each approximation is trustworthy.

Adapters satisfy the [`PosteriorAdapter`](protocols.md) Protocol and all
produce the same `Posterior` container, so downstream diagnostics (e.g. the
[posterior-agreement gate](utils.md)) are adapter-agnostic. `LaplaceCovariance` and
`GaussNewtonHessian` both build the inverse Gauss–Newton Hessian at the
mode, matrix-free; for a 4DVar model it runs through the model's
tangent-linear and adjoint rollouts, so pass the model and the window's
batch. `GaussNewtonHessian` exposes the Krylov budget. Choose
`EnsembleCovariance` when samples are cheaper than curvature or the
posterior is not Gaussian.

## Container

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [Posterior]

## Adapters

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [LaplaceCovariance, GaussNewtonHessian, EnsembleCovariance]

## Likelihoods

::: vardax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [GaussianMarkLikelihood]

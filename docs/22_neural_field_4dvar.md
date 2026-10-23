# 22 · Neural-Field 4D-Var

Strong-constraint 4D-Var ([chapter 6](06_strong_4dvar.md)) controls the
initial state and reaches every later time by rolling a model forward.
`SpaceTimeFieldVar` drops the rollout: the whole assimilation window is
one continuous field

$$
u(t, s) = f_\theta(t, s),
$$

a coordinate network whose weights $\theta$ are the control, fitted to
the observations with the dynamics as a *soft* constraint (Oh 2025).
The analysis is a function, not a grid: it can be evaluated at any
$(t, s)$ and differentiated there.

## 22.1 Running example: SSH from along-track altimetry

Satellite altimeters sample sea-surface height (SSH) along narrow
ground tracks, a few thousand scattered $(t_k, x_k, y_k, \eta_k)$
points per day with errors $\sigma_k$, and nothing in between. A
gridded 4D-Var first interpolates those points onto the model grid
([`GridPointObs`](api/obs_operators.md)) and then needs a background
covariance $B$ to spread their information. A neural field reads the
tracks as they are ([`PointObsBatch`](api/protocols.md)), and the
quasi-geostrophic equations, imposed at collocation points, carry the
information along the flow instead of $B$.

## 22.2 The cost

With observations $y_k$ at $(t_k, s_k)$ with errors $\sigma_k$ and
validity flags $m_k$ (zero on padding rows), a PDE operator
$\mathcal N$ with $\mathcal N[u] = 0$ for the true flow, collocation
points $c_1, \dots, c_{N_c}$ and weights $\lambda, \alpha \ge 0$,

$$
J(\theta) =
  \underbrace{\tfrac{1}{2}\sum_k m_k
    \frac{\big(y_k - f_\theta(t_k, s_k)\big)^2}{\sigma_k^2}}_{\texttt{point\_obs\_cost}}
+ \underbrace{\frac{\lambda}{2 N_c}\sum_{j=1}^{N_c}
    \big\lVert \mathcal N[f_\theta](c_j) \big\rVert^2}_{\texttt{ResidualPenalty}}
+ \underbrace{\frac{\alpha}{2}\lVert\theta\rVert^2}_{\text{weight decay}} .
$$

There is **no background term**. What plays the role of $B$ is the
network's spectral bias: gradient descent from a random initialisation
fits smooth, low-frequency functions first (Rahaman et al. 2019), and
the weight decay keeps $\theta$ near the origin. The PDE term is the
physics-informed loss of Raissi et al. (2019); its weight $\lambda$
trades fidelity to the equations against fidelity to the data.

Each term is a merged vardax building block:

| Term | Object |
|---|---|
| $f_\theta$ at the observations, $m_k$-masked and $\sigma_k$-whitened | `FieldPointObs` + `point_obs_cost` ([chapter 11](11_observation_operators.md)) |
| $\mathcal N[f_\theta]$ at collocation points | `ResidualPenalty(residual_fn, collocation, weight, sampler)`; residuals in `vardax.examples` |
| control split, $\tfrac{\alpha}{2}\lVert\theta\rVert^2$ | `NeuralFieldParameterization` (its `filter_spec` keeps frozen arrays out of $\theta$) |
| $\min_\theta J$ | `OptaxMinimiser` (Adam, early stopping, optional L-BFGS polish) |

## 22.3 Usage

Coordinates put time first, $(t, x, y)$, for the field, the observations
and the collocation points alike (the convention of the residuals in
`vardax.examples`):

```python
import geonnax, jax, optax, vardax
from vardax.examples import navier_stokes_vorticity_2d

ana = vardax.SpaceTimeFieldVar(
    field=geonnax.SIREN.init(3, 64, 1, depth=3, key=k_init),   # ψ(t, x, y)
    residual=vardax.ResidualPenalty(
        residual_fn=navier_stokes_vorticity_2d(1e-3),
        collocation=None,
        sampler=sample_box,          # key -> (N_c, 3)
        weight=1e-2,
    ),
    weight_decay=1e-6,
    minimiser=vardax.OptaxMinimiser(optax.adam(1e-3), max_steps=20_000),
)
result = ana(obs, key=key)            # obs: PointObsBatch, coords (t, x, y)
result.evaluate(points)               # u at any (N, 3) space-time points
x0 = result.initial_condition(grid)   # u(t₀, ·) on an (N, 2) grid
step = ana.as_analysis_step(grid)     # AnalysisStep returning u(t₀, grid);
                                      # without a grid it returns `result`
```

`result.grad_norm` reports $\lVert\nabla J(\theta^\star)\rVert$: a
first-order optimiser stopped by its step budget is not at a stationary
point, which matters if you differentiate through the analysis.

A **separable** field (one small network per axis, combined by a
low-rank product, as in Oh 2025) evaluates a whole space-time grid in
one pass and is much faster for dense collocation; any single-example
`eqx.Module` of $(t, s)$ works as `field`.

### Collocation points

A `ResidualPenalty` with a `sampler` draws its points **once per
analysis**, from `key`, and holds them fixed while the minimiser runs.
The minimiser assumes a deterministic cost: its patience test and
best-iterate tracking compare costs across steps, and the implicit
adjoint differentiates the stationarity condition
$\nabla J(\theta^\star) = 0$. Resampling every step (a stochastic
gradient of the expected residual) would break all three, so it is left
to a custom training loop; draw enough points to resolve the residual,
or refit with fresh keys.

### Uncertainty: a deep ensemble

`ana.ensemble(obs, field_init, n_members, key=key)` refits from
`n_members` random initialisations (`field_init: key -> field`), each
with its own collocation draw, under `eqx.filter_vmap`, and returns one
result whose arrays carry a leading member axis. The spread of the
members is the deep-ensemble heuristic of Lakshminarayanan et al. (2017):
small where observations pin the field down, large away from them
(`tests/test_spacetime_field.py` checks both). A Laplace push-forward
$J \Sigma_\theta J^\top$ with $J = \partial u/\partial\theta$ and
$\Sigma_\theta$ a Gauss–Newton inverse in weight space is deferred, as in
[design/latent_da.md](design/latent_da.md) §13.

## 22.4 The linear-Gaussian limit

If $f_\theta(t, s) = \phi(t, s)^\top w$ is linear in fixed features and
$\lambda = 0$, the cost is

$$
J(w) = \tfrac12 \lVert y - \Phi w \rVert^2_{R^{-1}}
     + \tfrac{\alpha}{2}\lVert w \rVert^2 ,
\qquad \Phi_{km} = \phi_m(t_k, s_k),\; R = \mathrm{diag}(\sigma_k^2),
$$

the 3D-Var cost for the coefficients with $B = \alpha^{-1} I$, $x_b = 0$
and $H = \Phi$; its minimiser is the OI / BLUE analysis
$w^\star = \alpha^{-1}\Phi^\top(\alpha^{-1}\Phi\Phi^\top + R)^{-1} y$,
the posterior mean of Bayesian linear regression. The test suite checks
that `SpaceTimeFieldVar` and `OptimalInterpolation` agree there, the same
correctness gate as Decision D14 for the classical methods.

## 22.5 When to use it, and when strong 4D-Var

| | `SpaceTimeFieldVar` | `StrongFourDVar` (+ `NeuralFieldParameterization`) |
|---|---|---|
| Control | the whole space-time field $f_\theta(t, s)$ | the initial state $x_0 = \psi(\theta)$ |
| Dynamics | soft: $\lambda$-weighted residual at collocation points | hard: $x_t = \mathcal M^t(x_0)$ |
| Needs | a differentiable PDE residual | a `ForwardModel` and its adjoint |
| Cost per iteration | residual at $N_c$ points, parallel in time | a sequential rollout and its adjoint |
| Output | a continuous field; $u(t_0, \cdot)$ for forecasting | $x_0$ on the grid; the trajectory by rollout |
| Prior | spectral bias + weight decay, no $B$ | $B$, or the parameterization's penalty |

Use the neural field when observations are scattered in space and time,
when no tangent-linear/adjoint model is at hand but the equations are
known, or when the window is long enough that a sequential rollout
dominates the cost. Prefer strong 4D-Var when the model is trusted
(the residual weight then has to grow without bound to match it) and
when a forecast from the analysis must be consistent with the model
step by step. The two meet in the **strong-constraint variant**: fit
only $u(t_0, \cdot)$ as a field and roll it out,
`StrongFourDVar(..., parameterization=NeuralFieldParameterization(field, coords, alpha))`
([chapter 6](06_strong_4dvar.md#parameterized-initial-states)).

## References

- Oh, Y. (2025). On the effect of neural field reparameterization for
  4DVAR. arXiv:2509.21751.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
  Physics-informed neural networks. *J. Comput. Phys.* 378, 686–707.
- Rahaman, N., et al. (2019). On the spectral bias of neural networks.
  *ICML*.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and
  scalable predictive uncertainty estimation using deep ensembles.
  *NeurIPS*.

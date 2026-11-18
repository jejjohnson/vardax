---
status: draft
version: 0.1.0
numbering:
  enumerator: "18.%s"
---

# 18 · Latent Variational Data Assimilation

This chapter formalises variational data assimilation in a learned low-dimensional latent space. It complements chapters 5 to 8 (3DVar, strong and weak 4DVar, incremental 4DVar) and chapter 9 (4DVarNet) by specifying three latent-space peers: **latent 3DVar**, **latent strong-4DVar**, and **latent hybrid 4DVar**.

**Design anchor:** [design/latent_da.md](design/latent_da.md),
[Decision D17](design/decisions.md#d17-latent-da-as-a-first-class-peer-family).

---

## 18.1  Setting and notation

Augment the notation of [chapter 1](01_problem_setting.md) with an
autoencoder $(g_\text{enc}, g_\text{dec})$:

| Symbol | Space | Meaning |
|---|---|---|
| $\vect{x}$ | $\mathbb{R}^{D_x}$ | physical state |
| $\vect{z}$ | $\mathbb{R}^{D_z}$, $D_z \ll D_x$ | latent code |
| $\vect{y}$ | $\mathbb{R}^{D_y}$ | observations |
| $g_\text{enc}: \vect{x} \mapsto \vect{z}$ | encoder | satisfies `pipekit_cycle.Encoder` |
| $g_\text{dec}: \vect{z} \mapsto \hat{\vect{x}}$ | decoder | satisfies `pipekit_cycle.Decoder` |
| $\mat{J}_\text{dec}(\vect{z})$ | $\mathbb{R}^{D_x \times D_z}$ | Jacobian of $g_\text{dec}$ |
| $f_x: \vect{x} \mapsto \vect{x}$ | physical forward model, one step | as in chapter 3, with Jacobian $\mat{F}_x(\vect{x}) \in \mathbb{R}^{D_x \times D_x}$ |
| $f_z: \vect{z} \mapsto \vect{z}$ | latent forward model, one step | learned or $g_\text{enc} \circ f_x \circ g_\text{dec}$, with Jacobian $\mat{F}_z(\vect{z}) \in \mathbb{R}^{D_z \times D_z}$ |
| $h: \vect{x} \mapsto \vect{y}$ | observation operator | as in chapter 2, with Jacobian $\mat{H}(\vect{x}) \in \mathbb{R}^{D_y \times D_x}$ |
| $h_z := h \circ g_\text{dec}$ | **lifted** obs operator | $\vect{z} \mapsto \vect{y}$, with Jacobian $\mat{H}_z(\vect{z}) = \mat{H}(g_\text{dec}(\vect{z})) \mat{J}_\text{dec}(\vect{z}) \in \mathbb{R}^{D_y \times D_z}$ |
| $\gvect{\mu}_z$ | $\mathbb{R}^{D_z}$ | background in $\vect{z}$ |
| $\gmat{\Sigma}_z$ | $\mathbb{R}^{D_z \times D_z}$ | background-error covariance in $\vect{z}$ |
| $\gmat{\Sigma}_\varepsilon$ | $\mathbb{R}^{D_y \times D_y}$ | observation-error covariance |

The pair $(g_\text{enc}, g_\text{dec})$ is **consistent on the data manifold** if $g_\text{dec}(g_\text{enc}(\vect{x})) \approx \vect{x}$ for $\vect{x}$ drawn from the climatology. The formalism does not require exact identity; it requires only that $g_\text{dec}$ be differentiable, which is automatic for neural decoders.

---

## 18.2  Latent 3DVar

### 18.2.1  Cost

The latent counterpart of the 3DVar cost [](#eq-3dvar-cost) replaces the control vector $\vect{x}$ with $\vect{z}$, the observation operator $h$ with $h_z$ (so $\vect{x} = g_\text{dec}(\vect{z})$ in the misfit) and the background term with one in $\vect{z}$:

$$\mathcal{J}_\text{3D}(\vect{z}) = \underbrace{\frac{1}{2} \Vert \vect{z} - \gvect{\mu}_z \Vert^2_{\gmat{\Sigma}_z^{-1}}}_{\text{background, in } \vect{z}} + \underbrace{\frac{1}{2} \Vert \vect{y} - h(g_\text{dec}(\vect{z})) \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}}}_{\text{obs misfit, lifted}},$$ (eq-latent-3dvar-cost)

where $\gvect{\mu}_z$ is the background in $\vect{z}$.

### 18.2.2  Gradient and Hessian

Let $\mat{H}_z(\vect{z}) := \mat{H}(g_\text{dec}(\vect{z})) \mat{J}_\text{dec}(\vect{z})$ be the lifted tangent linear, the Jacobian of $h_z$ (chain rule). Then the 3DVar gradient [](#eq-3dvar-grad) becomes

$$\nabla_{\vect{z}} \mathcal{J}_\text{3D}(\vect{z}) = \gmat{\Sigma}_z^{-1}(\vect{z} - \gvect{\mu}_z) + \mat{H}_z(\vect{z})^\top \gmat{\Sigma}_\varepsilon^{-1}(h_z(\vect{z}) - \vect{y}),$$ (eq-latent-3dvar-grad)

and the Gauss-Newton Hessian [](#eq-3dvar-gn-hessian) becomes

$$\nabla^2_{\vect{z}} \mathcal{J}_\text{3D}(\vect{z}) \approx \gmat{\Sigma}_z^{-1} + \mat{H}_z(\vect{z})^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H}_z(\vect{z}).$$ (eq-latent-3dvar-hessian)

When $\mat{H}_z$ is treated as constant near the linearisation point (the **incremental** form, cf. [chapter 8](08_incremental_4dvar.md)), the minimisation reduces to a linear solve and the Sherman-Morrison-Woodbury identity gives the closed form

$$\vect{z}^\star = \gvect{\mu}_z + \gmat{\Sigma}_z \mat{H}_z^\top \left(\mat{H}_z \gmat{\Sigma}_z \mat{H}_z^\top + \gmat{\Sigma}_\varepsilon\right)^{-1}(\vect{y} - h_z(\gvect{\mu}_z)).$$ (eq-latent-3dvar-mean)

This is the BLUE mean [](#eq-blue-mean) with $(\gvect{\mu}_x, \gmat{\Sigma}_x, \mat{H})$ replaced by $(\gvect{\mu}_z, \gmat{\Sigma}_z, \mat{H}_z)$: the $D_y \times D_y$ inverse matches the OI form, and the difference is that the gain is now multiplied by $\gmat{\Sigma}_z$ (size $D_z \times D_z$) rather than $\gmat{\Sigma}_x$ (size $D_x \times D_x$).
This is the structural source of the order-of-magnitude wall-clock win
when $D_z \ll D_x$.

### 18.2.3  Analysis output

The variational solution is in $\vect{z}$; the analysis in physical space is $\vect{x}^\star = g_\text{dec}(\vect{z}^\star)$. The posterior covariance, when needed, comes from the Laplace approximation [](#eq-laplace-cov) at $\vect{z}^\star$, pushed forward through the decoder:

$$\gmat{\Sigma}_{z \mid y} = \left(\nabla^2_{\vect{z}} \mathcal{J}_\text{3D}(\vect{z}^\star)\right)^{-1} \text{ and } \gmat{\Sigma}_{x \mid y} = \mat{J}_\text{dec}(\vect{z}^\star) \gmat{\Sigma}_{z \mid y} \mat{J}_\text{dec}(\vect{z}^\star)^\top.$$ (eq-latent-laplace-cov)

The pushforward $\gmat{\Sigma}_{x \mid y}$ has rank at most $D_z$, so it is a low-rank object naturally represented by `lineax.LowRankUpdate`.

---

## 18.3  Latent strong-constraint 4DVar

### 18.3.1  Cost

The latent strong-constraint cost is [](#eq-strong4dvar-cost) with the control $\vect{z}_0$, the rollout $\vect{z}_{t+1} = f_z(\vect{z}_t)$ in the latent space, and the observation residuals through the lifted operator:

$$\mathcal{J}_\text{4D}(\vect{z}_0) = \frac{1}{2} \Vert \vect{z}_0 - \gvect{\mu}_z \Vert^2_{\gmat{\Sigma}_z^{-1}} + \frac{1}{2} \sum_{t=0}^{T} \Vert \vect{y}_t - h(g_\text{dec}(\vect{z}_t)) \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}}, \text{ with } \vect{z}_{t+1} = f_z(\vect{z}_t),$$ (eq-latent-strong4dvar-cost)

where $t = 0, \ldots, T$ indexes the observation times of the window, which has $T$ steps and $T + 1$ times, and $\vect{y}_t \in \mathbb{R}^{D_y}$.

### 18.3.2  Gradient

Backpropagation through the latent rollout gives the analogue of the strong 4DVar gradient [](#eq-strong4dvar-grad),

$$\nabla_{\vect{z}_0} \mathcal{J}_\text{4D} = \gmat{\Sigma}_z^{-1}(\vect{z}_0 - \gvect{\mu}_z) + \sum_{t=0}^{T} \left(\prod_{j=0}^{t-1} \mat{F}_z(\vect{z}_j)\right)^\top \mat{H}_z(\vect{z}_t)^\top \gmat{\Sigma}_\varepsilon^{-1} (h_z(\vect{z}_t) - \vect{y}_t).$$ (eq-latent-strong4dvar-grad)

For learned $f_z$ implemented as an `eqx.Module`, the tangent-linear
operators come from JAX autodiff; no hand-coded adjoint is required.

### 18.3.3  Memory bound

The full unrolled adjoint stores $O(T D_z)$ values rather than $O(T D_x)$, another order-of-magnitude saving for large $D_x$.
With $T = 20$, $D_z = 8$ versus $D_x = 10^4$, the backward tape shrinks
from megabytes to kilobytes.

---

## 18.4  Latent hybrid 4DVar

### 18.4.1  Motivation

The strong-latent form (§18.3) commits the *forecast* to the latent manifold, which can accumulate reconstruction error over long windows. The hybrid form keeps the forecast in physical space, where the trusted physics model lives, and uses $\vect{z}$ only as the control variable and as the natural background space:

* Control: $\vect{z}_0 \in \mathbb{R}^{D_z}$.
* Initial physical state: $\vect{x}_0 = g_\text{dec}(\vect{z}_0)$.
* Rollout: $\vect{x}_{t+1} = f_x(\vect{x}_t)$ in $\mathbb{R}^{D_x}$ (physics, not AE).
* Observation residual: $\vect{y}_t - h(\vect{x}_t)$, no $g_\text{dec}$ inside the sum.

### 18.4.2  Cost

The hybrid cost is the strong 4DVar cost [](#eq-strong4dvar-cost) with $\vect{x}_0 = g_\text{dec}(\vect{z}_0)$ and the background term moved to $\vect{z}$:

$$\mathcal{J}_\text{hyb}(\vect{z}_0) = \frac{1}{2} \Vert \vect{z}_0 - \gvect{\mu}_z \Vert^2_{\gmat{\Sigma}_z^{-1}} + \frac{1}{2} \sum_{t=0}^{T} \Vert \vect{y}_t - h(\vect{x}_t) \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}}, \text{ with } \vect{x}_0 = g_\text{dec}(\vect{z}_0), \vect{x}_{t+1} = f_x(\vect{x}_t).$$ (eq-latent-hybrid-cost)

### 18.4.3  Gradient

$$\nabla_{\vect{z}_0} \mathcal{J}_\text{hyb} = \gmat{\Sigma}_z^{-1}(\vect{z}_0 - \gvect{\mu}_z) + \mat{J}_\text{dec}(\vect{z}_0)^\top \sum_{t=0}^{T} \left(\prod_{j=0}^{t-1} \mat{F}_x(\vect{x}_j)\right)^\top \mat{H}(\vect{x}_t)^\top \gmat{\Sigma}_\varepsilon^{-1}(h(\vect{x}_t) - \vect{y}_t).$$ (eq-latent-hybrid-grad)

The decoder Jacobian $\mat{J}_\text{dec}(\vect{z}_0)$ appears only **once**, at the start of the window. The expensive piece is the physics adjoint $\prod \mat{F}_x$, identical to the one in the strong 4DVar gradient [](#eq-strong4dvar-grad).

### 18.4.4  When to prefer hybrid over strong-latent

| Scenario | Recommended |
|---|---|
| AE reconstructs the manifold well over the assimilation window | strong-latent (§18.3), the cheapest |
| Physics model is trusted; AE only approximate | hybrid (§18.4) |
| No learned $f_z$ available, and `EncodedForwardModel` is too lossy | hybrid |
| $D_x$ huge ($10^6$+) and physics adjoint is the bottleneck | strong-latent (forces $f_z$ training) |

---

## 18.5  Latent incremental 4DVar (sketch)

The incremental algorithm of [chapter 8](08_incremental_4dvar.md) extends mutatis mutandis: the outer Gauss-Newton loop produces a sequence of inner problems, each the incremental cost [](#eq-incremental-cost) in $\vect{z}$,

$$\min_{\delta\vect{z}} \frac{1}{2} \Vert \vect{z}_0^{(i)} + \delta\vect{z} - \gvect{\mu}_z \Vert^2_{\gmat{\Sigma}_z^{-1}} + \frac{1}{2} \sum_{t=0}^{T} \Vert \vect{r}_t^{(i)} - \mat{H}_{z,t} \mat{F}_{z,t \leftarrow 0} \delta\vect{z} \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}},$$ (eq-latent-incremental-cost)

with $\vect{z}_0^{(i)}$ the outer guess at outer iteration $i$, $\vect{r}_t^{(i)} = \vect{y}_t - h_z(\vect{z}_t^{(i)})$ the residual (innovation), $\mat{F}_{z,t \leftarrow 0} \in \mathbb{R}^{D_z \times D_z}$ the tangent-linear model of the latent rollout to time $t$ and $\mat{H}_{z,t}$ the lifted Jacobian at $\vect{z}_t^{(i)}$, both evaluated at the current outer guess. The inner problem is linear-quadratic in $\delta\vect{z}$ and CG-able in $\mathbb{R}^{D_z}$, typically with two orders of magnitude fewer CG iterations than in $\mathbb{R}^{D_x}$ because the conditioning improves with dimensionality reduction.

A control-variable transform (CVT) in $\vect{z}$ is rarely needed: if the AE
was trained with a unit-Gaussian prior on $\vect{z}$, then $\gmat{\Sigma}_z = \mat{I}$
to leading order and the cost is already well-conditioned.

---

## 18.6  Identity-AE consistency

If $(g_\text{enc}, g_\text{dec}) = (\mathrm{id}, \mathrm{id})$ then $\vect{z} = \vect{x}$,
$h_z = h$, $f_z = f_x$, $\gmat{\Sigma}_z = \gmat{\Sigma}_x$, and the
three latent costs reduce exactly to the corresponding $\vect{x}$-space costs.
This is the canonical regression test for any implementation
(`vardax.identity_latent_map(N)`).

---

## 18.7  Connection to chapter 9 (4DVarNet)

`FourDVarNet` minimises the learned-prior cost [](#eq-4dvarnet-cost) with the prior $g_\text{ae} = g_\text{dec} \circ g_\text{enc}$, so its prior term is $\alpha_p \Vert \vect{x} - g_\text{dec}(g_\text{enc}(\vect{x})) \Vert_2^2$.

This is **prior-only latent**: the AE appears as a reconstruction penalty but the control remains $\vect{x}$. The latent methods of this chapter promote $\vect{z}$ itself to the control vector. The two are not incompatible, and `FourDVarNet` retains its place when the autoencoder is a *regulariser* rather than a *parameterisation* of the state space.

---

## 18.8  Worked example: Lorenz-96, $D_x = 40$, $D_z = 8$

The notebook `docs/notebooks/18_latent_lorenz.ipynb` (to be added with
the implementation) trains a bilinear AE on L96 trajectories and runs
all three latent methods against the $\vect{x}$-space baselines on the same
fixture. Expected pattern:

* `LatentThreeDVar` matches `ThreeDVar` analysis RMSE to within
  $0.05$ standard deviations; cost-per-cycle drops by $\sim 25\times$.
* `LatentStrongFourDVar` with `EncodedForwardModel` (no learned $f_z$)
  matches `StrongFourDVar`; cost-per-cycle drops by $\sim 10\times$
  (the AE round-trip eats some of the win).
* `LatentStrongFourDVar` with a trained residual-MLP $f_z$ wins
  $\sim 50\times$ but introduces a small RMSE bias controlled by AE
  reconstruction quality.
* `LatentHybridFourDVar` matches `StrongFourDVar` RMSE almost exactly;
  cost-per-cycle drops by $\sim 4\times$ (dominated by physics rollout
  which is unchanged).

---

## 18.9  References

1. Peyron, M., Fillion, A., Gürol, S., Marchais, V., Gratton, S.,
   Boudier, P. & Goret, G. (2021). *Latent space data assimilation by
   using deep learning.* QJRMS, 147(740), 3759-3777.
2. Cheng, S., Quilodrán-Casas, C., Ouala, S. et al. (2023).
   *Generalised latent assimilation in heterogeneous reduced spaces
   with machine learning surrogates.* J. Sci. Comput., 94, 11.
3. Fablet, R., Chapron, B., Drumetz, L., Mémin, E., Pannekoucke, O. &
   Rousseau, F. (2021). *Learning variational data assimilation
   models and solvers.* JAMES, 13(10).
4. Brajard, J., Carrassi, A., Bocquet, M. & Bertino, L. (2021).
   *Combining data assimilation and machine learning to infer
   unresolved scale parametrization.* Phil. Trans. R. Soc. A,
   379(2194).
5. Lguensat, R., Tandeo, P., Ailliot, P., Pulido, M. & Fablet, R.
   (2017). *The analog data assimilation.* MWR, 145(10).

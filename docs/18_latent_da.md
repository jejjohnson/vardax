---
status: draft
version: 0.1.0
---

# 18 · Latent Variational Data Assimilation

This chapter formalises variational data assimilation in a learned
low-dimensional latent space. It complements chapters 5–8 (3DVar,
strong-/weak-4DVar, incremental 4DVar) and chapter 9 (4DVarNet) by
specifying three latent-space peers: **latent 3DVar**, **latent
strong-4DVar**, and **latent hybrid 4DVar**.

**Design anchor:** [design/latent_da.md](design/latent_da.md),
[Decision D17](design/decisions.md#d17-latent-da-as-a-first-class-peer-family).

---

## 18.1  Setting and notation

Augment the notation of [chapter 1](01_problem_setting.md) with an
autoencoder $(\varphi, \psi)$:

| Symbol | Space | Meaning |
|---|---|---|
| $x$ | $\mathbb{R}^{N_x}$ | physical state |
| $z$ | $\mathbb{R}^{N_z}$, $N_z \ll N_x$ | latent code |
| $\varphi: x \mapsto z$ | encoder | satisfies `pipekit_cycle.Encoder` |
| $\psi: z \mapsto \hat{x}$ | decoder | satisfies `pipekit_cycle.Decoder` |
| $M_x: x \mapsto x$ | physical forward model | as in chapter 3 |
| $M_z: z \mapsto z$ | latent forward model | learned or $\varphi \circ M_x \circ \psi$ |
| $H: x \mapsto y$ | observation operator | as in chapter 2 |
| $\tilde{H} := H \circ \psi$ | **lifted** obs operator | $z \mapsto y$ |
| $\mathbf{B}_z$ | $\mathbb{R}^{N_z \times N_z}$ | background-error covariance in $z$ |
| $\mathbf{R}$ | $\mathbb{R}^{N_y \times N_y}$ | observation-error covariance |

The pair $(\varphi, \psi)$ is **consistent on the data manifold** if
$\psi(\varphi(x)) \approx x$ for $x$ drawn from the climatology. The
formalism does not require exact identity — it requires only that
$\psi$ be differentiable, which is automatic for neural decoders.

---

## 18.2  Latent 3DVar

### 18.2.1  Cost

The latent counterpart of [chapter 5's 3DVar](05_threedvar.md) replaces
the control vector $x$ with $z$ and the observation operator $H$ with
$\tilde{H}$:

$$
\boxed{\;
J_{3D}(z) \;=\;
\underbrace{\tfrac{1}{2}\|z - z_b\|^2_{\mathbf{B}_z^{-1}}}_{\text{background, in } z}
\;+\;
\underbrace{\tfrac{1}{2}\|y - H(\psi(z))\|^2_{\mathbf{R}^{-1}}}_{\text{obs misfit, lifted}}.
\;}
$$

### 18.2.2  Gradient and Hessian

Let $\tilde{H}'(z) := H'(\psi(z))\,\psi'(z)$ be the lifted tangent
linear (chain rule). Then

$$
\nabla_z J_{3D}(z) = \mathbf{B}_z^{-1}(z - z_b)
                   - \tilde{H}'(z)^\top \mathbf{R}^{-1}(y - \tilde{H}(z)),
$$

and the Gauss–Newton Hessian is

$$
\mathbf{H}_z(z) \;\approx\;
\mathbf{B}_z^{-1} + \tilde{H}'(z)^\top \mathbf{R}^{-1} \tilde{H}'(z).
$$

When $\tilde{H}'$ is treated as constant near the linearisation point
(the **incremental** form, cf. [chapter 8](08_incremental_4dvar.md)),
the minimisation reduces to a linear solve and the Sherman–Morrison
identity gives the closed form

$$
z^* = z_b + \mathbf{B}_z \tilde{H}'^\top \bigl(\tilde{H}' \mathbf{B}_z \tilde{H}'^\top + \mathbf{R}\bigr)^{-1}(y - \tilde{H}(z_b)).
$$

The $(N_y \times N_y)$ inverse matches the OI form (chapter 4); the
difference is that the gain is now multiplied by $\mathbf{B}_z$
(size $N_z \times N_z$) rather than $\mathbf{B}$ (size $N_x \times N_x$).
This is the structural source of the order-of-magnitude wall-clock win
when $N_z \ll N_x$.

### 18.2.3  Analysis output

The variational solution is in $z$; the analysis in physical space is
$x^* = \psi(z^*)$. The posterior covariance, when needed, comes from
the Laplace approximation at $z^*$:

$$
\mathrm{cov}_z = \mathbf{H}_z(z^*)^{-1},
\qquad
\mathrm{cov}_x = \psi'(z^*) \,\mathrm{cov}_z\, \psi'(z^*)^\top.
$$

The pushforward $\mathrm{cov}_x$ has rank at most $N_z$ — a low-rank
object naturally represented by `lineax.LowRankUpdate`.

---

## 18.3  Latent strong-constraint 4DVar

### 18.3.1  Cost

Control vector $z_0$; rollout $z_{k+1} = M_z(z_k)$ in the latent space;
observation residuals via the lifted operator:

$$
\boxed{\;
J_{4D}(z_0) \;=\;
\tfrac{1}{2}\|z_0 - z_b\|^2_{\mathbf{B}_z^{-1}}
\;+\;
\tfrac{1}{2}\sum_{k=0}^{K}\|y_k - H(\psi(z_k))\|^2_{\mathbf{R}^{-1}},
\quad z_{k+1} = M_z(z_k).
\;}
$$

### 18.3.2  Gradient

Backpropagation through the latent rollout yields, for each observation
window $k$,

$$
\frac{\partial J_{4D}}{\partial z_0}
=
\mathbf{B}_z^{-1}(z_0 - z_b)
- \sum_{k=0}^{K}
\Bigl(\prod_{j=0}^{k-1} M_z'(z_j)\Bigr)^{\!\!\top}
\tilde{H}'(z_k)^\top \mathbf{R}^{-1} (y_k - \tilde{H}(z_k)).
$$

For learned $M_z$ implemented as an `eqx.Module`, the tangent-linear
operators come from JAX autodiff; no hand-coded adjoint is required.

### 18.3.3  Memory bound

The full unrolled adjoint stores $O(K \cdot N_z)$ values rather than
$O(K \cdot N_x)$ — another order-of-magnitude saving for large $N_x$.
With $K = 20$, $N_z = 8$ versus $N_x = 10^4$, the backward tape shrinks
from megabytes to kilobytes.

---

## 18.4  Latent hybrid 4DVar

### 18.4.1  Motivation

The strong-latent form (§18.3) commits the *forecast* to the latent
manifold, which can accumulate reconstruction error over long windows.
The hybrid form keeps the forecast in physical space — where the
trusted physics model lives — and uses $z$ only as the control variable
and as the natural background space:

* Control: $z_0 \in \mathcal{Z}$.
* Initial physical state: $x_0 = \psi(z_0)$.
* Rollout: $x_{k+1} = M_x(x_k)$ in $\mathcal{X}$ (physics, not AE).
* Observation residual: $y_k - H(x_k)$, no $\psi$ inside the sum.

### 18.4.2  Cost

$$
\boxed{\;
J_{H}(z_0) \;=\;
\tfrac{1}{2}\|z_0 - z_b\|^2_{\mathbf{B}_z^{-1}}
+ \tfrac{1}{2}\sum_{k=0}^{K}\|y_k - H(x_k)\|^2_{\mathbf{R}^{-1}},
\quad
x_0 = \psi(z_0),\;\; x_{k+1} = M_x(x_k).
\;}
$$

### 18.4.3  Gradient

$$
\frac{\partial J_H}{\partial z_0}
=
\mathbf{B}_z^{-1}(z_0 - z_b)
- \psi'(z_0)^\top
\sum_{k=0}^{K}
\Bigl(\prod_{j=0}^{k-1} M_x'(x_j)\Bigr)^{\!\!\top}
H'(x_k)^\top \mathbf{R}^{-1}(y_k - H(x_k)).
$$

The decoder Jacobian $\psi'(z_0)$ appears only **once**, at the start
of the window. The expensive piece is the physics adjoint
$\prod M_x'$, identical to chapter 6.

### 18.4.4  When to prefer hybrid over strong-latent

| Scenario | Recommended |
|---|---|
| AE reconstructs the manifold well over the assimilation window | strong-latent (§18.3) — cheapest |
| Physics model is trusted; AE only approximate | hybrid (§18.4) |
| No learned $M_z$ available, and `EncodedForwardModel` is too lossy | hybrid |
| $N_x$ huge ($10^6$+) and physics adjoint is the bottleneck | strong-latent (forces $M_z$ training) |

---

## 18.5  Latent incremental 4DVar (sketch)

The incremental algorithm of [chapter 8](08_incremental_4dvar.md)
extends mutatis mutandis: outer Gauss–Newton loop produces a sequence
of inner problems

$$
\min_{\delta z}\; \tfrac{1}{2}\|\delta z\|^2_{\mathbf{B}_z^{-1}}
+ \tfrac{1}{2}\sum_k \|d_k - \tilde{H}_k' M_{z,k}' \delta z\|^2_{\mathbf{R}^{-1}},
$$

with $d_k = y_k - \tilde{H}(z_k^{(n)})$ the innovation at outer
iteration $n$ and primed quantities evaluated at the current outer
guess. The inner problem is linear-quadratic in $\delta z$ and CG-able
in $\mathcal{Z}$ — typically two orders of magnitude fewer CG
iterations than in $\mathcal{X}$ because the conditioning improves with
dimensionality reduction.

A control-variable transform (CVT) in $z$ is rarely needed: if the AE
was trained with a unit-Gaussian prior on $z$, then $\mathbf{B}_z = I$
to leading order and the cost is already well-conditioned.

---

## 18.6  Identity-AE consistency

If $(\varphi, \psi) = (\mathrm{id}, \mathrm{id})$ then $z = x$,
$\tilde{H} = H$, $M_z = M_x$, $\mathbf{B}_z = \mathbf{B}$, and the
three latent costs reduce exactly to the corresponding x-space costs.
This is the canonical regression test for any implementation
(`vardax.identity_latent_map(N)`).

---

## 18.7  Connection to chapter 9 (4DVarNet)

`FourDVarNet` minimises

$$
J_{NN}(x) = \alpha_{\text{obs}}\|y - H(x)\|^2_{\mathbf{R}^{-1}}
          + \alpha_{\text{prior}}\|x - \psi(\varphi(x))\|^2.
$$

This is **prior-only latent**: the AE appears as a reconstruction
penalty but the control remains $x$. The latent methods of this
chapter promote $z$ itself to the control vector. The two are not
incompatible — `FourDVarNet` retains its place when the autoencoder is
a *regulariser* rather than a *parameterisation* of the state space.

---

## 18.8  Worked example — Lorenz-96, $N_x=40$, $N_z=8$

The notebook `docs/notebooks/18_latent_lorenz.ipynb` (to be added with
the implementation) trains a bilinear AE on L96 trajectories and runs
all three latent methods against the x-space baselines on the same
fixture. Expected pattern:

* `LatentThreeDVar` matches `ThreeDVar` analysis RMSE to within
  $0.05$ standard deviations; cost-per-cycle drops by $\sim 25\times$.
* `LatentStrongFourDVar` with `EncodedForwardModel` (no learned $M_z$)
  matches `StrongFourDVar`; cost-per-cycle drops by $\sim 10\times$
  (the AE round-trip eats some of the win).
* `LatentStrongFourDVar` with a trained residual-MLP $M_z$ wins
  $\sim 50\times$ but introduces a small RMSE bias controlled by AE
  reconstruction quality.
* `LatentHybridFourDVar` matches `StrongFourDVar` RMSE almost exactly;
  cost-per-cycle drops by $\sim 4\times$ (dominated by physics rollout
  which is unchanged).

---

## 18.9  References

1. Peyron, M., Fillion, A., Gürol, S., Marchais, V., Gratton, S.,
   Boudier, P. & Goret, G. (2021). *Latent space data assimilation by
   using deep learning.* QJRMS, 147(740), 3759–3777.
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

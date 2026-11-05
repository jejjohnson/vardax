# Notation

Every page and docstring uses the symbols below: the letter says what kind of object a symbol is, the subscript says which variable it belongs to, and a bold symbol is a vector or a matrix.
Equations use $\vect{x}$ in the documentation and Unicode, such as Σ_x|y, in docstrings and code comments.

## Sizes and indices

| Symbol | Meaning | In code |
| --- | --- | --- |
| $D_x$, $D_y$, $D_z$ | length of the state, the observation vector and the coefficient vector | `N`, `M`, `Z` |
| $T$ | number of steps in a window, which holds $T + 1$ times | `T` |
| $t$, $k$ | a regular time step; a real time stamp $t_k$ when steps are uneven | |
| $N_\text{batch}$ | number of samples in a batch | `B` |
| $N_\text{iter}$, $i$ | number of optimiser iterations; one iteration, written as a superscript $\vect{x}^{(i)}$ | `n_steps` |
| $N_\mathcal{E}$, $e$ | number of ensemble members or Monte Carlo draws; one member $\vect{x}^{(e)}$ | `n_samples` |
| $\vect{s}$, $\Omega$ | a spatial coordinate and the domain it lives in | `coords` |

## State and observations

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $\vect{x}_t$ | $\mathbb{R}^{D_x}$ | the state at time $t$ |
| $\vect{y}_t$ | $\mathbb{R}^{D_y}$ | the observations at time $t$ |
| $\vect{m}_t$, $\mat{M}_t = \op{diag}(\vect{m}_t)$ | $\{0, 1\}^{D_y}$, $D_y \times D_y$ | the observation mask, with 1 where observed |
| $\vect{z}$ | $\mathbb{R}^{D_z}$ | the coefficients of a reduced basis or a whitened control |
| $\gvect{\theta}$ | $\mathbb{R}^{D_\theta}$ | parameters: a control, or a network's weights |
| $\gvect{\nu}$ | $\mathbb{R}^{D_\nu}$ | the encoder and head parameters of an amortized posterior |
| $\gvect{\varepsilon}_t$ | $\mathbb{R}^{D_y}$ | observation noise, $\gvect{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \gmat{\Sigma}_\varepsilon)$ |
| $\gvect{\eta}_t$ | $\mathbb{R}^{D_x}$ | model noise, $\gvect{\eta}_t \sim \mathcal{N}(\mathbf{0}, \gmat{\Sigma}_\eta)$ |
| $\vect{r}_t$ | $\mathbb{R}^{D_y}$ | the residual, also called the innovation |
| $\vect{x}^\star$ | $\mathbb{R}^{D_x}$ | an optimum, such as the analysis $\vect{x}_0^\star$ |
| $\hat{\vect{x}}$ | $\mathbb{R}^{D_x}$ | an estimate, such as a learned prior's reconstruction |

## Means, covariances and gains

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $\gvect{\mu}_x$ | $\mathbb{R}^{D_x}$ | prior mean of the state |
| $\gmat{\Sigma}_x$, $\gmat{\Lambda}_x = \gmat{\Sigma}_x^{-1}$ | $D_x \times D_x$ | prior covariance and precision |
| $\gmat{\Sigma}_\varepsilon$ | $D_y \times D_y$ | observation-error covariance |
| $\tilde{\gmat{\Sigma}}_{\varepsilon,t} = \mat{M}_t \gmat{\Sigma}_\varepsilon \mat{M}_t + (\mat{I} - \mat{M}_t)$ | $D_y \times D_y$ | the same covariance with masked entries dropped out |
| $\gmat{\Sigma}_\eta$ | $D_x \times D_x$ | model-error covariance |
| $\gmat{\Sigma}_y = \mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon$ | $D_y \times D_y$ | covariance of the observations |
| $\mat{G} = \gmat{\Sigma}_x \mat{H}^\top \gmat{\Sigma}_y^{-1}$ | $D_x \times D_y$ | the gain |
| $\gmat{\Sigma}_{x \mid y}$, $\gmat{\Lambda}_{x \mid y}$ | $D_x \times D_x$ | posterior covariance and precision |
| $\gmat{\Phi}$, $\gmat{\Sigma}_z$ | $D_x \times D_z$, $D_z \times D_z$ | a reduced basis, and the prior covariance of its coefficients |

## Functions and costs

| Symbol | Maps | Meaning |
| --- | --- | --- |
| $f$, $\mat{F}$ | $\mathbb{R}^{D_x} \to \mathbb{R}^{D_x}$ | the forward model, and its linear form or Jacobian |
| $f_{1:T}$ | $\mathbb{R}^{D_x} \to \mathbb{R}^{T \times D_x}$ | a rollout from the initial state |
| $h$, $\mat{H}$ | $\mathbb{R}^{D_x} \to \mathbb{R}^{D_y}$ | the observation operator, and its linear form or Jacobian |
| $\Psi_{\Delta t}$ | $\mathbb{R}^{D_x} \to \mathbb{R}^{D_x}$ | the flow over $\Delta t$ of an ODE $\dot{\vect{x}} = f(t, \vect{x}; \gvect{\theta})$ |
| $g_\text{dec}$, $g_\text{enc}$ | $\mathbb{R}^{D_z} \to \mathbb{R}^{D_x}$, its inverse direction | a learned decoder and encoder |
| $g_\text{ae}(\vect{x}; \gvect{\theta})$ | $\mathbb{R}^{D_x} \to \mathbb{R}^{D_x}$ | a learned autoencoder prior, with reconstruction $\hat{\vect{x}}$ |
| $g_\text{mod}$ | $(\nabla_{\vect{x}} \mathcal{J}, \vect{x}, \vect{c}) \mapsto (\Delta \vect{x}, \vect{c}')$ | a learned gradient modulator with carry $\vect{c}$ |
| $\mathcal{J}$, $\mathcal{J}_b$, $\mathcal{J}_o$, $\mathcal{J}_\eta$, $\mathcal{J}_p$, $\mathcal{J}_\text{res}$ | scalar | a cost over the state, and its background, observation, model-error, prior and PDE-residual terms |
| $\mathcal{L}$, $\mathcal{L}_\text{MSE}$, $\mathcal{L}_\text{NLL}$ | scalar | a loss over the parameters, always the quantity minimised |
| $\mathcal{R}(\gvect{\theta})$ | scalar | the penalty of a state parameterization |
| $q(\vect{x} \mid \vect{y}; \gvect{\nu})$ | density | an amortized posterior |

Scalar weights are $\alpha$ for a state step size or a term in a sum of costs, $\gamma$ for a learning rate, $\beta$ for a regularisation weight, $\beta_p$ for the prior term of 4DVarNet and $\beta_\text{res}$ for a PDE residual.

## Classic letters

Papers on data assimilation use the letters on the left, and this documentation quotes them only when citing a source.

| Classic | Here | Meaning |
| --- | --- | --- |
| $\vect{x}_b$ | $\gvect{\mu}_x$ | background, or prior mean |
| $\mat{B}$, $\mat{P}^f$ | $\gmat{\Sigma}_x$ | background, or forecast, covariance |
| $\mat{R}$ | $\gmat{\Sigma}_\varepsilon$ | observation-error covariance |
| $\mat{Q}$ | $\gmat{\Sigma}_\eta$ | model-error covariance |
| $\mat{S}$ | $\gmat{\Sigma}_y$ | innovation covariance |
| $\mat{K}$ | $\mat{G}$ | Kalman gain; $\mat{K}$ is kept for kernel matrices |
| $\vect{d}$ | $\vect{r}$ | innovation |
| $\mat{A}$ | $\gmat{\Sigma}_{x \mid y}$ | analysis covariance |
| $\vect{x}^f$, $\vect{x}^a$ | $\vect{x}_{t \mid t-1}$, $\vect{x}_{t \mid t}$ | forecast and analysis |
| $M$, $\mathcal{M}$ | $f$, $\mat{F}$ | the dynamical model |
| $\mathcal{H}$ | $h$, $\mat{H}$ | the observation operator |
| $J$ | $\mathcal{J}$ | the variational cost |
| $\varphi_\theta$ | $g_\text{ae}$ | the learned prior of 4DVarNet |
| $\Psi_\phi$ | $g_\text{mod}$ | the learned gradient modulator of 4DVarNet |
| $\lambda$ | $\beta_p$ | the prior weight of 4DVarNet |
| $K$ | $N_\text{iter}$ | the number of solver steps of 4DVarNet |

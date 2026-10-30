# Notation

| Symbol | Description |
|---|---|
| $\mathbf{x}$ | State vector / field |
| $\mathbf{y}$ | Observations |
| $\mathbf{m}$ | Binary observation mask |
| $\mathcal{H}$ | Observation operator |
| $\varphi_\theta$ | Learned prior (autoencoder) |
| $\Psi_\phi$ | Gradient modulator (ConvLSTM) |
| $J$ | Variational (energy) cost |
| $J_{obs}$ | Observation cost |
| $J_{prior}$ | Prior cost |
| $\lambda$ | Prior weight |
| $\alpha$ | Gradient step-size |
| $K$ | Number of solver steps |
| $B$ | Batch size |
| $T$ | Number of time steps |
| $N$ | Spatial size (1-D) |
| $H, W$ | Spatial height / width (2-D) |
| $C$ | Number of channels (multivariate) |
| $\odot$ | Element-wise product |
| $*$ | Convolution |
| $\sigma(\cdot)$ | Sigmoid activation |

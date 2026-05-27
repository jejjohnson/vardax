# vardax — Mathematical Reference

This directory contains the mathematical reference for the vardax (formerly
`fourdvarjax`) library.

## Contents

### Core 4DVarNet (current v0.1.x implementation)

1. [Problem Setting](01_problem_setting.md)
2. [Variational Cost](02_variational_cost.md)
3. [Autoencoder Architecture](03_autoencoder_architecture.md)
4. [Learned Gradient Solver](04_learned_gradient_solver.md)
5. [Gradient Modulator](05_gradient_modulator.md)
6. [Implicit Differentiation](06_implicit_differentiation.md)
7. [Training Objective](07_training_objective.md)
8. [Algorithm Pseudocode](08_algorithm_pseudocode.md)
9. [1-D Lorenz 63](09_1d_lorenz63.md)
10. [Multivariate 2-D](10_multivariate_2d.md)
11. [Model vs Learned Prior](11_model_vs_learned_prior.md)

### v0.3.0 additions

12. [Observation Operators](12_observation_operators.md) — masked identity,
    averaging kernel, multi-instrument fusion
13. [Incremental 4DVar](13_incremental_4dvar.md) — Gauss-Newton outer / CG
    inner / control-variable transform
14. [Posterior Covariance](14_posterior_covariance.md) — Laplace / GN-Hessian
    / ensemble adapters
15. [Amortized Inference](15_amortized_inference.md) — conditional flow,
    score-based, regression heads
16. [Six-Step Inference Cycle](16_six_step_cycle.md) — methodology for the
    research-to-operations arc

### Reference

- [Notation](notation.md)
- [References](references.md)

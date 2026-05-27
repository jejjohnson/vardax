# References

## 4DVarNet (chapters 1–11)

1. Fablet, R., Amar, M. M., Febvre, Q., Beauchamp, M., & Chapron, B. (2021).
   *End-to-end physics-informed representation learning for satellite ocean
   remote sensing data: Applications to satellite altimetry and ocean color.*
   ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information
   Sciences, V-3-2021, 295–302.

2. Fablet, R., Chapron, B., Drumetz, L., Mémin, E., Pannekoucke, O., &
   Rousseau, F. (2021). *Learning variational data assimilation models and
   solvers.* Journal of Advances in Modeling Earth Systems, 13(10).

3. Fablet, R., Febvre, Q., & Chapron, B. (2023). *Multimodal 4DVarNets for
   the reconstruction of sea surface dynamics from NADIR and wide-swath
   altimetry.* IEEE Transactions on Geoscience and Remote Sensing, 61, 1–14.

4. LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006).
   *A tutorial on energy-based learning.* Predicting structured data, 1(0).

## Differentiation strategies (chapter 6)

5. Bolte, J., Pauwels, E., & Vaiter, S. (2023). *One-step differentiation of
   iterative algorithms.* NeurIPS 36. arXiv:2305.13768.

## Incremental 4DVar (chapter 13)

6. Courtier, P., Thépaut, J.-N., & Hollingsworth, A. (1994). *A strategy
   for operational implementation of 4D-Var, using an incremental
   approach.* Quarterly Journal of the Royal Meteorological Society,
   120(519), 1367–1387.

7. Carrassi, A., Bocquet, M., Bertino, L., & Evensen, G. (2018). *Data
   assimilation in the geosciences: An overview of methods, issues, and
   perspectives.* WIREs Climate Change, 9(5), e535.

8. Bannister, R. N. (2017). *A review of operational methods of variational
   and ensemble-variational data assimilation.* Quarterly Journal of the
   Royal Meteorological Society, 143(703), 607–633.

## Posterior covariance (chapter 14)

9. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for
   Machine Learning.* MIT Press.

10. Cressie, N., & Wikle, C. K. (2011). *Statistics for Spatio-Temporal
    Data.* Wiley.

11. Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
    *Validating Bayesian inference algorithms with simulation-based
    calibration.* arXiv:1804.06788.

## Amortized inference (chapter 15)

12. Cranmer, K., Brehmer, J., & Louppe, G. (2020). *The frontier of
    simulation-based inference.* Proceedings of the National Academy of
    Sciences, 117(48), 30055–30062.

13. Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., &
    Lakshminarayanan, B. (2021). *Normalizing flows for probabilistic
    modeling and inference.* JMLR, 22(57), 1–64.

14. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., &
    Poole, B. (2021). *Score-based generative modeling through stochastic
    differential equations.* ICLR.

15. Cohen, S., Amos, B., Lipman, Y. (2023). *Score-based diffusion meets
    annealed importance sampling.* NeurIPS.

## Ecosystem

16. Predecessor: [mvardax](https://github.com/jejjohnson/mvardax) (deprecated).

17. Reference architecture: [CIA-Oceanix/4dvarnet-starter](https://github.com/CIA-Oceanix/4dvarnet-starter).

18. Forward-model libraries: [`somax`](https://github.com/jejjohnson/somax)
    (geophysics), `plumax` (methane / atmospheric transport).

19. Linear algebra: [`gaussx`](https://github.com/jejjohnson/gaussx)
    (structured operators, Matérn factorisation),
    [`lineax`](https://github.com/patrick-kidger/lineax) (linear solvers).

20. Orchestration: [`pipekit`](https://github.com/jejjohnson/pipekit) +
    `pipekit-cycle` (`ForwardModel`, `ObservationOperator`, `AnalysisStep`
    protocols and `DACycle` orchestrator).

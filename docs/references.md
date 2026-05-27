# References

## Foundational DA texts

1. Daley, R. (1991). *Atmospheric Data Analysis.* Cambridge
   University Press.
2. Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and
   Predictability.* Cambridge University Press.
3. Asch, M., Bocquet, M., & Nodet, M. (2016). *Data Assimilation:
   Methods, Algorithms, and Applications.* SIAM.
4. Carrassi, A., Bocquet, M., Bertino, L., & Evensen, G. (2018).
   *Data assimilation in the geosciences: An overview of methods,
   issues, and perspectives.* WIREs Climate Change 9(5), e535.

## Optimal Interpolation / BLUE (chapter 4)

5. Lorenc, A. (1981). *A global three-dimensional multivariate
   statistical interpolation scheme.* MWR 109(4).
6. Eliassen, A. (1954). *Provisional report on calculation of spatial
   covariance and autocorrelation of the pressure field.* Inst.
   Weather and Climate Res., Acad. Norway.
7. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes
   for Machine Learning.* MIT Press. (Ch. 2.7 for the GP-as-OI view.)

## 3DVar and 4DVar — strong constraint (chapters 5, 6)

8. Lorenc, A. C. (1986). *Analysis methods for numerical weather
   prediction.* QJRMS 112(474).
9. Talagrand, O., & Courtier, P. (1987). *Variational assimilation of
   meteorological observations with the adjoint vorticity equation.*
   QJRMS 113(478).
10. Le Dimet, F.-X., & Talagrand, O. (1986). *Variational algorithms
    for analysis and assimilation of meteorological observations.*
    Tellus A 38(2).
11. Errico, R. M. (1997). *What is an adjoint model?* BAMS 78(11).
12. Talagrand, O. (1997). *Assimilation of observations, an
    introduction.* JMSJ 75(1B).

## Weak-constraint 4DVar (chapter 7)

13. Trémolet, Y. (2006). *Accounting for an imperfect model in
    4D-Var.* QJRMS 132(621).
14. Fisher, M., Leutbecher, M., & Kelly, G. A. (2005). *On the
    equivalence between Kalman smoothing and weak-constraint
    four-dimensional variational data assimilation.* QJRMS 131(613).
15. Daescu, D. N., & Todling, R. (2010). *Adjoint sensitivity of the
    model forecast to data assimilation system error covariance
    parameters.* QJRMS 136(653).

## Incremental 4DVar with CVT (chapter 8)

16. Courtier, P., Thépaut, J.-N., & Hollingsworth, A. (1994). *A
    strategy for operational implementation of 4D-Var, using an
    incremental approach.* QJRMS 120(519).
17. Lorenc, A. C. (1997). *Development of an operational variational
    assimilation scheme.* JMSJ 75(1B).
18. Bannister, R. N. (2017). *A review of operational methods of
    variational and ensemble-variational data assimilation.* QJRMS
    143(703).
19. Bannister, R. N. (2008). *A review of forecast error covariance
    statistics in atmospheric variational data assimilation.* QJRMS
    134(637).

## 4DVarNet (chapter 9)

20. Fablet, R., Amar, M. M., Febvre, Q., Beauchamp, M., & Chapron, B.
    (2021). *End-to-end physics-informed representation learning for
    satellite ocean remote sensing data.* ISPRS Annals V-3-2021,
    295–302.
21. Fablet, R., Chapron, B., Drumetz, L., Mémin, E., Pannekoucke, O.,
    & Rousseau, F. (2021). *Learning variational data assimilation
    models and solvers.* JAMES 13(10).
22. Fablet, R., Febvre, Q., & Chapron, B. (2023). *Multimodal 4DVarNets
    for the reconstruction of sea surface dynamics from NADIR and
    wide-swath altimetry.* IEEE TGRS 61.
23. Bolte, J., Pauwels, E., & Vaiter, S. (2023). *One-step
    differentiation of iterative algorithms.* NeurIPS 36.
    arXiv:2305.13768.
24. LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F.
    (2006). *A tutorial on energy-based learning.* Predicting
    Structured Data 1(0).

## Amortized inference (chapter 10)

25. Cranmer, K., Brehmer, J., & Louppe, G. (2020). *The frontier of
    simulation-based inference.* PNAS 117(48).
26. Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., &
    Lakshminarayanan, B. (2021). *Normalizing flows for probabilistic
    modeling and inference.* JMLR 22(57).
27. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S.,
    & Poole, B. (2021). *Score-based generative modeling through
    stochastic differential equations.* ICLR.
28. Cohen, S., Amos, B., & Lipman, Y. (2023). *Score-based diffusion
    meets annealed importance sampling.* NeurIPS.

## Adjoint methods (chapter 12)

29. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D.
    (2018). *Neural ordinary differential equations.* NeurIPS.
30. Kidger, P. (2021). *On neural differential equations.* PhD
    thesis, University of Oxford. (`diffrax` author; Ch. 5 on
    adjoints.)
31. Blondel, M., Berthet, Q., Cuturi, M., Frostig, R., Hoyer, S.,
    Llinares-López, F., Pedregosa, F., & Vert, J.-P. (2022).
    *Efficient and modular implicit differentiation.* NeurIPS.

## Posterior + ensemble methods (chapter 13)

32. Cressie, N., & Wikle, C. K. (2011). *Statistics for
    Spatio-Temporal Data.* Wiley.
33. Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A.
    (2018). *Validating Bayesian inference algorithms with
    simulation-based calibration.* arXiv:1804.06788.
34. Evensen, G. (2003). *The Ensemble Kalman Filter: theoretical
    formulation and practical implementation.* Ocean Dynamics 53.

## Examples (chapters 15–17)

35. Lorenz, E. N. (1963). *Deterministic nonperiodic flow.* JAS
    20(2).
36. Lorenz, E. N. (1996). *Predictability — a problem partly solved.*
    ECMWF Seminar.
37. Le Guillou, F., et al. (2023). *Mapping altimetry in the
    forthcoming SWOT era by back-and-forth nudging a one-layer
    quasigeostrophic model.* JTECH 40(1). (OceanBench reference.)
38. Ubelmann, C., Klein, P., & Fu, L.-L. (2015). *Dynamic
    interpolation of sea surface height and potential applications
    for future high-resolution altimetry mapping.* JTECH 32.
39. Jacob, D. J., et al. (2022). *Quantifying methane emissions from
    the global scale down to point sources using satellite
    observations of atmospheric methane.* ACP 22(14).
40. Varon, D. J., et al. (2019). *Quantifying methane point sources
    from fine-scale satellite observations of atmospheric methane
    plumes.* AMT 12(10).
41. Cusworth, D. H., et al. (2021). *Multisatellite imaging of a gas
    well blowout enables quantification of total methane emissions.*
    GRL 48(2).

## Ecosystem libraries

- [`somax`](https://github.com/jejjohnson/somax) — geophysical fluid
  models
- `plumax` — atmospheric transport + RTM (methane)
- [`gaussx`](https://github.com/jejjohnson/gaussx) — structured
  linear operators, Matérn factorisation
- [`lineax`](https://github.com/patrick-kidger/lineax) — linear
  solvers, `AbstractLinearOperator`
- [`optimistix`](https://github.com/patrick-kidger/optimistix) —
  optimisers + adjoints
- [`diffrax`](https://github.com/patrick-kidger/diffrax) — ODE / SDE
  integration + adjoints
- `filterax` — ensemble methods
- [`pipekit`](https://github.com/jejjohnson/pipekit) +
  `pipekit-cycle` — operator composition + DA cycle protocols

## Predecessors

- [mvardax](https://github.com/jejjohnson/mvardax) (deprecated)
- [CIA-Oceanix/4dvarnet-starter](https://github.com/CIA-Oceanix/4dvarnet-starter) —
  reference 4DVarNet implementation in PyTorch

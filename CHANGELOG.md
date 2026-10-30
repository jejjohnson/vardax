# Changelog

## [0.2.1](https://github.com/jejjohnson/vardax/compare/v0.2.0...v0.2.1) (2026-10-30)


### Features

* add Rodgers optimal estimation with its posterior, averaging kernel and information content ([#187](https://github.com/jejjohnson/vardax/issues/187)) ([0deaaf8](https://github.com/jejjohnson/vardax/commit/0deaaf84b926268a6f62682e15352526be97efb6))
* **models:** add a reverse-mode-only Gauss–Newton to IncrementalFourDVar ([#181](https://github.com/jejjohnson/vardax/issues/181)) ([0aab1c1](https://github.com/jejjohnson/vardax/commit/0aab1c1f6f11ff4adad506f8e615347e72841e94)), closes [#162](https://github.com/jejjohnson/vardax/issues/162)
* **models:** add KalmanSmoother, a linear-Gaussian filter and RTS smoother over irregular times ([#180](https://github.com/jejjohnson/vardax/issues/180)) ([a8fd36b](https://github.com/jejjohnson/vardax/commit/a8fd36bee36dd5f5dc852e857c36430f0954dd43)), closes [#164](https://github.com/jejjohnson/vardax/issues/164)
* **models:** add the state-space (information) form to OptimalInterpolation ([#179](https://github.com/jejjohnson/vardax/issues/179)) ([2f81e10](https://github.com/jejjohnson/vardax/commit/2f81e10093678f5f05dffdd502d1ceebd79f871e)), closes [#163](https://github.com/jejjohnson/vardax/issues/163)
* **models:** reduced incremental 4D-Var through a rectangular control-variable transform ([#183](https://github.com/jejjohnson/vardax/issues/183)) ([4c6249e](https://github.com/jejjohnson/vardax/commit/4c6249e2c07112856c4a9c1d77e73ab5cd140039)), closes [#155](https://github.com/jejjohnson/vardax/issues/155)
* **models:** reduced model error in weak-constraint 4D-Var ([#184](https://github.com/jejjohnson/vardax/issues/184)) ([3bd830b](https://github.com/jejjohnson/vardax/commit/3bd830b0b019fd8577fc919c73fe45aa2c272413)), closes [#156](https://github.com/jejjohnson/vardax/issues/156)
* **obs:** make MultiInstrumentFusion.weights weight the fused cost and the flattened R ([#176](https://github.com/jejjohnson/vardax/issues/176)) ([b2920c3](https://github.com/jejjohnson/vardax/commit/b2920c33cf2438fad3c5bff884e0abb438a6e95b)), closes [#145](https://github.com/jejjohnson/vardax/issues/145)
* **posterior:** draw samples from a Laplace or Gauss–Newton posterior ([#178](https://github.com/jejjohnson/vardax/issues/178)) ([ee11d85](https://github.com/jejjohnson/vardax/commit/ee11d85a42718219ba3a430ddc4d670608f14bf0)), closes [#165](https://github.com/jejjohnson/vardax/issues/165)


### Bug Fixes

* **deps:** pin gaussx to v0.6.5 so its structured covariances work in every model ([#166](https://github.com/jejjohnson/vardax/issues/166)) ([dbb7d44](https://github.com/jejjohnson/vardax/commit/dbb7d44cc74f0d7458aabaee45a28364ca282377)), closes [#141](https://github.com/jejjohnson/vardax/issues/141)
* **models:** compute the OptimalInterpolation posterior in observation space ([#171](https://github.com/jejjohnson/vardax/issues/171)) ([6cf5690](https://github.com/jejjohnson/vardax/commit/6cf5690fdd460e45dda5c4cce9bf7e6be67c972f)), closes [#143](https://github.com/jejjohnson/vardax/issues/143)
* **models:** ignore NaN in masked-out cells instead of returning the background ([#174](https://github.com/jejjohnson/vardax/issues/174)) ([102961d](https://github.com/jejjohnson/vardax/commit/102961d885f843cede3c91c6726ddc564a2a4a56)), closes [#147](https://github.com/jejjohnson/vardax/issues/147)
* **models:** make the 4D-Var analysis steps analyse around the cycle's forecast ([#186](https://github.com/jejjohnson/vardax/issues/186)) ([0897786](https://github.com/jejjohnson/vardax/commit/089778660658de077a94bfb21cd82f3af07ae6db)), closes [#79](https://github.com/jejjohnson/vardax/issues/79)
* **models:** tag covariance operators PSD at construction so the default CG accepts σ² I ([#167](https://github.com/jejjohnson/vardax/issues/167)) ([7498c66](https://github.com/jejjohnson/vardax/commit/7498c669e3db39a90d52162fecf9df8cbece072c)), closes [#146](https://github.com/jejjohnson/vardax/issues/146)
* **posterior:** accept the (S, N) batches of (N,) states that OptimalInterpolation analyses ([#170](https://github.com/jejjohnson/vardax/issues/170)) ([0a48fd9](https://github.com/jejjohnson/vardax/commit/0a48fd935597adcdf864a1360bb2668ec963e25a)), closes [#142](https://github.com/jejjohnson/vardax/issues/142)
* **posterior:** build the cvt IncrementalFourDVar posterior over the whitened control ([#172](https://github.com/jejjohnson/vardax/issues/172)) ([c69e9c0](https://github.com/jejjohnson/vardax/commit/c69e9c0ff90dcb507c70d3bd920325ee8598b62c)), closes [#161](https://github.com/jejjohnson/vardax/issues/161)
* **priors:** differentiate DynTrajectory forward-model steps with DirectAdjoint ([#177](https://github.com/jejjohnson/vardax/issues/177)) ([1708d63](https://github.com/jejjohnson/vardax/commit/1708d6304e690c425cd1f1d6893a6b8803c65a9d)), closes [#148](https://github.com/jejjohnson/vardax/issues/148)

## [0.2.0](https://github.com/jejjohnson/vardax/compare/v0.1.13...v0.2.0) (2026-10-28)


### ⚠ BREAKING CHANGES

* public fields and keyword arguments are renamed, with no deprecated aliases:

### Features

* get_config / from_config for hyperparameter-only modules ([#135](https://github.com/jejjohnson/vardax/issues/135)) ([7023b81](https://github.com/jejjohnson/vardax/commit/7023b81f3fe9fb426a4337e243b4d479a0e01363))
* **minimisers:** report whether the minimiser converged ([#128](https://github.com/jejjohnson/vardax/issues/128)) ([24fbc72](https://github.com/jejjohnson/vardax/commit/24fbc727b42e3d66dd2633b1458bfda450402afe)), closes [#127](https://github.com/jejjohnson/vardax/issues/127)


### Code Refactoring

* descriptive names for public fields and parameters ([#139](https://github.com/jejjohnson/vardax/issues/139)) ([d002e43](https://github.com/jejjohnson/vardax/commit/d002e4379d35e3109e394f6da737fe32c8a2c458))

## [0.1.13](https://github.com/jejjohnson/vardax/compare/v0.1.12...v0.1.13) (2026-10-27)


### Features

* **models:** expose the minimiser diagnostics of a ThreeDVar analysis ([#125](https://github.com/jejjohnson/vardax/issues/125)) ([dfe464c](https://github.com/jejjohnson/vardax/commit/dfe464cbf50556a4b4f335569e1b68600dede4d0)), closes [#124](https://github.com/jejjohnson/vardax/issues/124)
* **models:** make the covariance solves configurable with cov_solver ([#121](https://github.com/jejjohnson/vardax/issues/121)) ([6189186](https://github.com/jejjohnson/vardax/commit/61891867c77ee4724c5a873aca5cc4d96c298cb7)), closes [#117](https://github.com/jejjohnson/vardax/issues/117)
* **obs_operators:** let CallableObs linearise reverse-mode-only models ([#119](https://github.com/jejjohnson/vardax/issues/119)) ([e3d0ccd](https://github.com/jejjohnson/vardax/commit/e3d0ccd9239211ea834bf0eb95ac25019de04289))
* **oi:** accept a linear solver for the innovation system ([#120](https://github.com/jejjohnson/vardax/issues/120)) ([65b49ed](https://github.com/jejjohnson/vardax/commit/65b49edb040d3ec7c41cf299016758917aed4070)), closes [#116](https://github.com/jejjohnson/vardax/issues/116)
* **posterior:** add Laplace and Gauss-Newton covariance for a linear control ([#122](https://github.com/jejjohnson/vardax/issues/122)) ([5e5cae9](https://github.com/jejjohnson/vardax/commit/5e5cae909c42bff55bf904fc6fe81376188ba1d6))

## [0.1.12](https://github.com/jejjohnson/vardax/compare/v0.1.11...v0.1.12) (2026-10-27)


### Features

* **obs_operators:** add CallableObs, any JAX function as an observation operator ([#113](https://github.com/jejjohnson/vardax/issues/113)) ([68b411b](https://github.com/jejjohnson/vardax/commit/68b411bf9b04dc88038224f9c643ef751d03de6c)), closes [#112](https://github.com/jejjohnson/vardax/issues/112)

## [0.1.11](https://github.com/jejjohnson/vardax/compare/v0.1.10...v0.1.11) (2026-10-26)


### Features

* control-variable transform for IncrementalFourDVar's inner loop ([#104](https://github.com/jejjohnson/vardax/issues/104)) ([1a0e396](https://github.com/jejjohnson/vardax/commit/1a0e3961b98794d959df0a8213952c7656eebd57))


### Bug Fixes

* honour forward_adjoint and add checkpointed rollouts to the 4D-Var models ([#102](https://github.com/jejjohnson/vardax/issues/102)) ([553b8c7](https://github.com/jejjohnson/vardax/commit/553b8c7b9099d50619a487cdb48524ca0ad40cde))
* use the tangent-linear model in the 4D-Var posterior adapters ([#103](https://github.com/jejjohnson/vardax/issues/103)) ([8718a75](https://github.com/jejjohnson/vardax/commit/8718a75eb5602ba3352816ffa62e89f3f02ad053))

## [0.1.10](https://github.com/jejjohnson/vardax/compare/v0.1.9...v0.1.10) (2026-10-23)


### Features

* **costs:** add ResidualPenalty and reference PDE residuals ([#90](https://github.com/jejjohnson/vardax/issues/90)) ([777d04a](https://github.com/jejjohnson/vardax/commit/777d04a693bf2d230b31ff77d360f45df67d26fb))
* **models:** add SpaceTimeFieldVar, neural-field 4d-var after oh 2025 ([#93](https://github.com/jejjohnson/vardax/issues/93)) ([6566976](https://github.com/jejjohnson/vardax/commit/65669767857f63383fd976c06d53ee0be32a590b))
* **obs:** add PointObsBatch, GridPointObs, FieldPointObs and point batches in the models ([#92](https://github.com/jejjohnson/vardax/issues/92)) ([f7bd3e5](https://github.com/jejjohnson/vardax/commit/f7bd3e5142af9645dd342558f65e8550644ce82b))
* **protocols:** add StateParameterization and parameterization= on the models ([#91](https://github.com/jejjohnson/vardax/issues/91)) ([d7572d3](https://github.com/jejjohnson/vardax/commit/d7572d390bf1fd78b73ac62e1742d9a3acb91219))
* **solvers:** add OptaxMinimiser and minimiser= on the classical models ([#89](https://github.com/jejjohnson/vardax/issues/89)) ([a36e32c](https://github.com/jejjohnson/vardax/commit/a36e32cad841b07500facc22d64117ebd149e6da))


### Bug Fixes

* **cycle:** analyse around the forecast in the OI and 3D-Var analysis steps ([#77](https://github.com/jejjohnson/vardax/issues/77)) ([623fe69](https://github.com/jejjohnson/vardax/commit/623fe690efb332c03913213a271acf90c09f838b))

## [0.1.9](https://github.com/jejjohnson/vardax/compare/v0.1.8...v0.1.9) (2026-07-25)


### Features

* general reduced-basis operators backed by geonnax ([#61](https://github.com/jejjohnson/vardax/issues/61)) ([ce9e975](https://github.com/jejjohnson/vardax/commit/ce9e975bb839652c910627d9b1f96b0174193699))
* k-step adjoints and the pipekit adjoint-spec interpreter ([#55](https://github.com/jejjohnson/vardax/issues/55)) ([b8ee08c](https://github.com/jejjohnson/vardax/commit/b8ee08c82ac87aec1eb8a43cca0f8606b3f25a6c))
* mfourdvar migration Phase 2 — temporal prior seam and varcost API (D18/D19) ([#62](https://github.com/jejjohnson/vardax/issues/62)) ([8a9aec8](https://github.com/jejjohnson/vardax/commit/8a9aec85b49bfa99dea1e4880f0acde79440fd16))
* port mfourdvar legacy components (Phase 1 of [#13](https://github.com/jejjohnson/vardax/issues/13)) ([#60](https://github.com/jejjohnson/vardax/issues/60)) ([bd7258b](https://github.com/jejjohnson/vardax/commit/bd7258b9e69cd0cfe092f8c46f3cf31f193844d0))
* reduced-basis 4DVar analysis step Incremental4DVar ([#50](https://github.com/jejjohnson/vardax/issues/50)) ([92ba369](https://github.com/jejjohnson/vardax/commit/92ba36911e3307b2f4f1ed5b47bdbc654b5ae8f5))
* sparse bilinear InterpObs observation operator ([#63](https://github.com/jejjohnson/vardax/issues/63)) ([f7ad607](https://github.com/jejjohnson/vardax/commit/f7ad6071f53b8633ecbb38b481619da1ed89b554))

## [0.1.8](https://github.com/jejjohnson/vardax/compare/v0.1.7...v0.1.8) (2026-05-27)


### Features

* phase 4 — pipekit cycle integration + amortized inference (epics 7, 8) ([#42](https://github.com/jejjohnson/vardax/issues/42)) ([5c2aac0](https://github.com/jejjohnson/vardax/commit/5c2aac002e841658869150139ed0fc2156e48b96))

## [0.1.7](https://github.com/jejjohnson/vardax/compare/v0.1.6...v0.1.7) (2026-05-27)


### Bug Fixes

* address review comments and unblock type check on PR [#30](https://github.com/jejjohnson/vardax/issues/30) ([9a283d0](https://github.com/jejjohnson/vardax/commit/9a283d04b16fac17c1455649a82af9629f95ea2c))
* **ci:** drop uv cache from Deploy Docs workflow ([db30d8e](https://github.com/jejjohnson/vardax/commit/db30d8edf7d12bf1fee574cd8905c59f343bbb2c))
* **ci:** drop uv cache from Deploy Docs workflow ([6043d7d](https://github.com/jejjohnson/vardax/commit/6043d7d0cfef743b4714a25b1d6244dbb630e0f4))

## [0.1.6](https://github.com/jejjohnson/vardax/compare/v0.1.5...v0.1.6) (2026-03-01)


### Features

* add one-step differentiation strategy and `grad_mode` parameter ([ec5d1c9](https://github.com/jejjohnson/vardax/commit/ec5d1c9dbd8950feb14ca7d02c9e69da1be45be6))
* add one-step differentiation strategy with GradMode parameter ([e9fd271](https://github.com/jejjohnson/vardax/commit/e9fd271a547fe4ab04dc2fc4e126b28a0df17798))


### Bug Fixes

* correct RUF059 lint errors and reformat test files for CI ([4c8dc68](https://github.com/jejjohnson/vardax/commit/4c8dc68d807fd8a49e1e64f6395a6813cec53a44))
* thread prior_weight through one-step solver, fix unused var, update docstrings ([9b8d834](https://github.com/jejjohnson/vardax/commit/9b8d8344b10d2d7e51d9ce7762586809f00c7624))

## [0.1.5](https://github.com/jejjohnson/vardax/compare/v0.1.4...v0.1.5) (2026-03-01)


### Bug Fixes

* remove unused jax import in test_priors and update notebooks to NNX API ([736a97a](https://github.com/jejjohnson/vardax/commit/736a97afabe5ad8924b70d8b1ed515b34360f749))

## [0.1.4](https://github.com/jejjohnson/vardax/compare/v0.1.3...v0.1.4) (2026-03-01)


### Features

* port l96 functionality ([bb259b7](https://github.com/jejjohnson/vardax/commit/bb259b792d8e1fc6ea3c740d5f492ff5813c4537))


### Bug Fixes

* add IdentityPrior class to priors.py lost during merge with main ([6ad893f](https://github.com/jejjohnson/vardax/commit/6ad893ff811bfe5edc62b124070ad715ce5407b9))
* use type: ignore comments for N annotation in Lorenz96.__call__ to fix ty check ([d928e8f](https://github.com/jejjohnson/vardax/commit/d928e8fb033870ccb830ede9685d25d43bb30c2c))

## [0.1.3](https://github.com/jejjohnson/vardax/compare/v0.1.2...v0.1.3) (2026-03-01)


### Bug Fixes

* address review comments and CI failures (conventional commits, lint, tests) ([a65f05c](https://github.com/jejjohnson/vardax/commit/a65f05c14e93eef6cb53abc35fc268aabce9f616))
* resolve ty type-check failure in obs_interpolation_init ([aeac8e4](https://github.com/jejjohnson/vardax/commit/aeac8e46880073b42bb927d089001201cdfde034))

## [0.1.2](https://github.com/jejjohnson/vardax/compare/v0.1.1...v0.1.2) (2026-03-01)


### Features

* migrate 4dvarjax → fourdvarjax utils subpackage (L63 simulation, xarray pipeline, viz) ([93e43d7](https://github.com/jejjohnson/vardax/commit/93e43d7202220d7d295255e5eeed61badabbe63e))
* migrate 4dvarjax functionality into fourdvarjax utils subpackage ([86d6597](https://github.com/jejjohnson/vardax/commit/86d65975dab4e7bb131231d7e40b4d527ce1cc2d))


### Bug Fixes

* resolve CI failures (ruff format + xarray import at test time) ([3489196](https://github.com/jejjohnson/vardax/commit/34891960fd24b0d04ce3bbb24b8f0e2a2aa03093))
* resolve ty type-check failures and standardize.py NaN/ZeroDiv bug ([5cffd38](https://github.com/jejjohnson/vardax/commit/5cffd3834725a6b03d1b37451d0c0365b1d316e5))
* suppress ty unresolved-attribute for set_zlabel on 3D axes ([80d5442](https://github.com/jejjohnson/vardax/commit/80d544226f50307a807a60749a3b5287775d9cb5))

## [0.1.1](https://github.com/jejjohnson/vardax/compare/v0.1.0...v0.1.1) (2026-03-01)


### Features

* bootstrap fourdvarjax — full project scaffold + 4DVarNet implementation ([ab52941](https://github.com/jejjohnson/vardax/commit/ab529416d5fa436c4fbedbf0dd0a4a791ad3fe22))
* scaffold fourdvarjax repo with full 4DVarNet implementation ([c8e2aa4](https://github.com/jejjohnson/vardax/commit/c8e2aa4268378dd682ae9ef4d3bd48f744397c7e))


### Bug Fixes

* add explicit permissions to pytest.yaml workflow ([153a073](https://github.com/jejjohnson/vardax/commit/153a07352caa521cb4c2e4c91d574a26bdb81332))
* address review comments - lint, unused vars, fit() init, obs_dim removal ([4e5cdde](https://github.com/jejjohnson/vardax/commit/4e5cddee53a7ec40b1fcc36d256ceee70c7dbf50))

## Changelog

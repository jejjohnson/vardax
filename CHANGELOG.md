# Changelog

## [0.1.7](https://github.com/jejjohnson/vardax/compare/v0.1.6...v0.1.7) (2026-05-27)


### Bug Fixes

* address review comments and unblock type check on PR [#30](https://github.com/jejjohnson/vardax/issues/30) ([9a283d0](https://github.com/jejjohnson/vardax/commit/9a283d04b16fac17c1455649a82af9629f95ea2c))
* **ci:** drop uv cache from Deploy Docs workflow ([db30d8e](https://github.com/jejjohnson/vardax/commit/db30d8edf7d12bf1fee574cd8905c59f343bbb2c))
* **ci:** drop uv cache from Deploy Docs workflow ([6043d7d](https://github.com/jejjohnson/vardax/commit/6043d7d0cfef743b4714a25b1d6244dbb630e0f4))

## [0.1.6](https://github.com/jejjohnson/fourdvarjax/compare/v0.1.5...v0.1.6) (2026-03-01)


### Features

* add one-step differentiation strategy and `grad_mode` parameter ([ec5d1c9](https://github.com/jejjohnson/fourdvarjax/commit/ec5d1c9dbd8950feb14ca7d02c9e69da1be45be6))
* add one-step differentiation strategy with GradMode parameter ([e9fd271](https://github.com/jejjohnson/fourdvarjax/commit/e9fd271a547fe4ab04dc2fc4e126b28a0df17798))


### Bug Fixes

* correct RUF059 lint errors and reformat test files for CI ([4c8dc68](https://github.com/jejjohnson/fourdvarjax/commit/4c8dc68d807fd8a49e1e64f6395a6813cec53a44))
* thread prior_weight through one-step solver, fix unused var, update docstrings ([9b8d834](https://github.com/jejjohnson/fourdvarjax/commit/9b8d8344b10d2d7e51d9ce7762586809f00c7624))

## [0.1.5](https://github.com/jejjohnson/fourdvarjax/compare/v0.1.4...v0.1.5) (2026-03-01)


### Bug Fixes

* remove unused jax import in test_priors and update notebooks to NNX API ([736a97a](https://github.com/jejjohnson/fourdvarjax/commit/736a97afabe5ad8924b70d8b1ed515b34360f749))

## [0.1.4](https://github.com/jejjohnson/fourdvarjax/compare/v0.1.3...v0.1.4) (2026-03-01)


### Features

* port l96 functionality ([bb259b7](https://github.com/jejjohnson/fourdvarjax/commit/bb259b792d8e1fc6ea3c740d5f492ff5813c4537))


### Bug Fixes

* add IdentityPrior class to priors.py lost during merge with main ([6ad893f](https://github.com/jejjohnson/fourdvarjax/commit/6ad893ff811bfe5edc62b124070ad715ce5407b9))
* use type: ignore comments for N annotation in Lorenz96.__call__ to fix ty check ([d928e8f](https://github.com/jejjohnson/fourdvarjax/commit/d928e8fb033870ccb830ede9685d25d43bb30c2c))

## [0.1.3](https://github.com/jejjohnson/fourdvarjax/compare/v0.1.2...v0.1.3) (2026-03-01)


### Bug Fixes

* address review comments and CI failures (conventional commits, lint, tests) ([a65f05c](https://github.com/jejjohnson/fourdvarjax/commit/a65f05c14e93eef6cb53abc35fc268aabce9f616))
* resolve ty type-check failure in obs_interpolation_init ([aeac8e4](https://github.com/jejjohnson/fourdvarjax/commit/aeac8e46880073b42bb927d089001201cdfde034))

## [0.1.2](https://github.com/jejjohnson/fourdvarjax/compare/v0.1.1...v0.1.2) (2026-03-01)


### Features

* migrate 4dvarjax → fourdvarjax utils subpackage (L63 simulation, xarray pipeline, viz) ([93e43d7](https://github.com/jejjohnson/fourdvarjax/commit/93e43d7202220d7d295255e5eeed61badabbe63e))
* migrate 4dvarjax functionality into fourdvarjax utils subpackage ([86d6597](https://github.com/jejjohnson/fourdvarjax/commit/86d65975dab4e7bb131231d7e40b4d527ce1cc2d))


### Bug Fixes

* resolve CI failures (ruff format + xarray import at test time) ([3489196](https://github.com/jejjohnson/fourdvarjax/commit/34891960fd24b0d04ce3bbb24b8f0e2a2aa03093))
* resolve ty type-check failures and standardize.py NaN/ZeroDiv bug ([5cffd38](https://github.com/jejjohnson/fourdvarjax/commit/5cffd3834725a6b03d1b37451d0c0365b1d316e5))
* suppress ty unresolved-attribute for set_zlabel on 3D axes ([80d5442](https://github.com/jejjohnson/fourdvarjax/commit/80d544226f50307a807a60749a3b5287775d9cb5))

## [0.1.1](https://github.com/jejjohnson/fourdvarjax/compare/v0.1.0...v0.1.1) (2026-03-01)


### Features

* bootstrap fourdvarjax — full project scaffold + 4DVarNet implementation ([ab52941](https://github.com/jejjohnson/fourdvarjax/commit/ab529416d5fa436c4fbedbf0dd0a4a791ad3fe22))
* scaffold fourdvarjax repo with full 4DVarNet implementation ([c8e2aa4](https://github.com/jejjohnson/fourdvarjax/commit/c8e2aa4268378dd682ae9ef4d3bd48f744397c7e))


### Bug Fixes

* add explicit permissions to pytest.yaml workflow ([153a073](https://github.com/jejjohnson/fourdvarjax/commit/153a07352caa521cb4c2e4c91d574a26bdb81332))
* address review comments - lint, unused vars, fit() init, obs_dim removal ([4e5cdde](https://github.com/jejjohnson/fourdvarjax/commit/4e5cddee53a7ec40b1fcc36d256ceee70c7dbf50))

## Changelog

# Contributing to vardax

Thanks for your interest. This guide is short because the workflow is
standard and the design docs do the heavy lifting.

## Where to read first

- [`docs/`](docs/) — math reference and design docs. Read
  [`design/vision.md`](docs/design/vision.md) and
  [`design/decisions.md`](docs/design/decisions.md) before proposing
  architectural changes.
- [`docs/design/boundaries.md`](docs/design/boundaries.md) — the
  ownership map and Epic 0–13 roadmap. If you're not sure whether your
  change belongs in vardax or upstream (`somax` / `plumax` / `gaussx`
  / `optimistix` / `diffrax` / `pipekit-cycle`), this doc will tell
  you.

## Development setup

```bash
git clone https://github.com/jejjohnson/vardax.git
cd vardax
make install            # uv sync --all-extras + pre-commit install
```

Requires Python ≥ 3.12, < 3.14, and [`uv`](https://docs.astral.sh/uv/).

## Pre-commit checklist

Before opening a PR, run:

```bash
uv run pytest tests/ -v               # 147 tests should pass
uv run ruff check .                   # lint
uv run ruff format --check .          # format
uv run ty check src/vardax            # typecheck
uv run --group docs mkdocs build --strict   # docs build
```

Or just `make uv-test && make uv-lint`. Pre-commit hooks run a subset
on every commit; CI runs the full set on every PR.

## Conventional commits

Commit messages follow the [Conventional Commits](https://www.conventionalcommits.org/)
spec. Release-please cuts versions automatically from the commit
history; the prefix matters:

| Prefix | Effect on version |
|---|---|
| `feat:` | Minor bump (or patch pre-1.0) |
| `fix:` | Patch bump |
| `feat!:` or `BREAKING CHANGE:` footer | Major bump (or minor pre-1.0) |
| `docs:` / `chore:` / `test:` / `refactor:` / `style:` / `build:` / `ci:` | Hidden from changelog, no version bump |

PR titles are validated by the `conventional-commits` workflow.

## PR review

- All four CI checks (tests, lint, format, typecheck) must pass.
- Docs build (`mkdocs build --strict`) must pass.
- For substantial changes, link to relevant design decisions in
  [`docs/design/decisions.md`](docs/design/decisions.md).
- Resolve review comment threads after addressing them.

## Adding a new analysis method

If you're adding a new analysis method (eighth class beyond the seven
documented), follow the existing pattern:

1. Implement under `src/vardax/_src/models/<method>.py`.
2. Satisfy `pipekit_cycle.AnalysisStep` via `.as_analysis_step()`.
3. Add the linear-Gaussian agreement test to
   `tests/test_linear_gaussian_agreement.py` — the new method must
   agree with `OptimalInterpolation` in the linear-Gaussian limit
   (Decision D14 invariant).
4. Add a math reference chapter under `docs/` following the style of
   chapters 4–10.
5. Update `docs/design/api/models.md` and the seven-method table in
   `README.md` and `docs/index.md`.

## License

By contributing, you agree your contributions are licensed under MIT.

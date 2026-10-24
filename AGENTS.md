# Copilot Agent Guidelines for vardax

This document defines the workflow rules that every Copilot coding-agent session
**must** follow when contributing to this repository.

---

## 1. Quality gates — run before every commit

Before calling `report_progress` (which commits and pushes), always verify that
all four quality gates pass locally.  Fix any failures before committing.

```bash
# Formatting — the whole repo, as the CI Format job does (tests included)
ruff format .

# Linting — the whole repo, as the CI Lint job does (tests included)
ruff check --no-fix .

# Type checking — the package only, as the CI Type Check job does
ty check src/vardax/

# Tests
python -m pytest tests/ -x -q
```

Ruff runs on `.` rather than `src/vardax/` because CI does: a lint error in
`tests/` fails the PR just like one in the package. `--no-fix` makes the
local check strict; `pyproject.toml` sets `fix = true`, so a plain
`ruff check` would silently rewrite files instead of reporting them.

Alternatively, use the Makefile shortcuts:

```bash
make uv-format   # ruff format . + ruff check --fix .
make uv-lint     # ruff format --check . + ruff check --no-fix . + ty check src/vardax
make uv-test     # pytest
```

All four must be **clean (exit 0)** before any commit is pushed.

---

## 2. Conventional Commits

Every **PR title** and every **commit message** must follow the
[Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<optional scope>): <subject>
```

- `type` must be one of: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`,
  `test`, `build`, `ci`, `chore`, `revert`
- `subject` must start with a **lowercase letter**
- No period at the end of the subject

**Valid examples:**

```
feat: add simulate_lorenz96 to dynamical_systems
fix: correct unused-variable lint error in priors
docs: add AGENTS.md with workflow guidelines
test: add tests for L96Prior forward pass
chore: auto-format dynamical_systems.py with ruff
```

**Invalid examples (do not use):**

```
Add Lorenz-96 support                  # missing type prefix
Feat: Add new feature                  # type must be lowercase; subject capitalised
feat: Add new feature.                 # subject must start lowercase; trailing period
```

---

## 3. PR title and description stability

- The **PR title** must be set once (using conventional-commit format) and
  **never changed** in subsequent sessions.
- The **PR description** is a living document.  Each session may only
  **append** new information (e.g. a new checklist section or bullet points).
  Do **not** rewrite or delete existing content.
- Use `report_progress` to incrementally update the description checklist;
  keep the overall structure consistent across updates.

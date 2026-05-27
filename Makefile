.PHONY: help install install_precommit
.DEFAULT_GOAL = help

# ANSI Color Codes for pretty terminal output
BLUE   := \033[36m
YELLOW := \033[33m
GREEN  := \033[32m
RED    := \033[31m
RESET  := \033[0m

PKGROOT = fourdvarjax
TESTS = tests
NOTEBOOKS_DIR = notebooks

help:	## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation
.PHONY: install
install: ## Install all project dependencies
	@printf "$(YELLOW)>>> Initiating environment synchronization and dependency installation...$(RESET)\n"
	@uv sync --all-extras
	@uv run pre-commit install
	@printf "$(GREEN)>>> Environment is ready and pre-commit hooks are active.$(RESET)\n"

.PHONY: uv-sync
uv-sync: ## Update lock file and sync dependencies using uv
	@printf "$(YELLOW)>>> Updating and syncing dependencies with uv...$(RESET)\n"
	@uv lock --upgrade
	@uv sync --all-extras
	@printf "$(GREEN)>>> uv environment synchronized.$(RESET)\n"

install_precommit: ## Install precommit tools
	pre-commit install --all-files

##@ Formatting
.PHONY: uv-format
uv-format: ## Run ruff formatter
	@printf "$(YELLOW)>>> Formatting code with ruff...$(RESET)\n"
	@uv run ruff format $(PKGROOT)
	@uv run ruff check --fix $(PKGROOT)
	@printf "$(GREEN)>>> Codebase formatted successfully.$(RESET)\n"

.PHONY: uv-lint
uv-lint: ## Run ruff check and ty
	@printf "$(YELLOW)>>> Executing static analysis and type checking...$(RESET)\n"
	@uv run ruff check $(PKGROOT)
	@uv run ty check $(PKGROOT)
	@printf "$(GREEN)>>> Linting checks passed.$(RESET)\n"

.PHONY: uv-pre-commit
uv-pre-commit: ## Run all pre-commit hooks
	@printf "$(YELLOW)>>> Running pre-commit hooks on all files...$(RESET)\n"
	@uv run pre-commit run --all-files
	@printf "$(GREEN)>>> Pre-commit checks passed.$(RESET)\n"

##@ Testing
.PHONY: uv-test
uv-test: ## Run pytest with coverage using uv
	@printf "$(YELLOW)>>> Launching test suite with verbosity...$(RESET)\n"
	@uv run pytest $(TESTS) -v
	@printf "$(GREEN)>>> All tests passed.$(RESET)\n"

.PHONY: uv-test-cov
uv-test-cov: ## Run pytest with coverage report
	@printf "$(YELLOW)>>> Running tests with coverage...$(RESET)\n"
	@uv run pytest $(TESTS) -v --cov=$(PKGROOT) --cov-report=xml:./coverage.xml
	@printf "$(GREEN)>>> Tests with coverage complete.$(RESET)\n"

##@ Notebooks (Jupytext)
.PHONY: nb-to-py
nb-to-py: ## Convert all .ipynb notebooks to .py (percent format)
	@printf "$(YELLOW)>>> Converting notebooks to Python scripts...$(RESET)\n"
	@uv run jupytext --to py:percent $(NOTEBOOKS_DIR)/*.ipynb 2>/dev/null || printf "$(YELLOW)>>> No .ipynb files found.$(RESET)\n"
	@printf "$(GREEN)>>> Conversion complete.$(RESET)\n"

.PHONY: nb-to-ipynb
nb-to-ipynb: ## Convert all .py notebooks to .ipynb
	@printf "$(YELLOW)>>> Converting Python scripts to notebooks...$(RESET)\n"
	@uv run jupytext --to notebook $(NOTEBOOKS_DIR)/*.py
	@printf "$(GREEN)>>> Conversion complete.$(RESET)\n"

.PHONY: nb-sync
nb-sync: ## Sync .py and .ipynb notebooks (update whichever is older)
	@printf "$(YELLOW)>>> Syncing notebooks...$(RESET)\n"
	@uv run jupytext --sync $(NOTEBOOKS_DIR)/*.py
	@printf "$(GREEN)>>> Notebooks synced.$(RESET)\n"

.PHONY: nb-clean
nb-clean: ## Remove all .ipynb files from notebooks directory
	@printf "$(YELLOW)>>> Removing .ipynb files...$(RESET)\n"
	@rm -f $(NOTEBOOKS_DIR)/*.ipynb
	@rm -rf $(NOTEBOOKS_DIR)/.ipynb_checkpoints
	@printf "$(GREEN)>>> Cleanup complete.$(RESET)\n"

##@ Documentation
.PHONY: docs
docs: ## Build documentation with mkdocs
	@printf "$(YELLOW)>>> Building docs...$(RESET)\n"
	@uv run --extra docs mkdocs build
	@printf "$(GREEN)>>> Docs built in site/$(RESET)\n"

.PHONY: docs-serve
docs-serve: ## Serve documentation locally at http://127.0.0.1:8000
	@uv run --extra docs mkdocs serve

.PHONY: docs-deploy
docs-deploy: ## Deploy documentation to GitHub Pages
	@printf "$(YELLOW)>>> Deploying docs to GitHub Pages...$(RESET)\n"
	@uv run --extra docs mkdocs gh-deploy --force
	@printf "$(GREEN)>>> Deployment complete.$(RESET)\n"

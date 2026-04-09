# Makefile - Reply AI Agent Challenge 2026
# Single-command environment setup for the entire repository.
#
# Usage:
#   make          - full setup (creates .venv, installs deps, registers Jupyter kernel)
#   make setup    - same as make
#   make check    - verify environment and .env credentials
#   make jupyter  - launch Jupyter Lab in the Learning Notebooks folder
#   make clean    - remove the root virtual environment

PYTHON      := python3
VENV        := .venv
PIP         := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
JUPYTER     := $(VENV)/bin/jupyter

.DEFAULT_GOAL := setup
.PHONY: setup check jupyter clean

# -----------------------------------------------------------------------
# setup: create root venv and install all dependencies
# -----------------------------------------------------------------------
setup: $(VENV)/bin/activate

$(VENV)/bin/activate: .setup/requirements.txt
	@echo ""
	@echo "--- Reply AI Agent Challenge 2026 - Environment Setup ---"
	@echo ""
	@$(PYTHON) --version
	@echo ""
	@echo "Step 1/3: Creating virtual environment at $(VENV)/ ..."
	@$(PYTHON) -m venv $(VENV)
	@echo "Step 2/3: Installing dependencies from .setup/requirements.txt ..."
	@$(PIP) install --upgrade pip --quiet
	@$(PIP) install -r .setup/requirements.txt --quiet
	@echo "Step 3/3: Registering Jupyter kernel (reply-challenge) ..."
	@$(PYTHON_VENV) -m ipykernel install --user --name=reply-challenge --display-name "Reply Challenge 2026"
	@touch $(VENV)/bin/activate
	@echo ""
	@echo "Setup complete."
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate the environment:  source $(VENV)/bin/activate"
	@echo "  2. Configure credentials:     cp .env.example .env  (then fill in your values)"
	@echo "  3. Verify everything works:   make check"
	@echo "  4. Launch Jupyter:            make jupyter"
	@echo ""

# -----------------------------------------------------------------------
# check: verify imports and .env credentials
# -----------------------------------------------------------------------
check:
	@echo ""
	@echo "--- Environment Check ---"
	@echo ""
	@$(PYTHON_VENV) .setup/check_setup.py

# -----------------------------------------------------------------------
# jupyter: launch Jupyter Lab in the learning notebooks folder
# -----------------------------------------------------------------------
jupyter:
	@echo "Launching Jupyter Lab at 00_AI_Agents_Learning/Notebooks/ ..."
	@$(JUPYTER) lab 00_AI_Agents_Learning/Notebooks/

# -----------------------------------------------------------------------
# clean: remove the virtual environment
# -----------------------------------------------------------------------
clean:
	@echo "Removing virtual environment at $(VENV)/ ..."
	@rm -rf $(VENV)
	@echo "Done. Run 'make setup' to recreate it."

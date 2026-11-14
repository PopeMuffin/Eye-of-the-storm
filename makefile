PIP := .venv/bin/pip3
PYTHON := .venv/bin/python3

all:
	make _run	

init:
	python3 -m venv .venv
	make require

require:
	$(PIP) install -r requirements.txt

_run:
	rm -rf out
	$(PYTHON) main.py

help:
	@echo "Available targets:"
	@echo "make                    - Runs main.py."
	@echo "make setup              - Sets up the python venv and installs dependencies."
	@echo ""
	@echo "Vars:"
	@echo "PYTHON=$(PYTHON)"
	@echo "PIP   =$(PIP)"


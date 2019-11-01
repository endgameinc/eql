#################
### EQL
#################

VENV := ./env/eql-build
VENV_BIN := $(VENV)/bin
PYTHON := $(VENV_BIN)/python
PIP := $(PYTHON) -m pip
SPHINXBUILD ?= $(VENV_BIN)/sphinx-build


$(VENV):
	pip install virtualenv
	virtualenv $(VENV)
	$(PIP) install -q -r requirements.txt
	$(PIP) install setuptools -U


.PHONY: clean
clean:
	rm -rf $(VENV) *.egg-info .eggs *.egg htmlcov build dist .build .tmp .tox *.egg-info .coverage coverage.xml junit.xml .pytest_cache
	find . -type f -name '*.pyc' -delete
	find . -type f -name '__pycache__' -delete

.PHONY: testdeps
testdeps:
	$(PIP) install -r requirements_test.txt

.PHONY: pytest
pytest: $(VENV) testdeps
	$(PYTHON) setup.py -q test


.PHONY: pylint
pylint: $(VENV) testdeps
	$(PYTHON) setup.py -q lint


.PHONY: test
test: $(VENV) pylint pytest


.PHONY: sdist
sdist: $(VENV)
	$(PYTHON) setup.py sdist


.PHONY: bdist_egg
bdist_egg: $(VENV)
	$(PYTHON) setup.py bdist_egg


.PHONY: bdist_wheel
bdist_wheel: $(VENV)
	$(PYTHON) setup.py bdist_wheel


.PHONY: install
install: $(VENV)
	$(PYTHON) setup.py install

.PHONY: all
all: sdist

.PHONY: docs
docs: $(VENV) install
	$(PIP) install sphinx sphinx_rtd_theme
	cd docs && ../$(SPHINXBUILD) -M html . _build


.PHONY: upload
upload: $(VENV)
	$(PIP) install twine~=1.13
	$(VENV_BIN)/twine upload dist/*

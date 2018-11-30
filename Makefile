#################
### EQL
#################

VENV := ./eql-env
VENV_BIN := $(VENV)/bin
PYTHON := $(VENV_BIN)/python
PIP := $(VENV_BIN)/pip

init:
	pip install virtualenv
	virtualenv $(VENV)
	$(VENV_BIN)/pip install -q -r requirements.txt

clean:
	rm -rf $(VENV) *.egg-info .eggs *.egg htmlcov build dist .build .tmp .tox

test:
	$(PYTHON) setup.py -q test

lint:
	$(PYTHON) setup.py -q lint

sdist:
	$(PYTHON) setup.py sdist

bdist_egg:
	$(PYTHON) setup.py bdist_egg

bdist_wheel:
	$(PYTHON) setup.py bdist_wheel

.PHONY: docs
docs:
	$(PIP) install sphinx sphinx_rtd_theme
	$(PYTHON) setup.py install
	$(VENV_BIN)/activate; cd docs; make html

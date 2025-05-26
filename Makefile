# SmartSurge Makefile

.PHONY: help install test lint docs docs-pdf docs-serve clean

help:
	@echo "SmartSurge Development Commands"
	@echo "=============================="
	@echo "install      Install package in development mode"
	@echo "test         Run tests"
	@echo "lint         Run linting and formatting"
	@echo "docs         Build MkDocs documentation"
	@echo "docs-pdf     Build PDF documentation"
	@echo "docs-serve   Serve documentation locally"
	@echo "clean        Clean build artifacts"

install:
	pip install -e ".[dev,docs]"

test:
	pytest tests/

lint:
	black src/smartsurge tests/
	isort src/smartsurge tests/
	flake8 src/smartsurge tests/
	mypy src/smartsurge

docs:
	mkdocs build

docs-pdf:
	python build_docs.py

docs-serve:
	mkdocs serve

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf docs/pdf/
	rm -rf site/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
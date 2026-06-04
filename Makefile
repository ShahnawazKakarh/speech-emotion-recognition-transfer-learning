.PHONY: help install install-dev lint format test clean download-ravdess prepare-meld train-all demo

help:
	@echo "Available targets:"
	@echo "  install         - Install package only"
	@echo "  install-dev     - Install with dev + demo extras"
	@echo "  lint            - Run ruff + mypy"
	@echo "  format          - Run black + ruff --fix"
	@echo "  test            - Run pytest"
	@echo "  download-ravdess - Download RAVDESS dataset"
	@echo "  prepare-meld    - Download and prepare MELD"
	@echo "  train-all       - Run all baseline experiments"
	@echo "  demo            - Launch Gradio demo"
	@echo "  clean           - Remove caches and build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,demo]"
	pre-commit install || true

lint:
	ruff check src tests
	mypy src --ignore-missing-imports || true

format:
	black src tests demo
	ruff check --fix src tests

test:
	pytest tests/ -v

download-ravdess:
	bash scripts/download_ravdess.sh

prepare-meld:
	bash scripts/prepare_meld.sh

train-all:
	bash scripts/run_all_experiments.sh

demo:
	python demo/gradio_app.py

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

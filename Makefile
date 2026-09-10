.PHONY: train-sklearn train-dhompo train-surabaya check-surabaya up down test test-features test-models test-api mlflow-ui install install-dev docs

# Training
train-sklearn: train-dhompo

train-dhompo:
	python training/dhompo/train_sklearn.py --config configs/shared/sklearn_model.yaml

train-surabaya:
	python training/surabaya/train_urban_sklearn.py --models ridge

check-surabaya:
	python training/surabaya/train_urban_sklearn.py --dry-run

# Docker
up:
	docker compose up --build -d

down:
	docker compose down

# Testing
test:
	python -m pytest tests/ -v

test-features:
	python -m pytest tests/test_urban_features.py tests/test_config_paths.py -v

test-models:
	python -m pytest tests/test_file_predictor.py tests/test_urban_file_predictor.py -v

test-api:
	python -m pytest tests/test_api.py -v

# MLflow
mlflow-ui:
	mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db

# Dev setup
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

docs:
	python -m mkdocs build --strict

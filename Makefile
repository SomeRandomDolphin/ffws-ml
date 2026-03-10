.PHONY: train-sklearn up down test mlflow-ui install install-dev

# Training
train-sklearn:
	python training/train_sklearn.py --config configs/sklearn_model.yaml

# Docker
up:
	docker-compose up --build -d

down:
	docker-compose down

# Testing
test:
	pytest tests/ -v

test-features:
	pytest tests/test_features.py -v

test-models:
	pytest tests/test_sklearn_models.py -v

test-api:
	pytest tests/test_api.py -v

# MLflow
mlflow-ui:
	mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db

# Dev setup
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

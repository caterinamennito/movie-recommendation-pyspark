.PHONY: help install test lint format clean run docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies and package"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean up build artifacts"
	@echo "  run         - Run the Streamlit app"
	@echo "  train       - Train the model"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run with Docker Compose"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests/ -v

lint:
	python -m flake8 src/ tests/
	python -m black --check src/ tests/
	python -m isort --check-only src/ tests/

format:
	python -m black src/ tests/
	python -m isort src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/

run:
	streamlit run src/app/streamlit_app.py

train:
	python -m src.main --mode train --small-dataset

docker-build:
	docker build -t movie-recommender .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down
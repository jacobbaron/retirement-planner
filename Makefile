# Retirement Planner Development Makefile
# This Makefile provides common development tasks for the retirement planner project

.PHONY: help install test lint typecheck format coverage check clean docker-up docker-down docker-test docker-check pre-commit-install pre-commit-run pre-commit-update

# Default target
help:
	@echo "Retirement Planner Development Commands:"
	@echo ""
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run tests with pytest"
	@echo "  make lint        - Run linting with flake8"
	@echo "  make typecheck   - Run type checking with mypy"
	@echo "  make format      - Format code with black and isort"
	@echo "  make coverage    - Run tests with coverage report"
	@echo "  make check       - Run all quality checks (test + lint + typecheck)"
	@echo "  make clean       - Clean up temporary files"
	@echo ""
	@echo "Pre-commit Commands:"
	@echo "  make pre-commit-install - Install pre-commit hooks"
	@echo "  make pre-commit-run     - Run pre-commit hooks on all files"
	@echo "  make pre-commit-update  - Update pre-commit hooks to latest versions"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-up   - Start Docker development environment"
	@echo "  make docker-down - Stop Docker development environment"
	@echo "  make docker-test - Run tests in Docker container"
	@echo "  make docker-check - Run all quality checks in Docker container"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	make typecheck
	pytest tests/ -v
	make lint

# Run linting
lint:
	flake8 app/ tests/ --max-line-length=88 --extend-ignore=E203,W503,E501 --per-file-ignores="tests/*:F401,F841"

# Run type checking
typecheck:
	mypy app/ --ignore-missing-imports

# Format code
format:
	black app/ tests/ --line-length=88
	isort app/ tests/ --profile=black

# Auto-fix common issues (unused imports/vars, formatting, imports)
autofix:
	autoflake --in-place --remove-all-unused-imports --remove-unused-variables -r app tests
	black app/ tests/ --line-length=88
	isort app/ tests/ --profile=black

# Run tests with coverage
coverage:
	pytest tests/ --cov=app --cov-report=term-missing --cov-report=html --cov-report=xml --cov-fail-under=80

# Run all quality checks
check: test lint typecheck coverage
	@echo "✅ All quality checks passed!"

# Clean up temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

# Docker commands
docker-up:
	docker compose up -d

docker-down:
	docker compose down

# Run tests in Docker container
docker-test:
	docker compose exec app make test

# Run linting in Docker container
docker-lint:
	docker compose exec app make lint

# Run type checking in Docker container
docker-typecheck:
	docker compose exec app make typecheck

# Run coverage in Docker container
docker-coverage:
	docker compose exec app make coverage

# Format code in Docker container
docker-format:
	docker compose exec app make format

# Run autofix inside Docker container
docker-autofix:
	docker compose exec app make autofix

# Run all quality checks in Docker container
docker-check:
	docker compose exec app make check

# Pre-commit hooks
pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate

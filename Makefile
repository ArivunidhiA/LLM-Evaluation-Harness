.PHONY: install dev test lint format serve docker-up docker-down clean

# Install production dependencies
install:
	pip install -e .

# Install with dev dependencies
dev:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v --cov=llm_eval --cov-report=term-missing

# Lint code
lint:
	mypy src/
	pylint src/llm_eval/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Start API server
serve:
	llm-eval serve --port 8000

# Run evaluation
run:
	llm-eval run --suite $(SUITE) --workers 50

# Docker commands
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-logs:
	docker-compose logs -f

# Database migrations
db-migrate:
	alembic upgrade head

db-revision:
	alembic revision --autogenerate -m "$(MSG)"

# Generate test cases
generate:
	python scripts/generate_test_cases.py --suite $(SUITE) --count $(COUNT)

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +

.PHONY: install test lint train forecast backtest drift run demo clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=gridcast --cov-report=term-missing

lint:
	ruff check gridcast tests

train:
	python -m gridcast.cli train

backtest:
	python -m gridcast.cli backtest

drift:
	python -m gridcast.cli drift

run:
	uvicorn gridcast.main:app --reload --port 8000

demo:
	python -m gridcast.cli demo

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov artifacts/* mlruns
	touch artifacts/.gitkeep

.PHONY: install style typecheck test

install:
	uv sync

style:
	uv run --frozen ruff check --fix .
	uv run --frozen ruff format .

typecheck:
	uv run --frozen basedpyright

test:
	uv run --frozen python -m unittest discover -s tests -v



test:
	pytest tests/ --cov=microcalibrate --cov-report=xml --maxfail=0 -v

install:
	pip install -e ".[dev]"

check-format:
	linecheck .
	isort --check-only --profile black src/
	black . -l 79 --check

format:
	linecheck . --fix
	isort --profile black src/
	black . -l 79

documentation:
	cd docs && jupyter-book build .
	python docs/add_plotly_to_book.py docs/_build/html

build:
	pip install build
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info/
	rm -rf docs/_build/

changelog:
	python .github/bump_version.py
	towncrier build --yes --version $$(python -c "import re; print(re.search(r'version = \"(.+?)\"', open('pyproject.toml').read()).group(1))")
dashboard-install:
	cd microcalibration-dashboard && bun install --frozen-lockfile

dashboard-dev:
	cd microcalibration-dashboard && bun run dev

dashboard-build:
	cd microcalibration-dashboard && bun run build

dashboard-start:
	cd microcalibration-dashboard && bunx serve@14.2.6 out

dashboard-clean:
	cd microcalibration-dashboard && rm -rf .next out node_modules

dashboard-static:
	cd microcalibration-dashboard && bun run static

dashboard-preview:
	cd microcalibration-dashboard && bun run static && bunx serve@14.2.6 out

dashboard-check:
	cd microcalibration-dashboard && bun run lint && bun run test && bun run static && echo "✅ Dashboard lint, tests and static build passed"

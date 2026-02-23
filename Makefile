

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
	cd microcalibration-dashboard && npm install

dashboard-dev:
	cd microcalibration-dashboard && npm run dev

dashboard-build:
	cd microcalibration-dashboard && npm run build

dashboard-start:
	cd microcalibration-dashboard && npm start

dashboard-clean:
	cd microcalibration-dashboard && rm -rf .next node_modules

dashboard-static:
	cd microcalibration-dashboard && npm run static

dashboard-preview:
	cd microcalibration-dashboard && npm run static && npx serve out

dashboard-check:
	cd microcalibration-dashboard && npm run lint && npm run static && echo "✅ Dashboard ready for GitHub Pages deployment"

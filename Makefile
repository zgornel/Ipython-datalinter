# Makefile for building and publishing the package
#
# Usage:
#   make build    - Build source and wheel distributions
#   make upload   - Upload distributions to PyPI (requires twine credentials)
#   make publish  - Build and upload in one step
#   make clean    - Remove build artifacts

.PHONY: build upload publish clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  make build    - Build sdist and wheel into dist/"
	@echo "  make upload   - Upload dist/* to PyPI via twine"
	@echo "  make publish  - Build then upload"
	@echo "  make clean    - Remove dist/ and build artifacts"

build:
	python -m build

upload:
	python -m twine upload dist/* --verbose

publish: build upload

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

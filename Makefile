PYTHON ?= python3
CONSTRAINTS ?= constraints.txt

.PHONY: help bump-version commit-version ensure-clean git-release test build-check pypi test-pypi publish

help:
	@echo "Targets:"
	@echo "  bump-version [PART=minor|major]     - bump version (defaults to patch)"
	@echo "  commit-version                      - commit version bump files"
	@echo "  ensure-clean                        - fail if git tree is dirty"
	@echo "  test                                - run unit tests"
	@echo "  build-check                         - build dist artifacts and run twine metadata checks"
	@echo "  git-release                         - create git tag v<version> and push"
	@echo "  pypi                                - upload prebuilt dist artifacts to PyPI"
	@echo "  test-pypi                           - upload prebuilt dist artifacts to TestPyPI via twine"

ensure-clean:
	@git diff --quiet && git diff --cached --quiet || (echo "Working tree is dirty; commit or stash changes first." && exit 1)

bump-version:
	@if [ -n "$(PART)" ]; then \
		PART_LABEL=$(PART); \
		$(PYTHON) scripts/bump_version.py $${PART_LABEL}; \
	else \
		PART_LABEL=patch; \
		$(PYTHON) scripts/bump_version.py; \
	fi; \
	echo "Version bumped ($$PART_LABEL)."

commit-version:
	@if git diff --quiet -- mlx_genkit/__init__.py pyproject.toml; then \
		echo "No version changes to commit."; \
	else \
		VERSION=$$(sed -n "s/^__version__ = ['\"]\\(.*\\)['\"]/\1/p" mlx_genkit/__init__.py); \
		if [ -z "$$VERSION" ]; then echo "Could not read version"; exit 1; fi; \
		git add mlx_genkit/__init__.py pyproject.toml && git commit -m "chore: bump version to v$${VERSION}"; \
	fi

git-release:
	@$(MAKE) ensure-clean
	@VERSION=$$(sed -n "s/^__version__ = ['\"]\(.*\)['\"]/\1/p" mlx_genkit/__init__.py); \
	if [ -z "$$VERSION" ]; then echo "Could not read version"; exit 1; fi; \
	echo "Tagging v$$VERSION"; \
	git tag v$$VERSION && git push origin v$$VERSION && git push

test:
	$(PYTHON) -m unittest discover -s mlx_genkit/tests -p 'test_*.py' -q

build-check:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -c $(CONSTRAINTS) build twine
	rm -rf dist/ build/
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

pypi:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -c $(CONSTRAINTS) twine
	@test -n "$$(ls dist/* 2>/dev/null)" || (echo "dist/ is empty; run 'make build-check' first." && exit 1)
	$(PYTHON) -m twine upload dist/*

test-pypi:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -c $(CONSTRAINTS) twine
	@test -n "$$(ls dist/* 2>/dev/null)" || (echo "dist/ is empty; run 'make build-check' first." && exit 1)
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish:
	@$(MAKE) ensure-clean
	@echo "==> Bumping patch version"
	$(MAKE) bump-version
	@echo "==> Committing version bump"
	$(MAKE) commit-version
	@echo "==> Running unit tests"
	$(MAKE) test
	@echo "==> Building and validating distribution artifacts"
	$(MAKE) build-check
	@VERSION=$$(sed -n "s/^__version__ = ['\"]\(.*\)['\"]/\1/p" mlx_genkit/__init__.py); \
		echo "==> Uploading to PyPI"; \
		$(MAKE) pypi; \
		echo "==> Tagging and pushing git release"; \
		$(MAKE) git-release; \
		echo "==> Creating GitHub release v$${VERSION}"; \
		gh release create v$${VERSION} dist/* --generate-notes

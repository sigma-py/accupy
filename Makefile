VERSION=$(shell python3 -c "import accupy; print(accupy.__version__)")

default:
	@echo "\"make publish\"?"

# https://packaging.python.org/distributing/#id72
upload: setup.py
	# Make sure we're on the master branch
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	rm -rf dist/*
	python3 setup.py sdist
	twine upload dist/*.tar.gz

tag:
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@echo "Tagging v$(VERSION)..."
	git tag v$(VERSION)
	git push --tags

publish: tag upload

clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	@rm -rf *.egg-info/ build/ dist/ MANIFEST

black:
	black setup.py accupy/ test/*.py

lint:
	black --check setup.py accupy/ test/*.py
	flake8 setup.py accupy/ test/*.py

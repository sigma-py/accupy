VERSION=$(shell python3 -c "from configparser import ConfigParser; p = ConfigParser(); p.read('setup.cfg'); print(p['metadata']['version'])")

default:
	@echo "\"make publish\"?"

# https://packaging.python.org/distributing/#id72
upload: clean
	# Make sure we're on the main branch
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi
	python3 setup.py sdist
	twine upload dist/*.tar.gz

tag:
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "main" ]; then exit 1; fi
	# @echo "Tagging v$(VERSION)..."
	# git tag v$(VERSION)
	# git push --tags
	curl -H "Authorization: token `cat $(HOME)/.github-access-token`" -d '{"tag_name": "v$(VERSION)"}' https://api.github.com/repos/nschloe/accupy/releases

publish: tag upload

clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	@rm -rf *.egg-info/ build/ dist/ MANIFEST

format:
	isort .
	black .
	blacken-docs README.md

black:
	black .

lint:
	black --check .
	flake8 .

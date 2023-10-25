default: help
.PHONY: install format lint clean_cache clean_data

help:
	@echo "install - install dependencies"
	@echo "format - format code with black"
	@echo "lint - lint code with flake8"
	@echo "clean_cache - remove Python cache files"
	@echo "clean_data - remove data files and directories"

install: requirements.txt
	pip install -r requirements.txt

format:
	# Format Python files in all dirs
	find . -name '*.py' -exec black {} +

lint:
	# Lint Python files in all dirs
	find . -name '*.py' -exec flake8 {} +

clean_cache:
	rm -rf src/__pycache__
	rm -rf .ipynb_checkpoints

clean_data:
	rm -rf landmark_images
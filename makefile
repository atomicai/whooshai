LINE_WIDTH=132
NAME := $(shell python setup.py --name)
UNAME := $(shell uname -s)
ISORT_FLAGS=--line-width=${LINE_WIDTH} --profile black
FLAKE_FLAGS=--remove-unused-variables --ignore-init-module-imports --recursive
# "" is for multi-lang strings (comments, logs), '' is for everything else.
BLACK_FLAGS=--skip-string-normalization --line-length=${LINE_WIDTH}
PYTEST_FLAGS=-p no:warnings
export FLASK_APP=whooshai.tdk.rises

install:
	pip install -e '.[all]'

setup-pre-commit:
	pip install -q pre-commit
	pre-commit install
  	# To check whole pipeline.
	pre-commit run --all-files

format:
	isort ${ISORT_FLAGS} --check-only --diff ${NAME} whooshai test
	black ${BLACK_FLAGS} --check --diff ${NAME} whooshai test
	autoflake ${FLAKE_FLAGS} --in-place ${NAME} whooshai test

format-fix:
	isort ${ISORT_FLAGS} ${NAME} whooshai test
	black ${BLACK_FLAGS} ${NAME} whooshai test
	autoflake ${FLAKE_FLAGS} ${NAME} whooshai test

run:
	celery -A whooshai.tdk.fire.flow worker -P gevent --loglevel=info --detach
	gunicorn -k gevent -w 4 -b 127.0.0.1:4321  whooshai.tdk.rises:app

rises:
	gunicorn -k gevent -w 4 -b 127.0.0.1:4321  whooshai.tdk.rises:app

fire:
	celery -A whooshai.tdk.fire.flow worker -P gevent --concurrency=1 --loglevel=info

stop:
	for pid in $$(ps aux | grep celery | grep -v grep | awk '{print $$2}'); do kill -9 "$$pid"; done

test:
	pytest test ${PYTEST_FLAGS} --testmon --suppress-no-test-exit-code

test-all:
	pytest test ${PYTEST_FLAGS}

clean:
	rm -rf .ipynb_checkpoints
	rm -rf **/.ipynb_checkpoints
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf downloads
	rm -rf wandb
	find . -name ".DS_Store" -print -delete
	rm -rf .cache
	pyclean .
test:
	poetry run pytest

lint:
	poetry run flake8 --exclude=venv* --statistics --count src tests

install:
	poetry install

typecheck:
	poetry run pyright src tests

run:
	poetry run python src/vo/app.py
lint: fmt
	fd -e py | xargs ruff check

fmt:
	fd -e py | xargs isort
	fd -e py | xargs black

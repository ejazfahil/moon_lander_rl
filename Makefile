.PHONY: test clean
test:
	pytest tests/ -v
clean:
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete

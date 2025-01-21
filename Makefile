.PHONY: setup clean test lint format install run-soapnuts run-woolballs run-all run-frontend

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -e .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type f -name "*.log" -delete

test:
	python -m pytest tests/

lint:
	black src/ tests/ frontend/
	flake8 src/ tests/ frontend/
	mypy src/ tests/ frontend/

format:
	black src/ tests/ frontend/
	isort src/ tests/ frontend/

install:
	pip install -r requirements.txt
	pip install -e .

run-soapnuts:
	python src/run_experiment.py \
		--data_path data/soapnutshistory.csv \
		--experiment_name soapnuts \
		--base_dir experiments \
		--num_eval_episodes 10

run-woolballs:
	python src/run_experiment.py \
		--data_path data/woolballhistory.csv \
		--experiment_name woolballs \
		--base_dir experiments \
		--num_eval_episodes 10

run-all:
	python run_all.py

run-frontend:
	streamlit run frontend/app.py 
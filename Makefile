install:
	pip install -r requirements.txt

install-dl:
	pip install -r requirements-dl.txt

install-optional:
	pip install -r requirements-optional.txt

data:
	python scripts/make_dataset.py

check:
	python scripts/run_data_check.py

features:
	python scripts/run_feature_engineering.py

traditional:
	python scripts/run_traditional_ml.py

clustering:
	python scripts/run_clustering.py

reduction:
	python scripts/run_dimensionality_reduction.py

shap:
	python scripts/run_shap_analysis.py

deep-learning:
	python scripts/run_deep_learning.py

all:
	python scripts/run_all.py

test:
	pytest tests/

clean:
	rm -rf data/processed/*
	rm -rf outputs/models/*
	rm -rf outputs/reports/*
	rm -rf outputs/figures/*
	rm -rf outputs/predictions/*
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
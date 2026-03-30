.PHONY: help install setup test lint clean run data models deploy

help:
	@echo "Hospital Readmission Predictor - Available Commands"
	@echo "===================================================="
	@echo "make install    - Install dependencies"
	@echo "make setup     - Full setup (install + download data)"
	@echo "make data      - Download dataset"
	@echo "make models    - Train models"
	@echo "make run       - Run Streamlit dashboard"
	@echo "make test      - Run tests"
	@echo "make lint      - Run linters"
	@echo "make clean     - Clean cache files"
	@echo "make deploy    - Deploy to Streamlit Cloud"

install:
	pip install -r requirements.txt

setup: install data
	pip install -e .

data:
	python download_data.py

models:
	jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb
	jupyter nbconvert --to notebook --execute notebooks/02_Preprocessing.ipynb
	jupyter nbconvert --to notebook --execute notebooks/03_Modeling.ipynb

run:
	streamlit run app.py

test:
	pytest tests/ -v

lint:
	flake8 src/ app.py
	black --check src/ app.py
	mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info

deploy:
	@echo "Deploy to Streamlit Cloud:"
	@echo "1. Push code to GitHub"
	@echo "2. Go to https://share.streamlit.io"
	@echo "3. Connect your repository"
	@echo "4. Deploy!"

docker-build:
	docker build -t hospital-readmission-predictor .

docker-run:
	docker run -p 8501:8501 hospital-readmission-predictor

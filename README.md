# 🏥 Hospital Readmission Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.24+-blue.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/XGBoost-1.7+-blue.svg" alt="XGBoost">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img https://img.shields.io/badge/Tests-Passing-brightgreen.svg alt="Tests">
</p>

> End-to-end machine learning project to predict 30-day hospital readmissions for diabetic patients.

## 📋 Problem Statement

Hospital readmissions for diabetic patients represent a significant challenge for healthcare systems, affecting patient outcomes and generating substantial costs. This project develops a machine learning solution to **predict 30-day hospital readmissions**, enabling healthcare providers to identify high-risk patients and implement proactive intervention strategies.

### Business Impact

| Impact Area | Description |
|-------------|-------------|
| **Cost Reduction** | Reduce hospital readmission costs by 10-20% |
| **Patient Outcomes** | Early identification leads to better care coordination |
| **Resource Optimization** | Allocate case management resources more effectively |
| **Quality Metrics** | Improve Hospital Quality (HQ) ratings |

---

## 🎯 Project Overview

This end-to-end machine learning project predicts whether a diabetic patient will be readmitted to the hospital within 30 days:

- 📊 **Exploratory Data Analysis (EDA)** - Data exploration & visualization
- 🧹 **Data Preprocessing** - Cleaning, feature engineering, SMOTE
- 🤖 **Model Training** - XGBoost, LightGBM, Random Forest with hyperparameter tuning
- 📈 **Interactive Dashboard** - Streamlit web application
- ☁️ **Cloud Deployment** - Streamlit Cloud / Hugging Face / Docker

---

## 🗂️ Project Structure

```
Hospital-Readmission-Predictor/
├── README.md                    # Project documentation
├── LICENSE                     # MIT License
├── setup.py                    # Package installation
├── Makefile                   # Development commands
├── requirements.txt            # Python dependencies
├── pyproject.toml            # Project metadata
│
├── app.py                     # Streamlit dashboard
├── download_data.py           # Dataset download script
│
├── configs/
│   └── config.yaml           # Configuration file
│
├── data/                      # Data directory
│   └── diabetic_data.csv     # Dataset (after download)
│
├── models/                    # Trained models
│   ├── best_model.pkl
│   ├── feature_cols.pkl
│   └── model_metrics.pkl
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_EDA.ipynb         # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb # Data Cleaning & Feature Engineering
│   ├── 03_Modeling.ipynb     # Model Training & Evaluation
│   └── 04_Deployment.ipynb  # Deployment Instructions
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_cleaning.py      # Data cleaning utilities
│   ├── modeling.py           # ML model utilities
│   └── visualization.py      # Plotting utilities
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_model.py
│
├── docker/                    # Docker files
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── dashboard/                 # Dashboard screenshots
```

---

## 🚀 Quick Start

### Using Make (Recommended)

```bash
# Clone and setup
git clone https://github.com/logeshkannan19/Hospital-Readmission-Predictor.git
cd Hospital-Readmission-Predictor

# Install dependencies and download data
make setup

# Train models
make models

# Run dashboard
make run
```

### Manual Setup

```bash
# 1. Clone repository
git clone https://github.com/logeshkannan19/Hospital-Readmission-Predictor.git
cd Hospital-Readmission-Predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
python download_data.py

# 5. Run notebooks
jupyter notebook

# 6. Run dashboard
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Class Imbalance** | SMOTE (imbalanced-learn) |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Testing** | pytest |
| **Deployment** | Streamlit Cloud / Docker |

---

## 📊 Key Findings

### Top Risk Factors for 30-Day Readmission

| Rank | Risk Factor | Impact |
|------|-------------|--------|
| 1 | Prior Inpatient Visits | 35% |
| 2 | Time in Hospital | 28% |
| 3 | Number of Medications | 20% |
| 4 | Number of Diagnoses | 15% |
| 5 | Emergency Visits | 10% |

### Key Insights from EDA

1. **Age Risk**: Patients aged 70+ have the highest 30-day readmission rate (~15%)
2. **Hospital Stay**: Longer stays correlate with higher readmission risk
   - 1-3 days: ~9% readmission rate
   - 8-14 days: ~16% readmission rate
3. **Prior Visits**: Patients with 3+ prior inpatient visits have ~22% readmission rate
4. **Medication Count**: Patients on 40+ medications have ~18% readmission rate

---

## 📈 Model Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| **ROC-AUC** | >0.70 | ✅ ~0.72 |
| **Recall** | >0.65 | ✅ ~0.68 |
| **Precision** | >0.40 | ✅ ~0.42 |
| **F1 Score** | >0.50 | ✅ ~0.52 |

### Model Comparison

| Model | ROC-AUC | Recall | Precision | F1 |
|-------|---------|--------|-----------|-----|
| **XGBoost** | 0.72 | 0.68 | 0.42 | 0.52 |
| LightGBM | 0.71 | 0.66 | 0.41 | 0.50 |
| Random Forest | 0.69 | 0.62 | 0.38 | 0.47 |
| Logistic Regression | 0.65 | 0.58 | 0.35 | 0.43 |

---

## 🧪 Testing

```bash
# Run tests
make test

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## ☁️ Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file: `app.py`
5. Click **Deploy!**

### Option 2: Docker

```bash
# Build image
docker build -t hospital-readmission-predictor -f docker/Dockerfile .

# Run container
docker run -p 8501:8501 hospital-readmission-predictor

# Or use docker-compose
docker-compose -f docker/docker-compose.yml up
```

### Option 3: Hugging Face Spaces

1. Push code to GitHub
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
3. Create new Space → Select **Streamlit**
4. Link your GitHub repository
5. Deploy!

---

## 📱 Dashboard Features

### Patient Input Form
- Time in hospital (days)
- Number of lab procedures
- Number of medications
- Prior healthcare utilization (inpatient, outpatient, emergency)
- Number of diagnoses
- Primary diagnosis indicator

### Prediction Display
- Risk probability percentage
- Risk level indicator (High/Medium/Low)
- Color-coded risk assessment

### Recommendations
- Follow-up scheduling guidance
- Case management recommendations
- Medication reconciliation suggestions

---

## 📂 Data Dictionary

| Feature | Description |
|---------|-------------|
| `time_in_hospital` | Number of days spent in the hospital |
| `num_lab_procedures` | Number of lab procedures performed |
| `num_medications` | Number of medications prescribed |
| `num_procedures` | Number of procedures performed |
| `number_inpatient` | Prior inpatient visits in the past year |
| `number_outpatient` | Prior outpatient visits in the past year |
| `number_emergency` | Prior emergency visits in the past year |
| `num_diagnoses` | Number of diagnoses |
| `precode` | Diabetes as primary diagnosis (0/1) |
| `readmitted` | Target: <30, >30, or NO |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- Based on research: [Impact of HbA1c Measurement on Hospital Readmission Rates](https://www.hindawi.com/journals/bmri/2014/781670/)

---

<p align="center">
  Made with ❤️ for healthcare innovation
</p>

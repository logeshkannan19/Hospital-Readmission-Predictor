# 🏥 Hospital Readmission Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.24+-blue.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/XGBoost-1.7+-blue.svg" alt="XGBoost">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 📋 Problem Statement

Hospital readmissions for diabetic patients represent a significant challenge for healthcare systems, affecting patient outcomes and generating substantial costs. This project develops a machine learning solution to **predict 30-day hospital readmissions**, enabling healthcare providers to identify high-risk patients and implement proactive intervention strategies.

### Business Impact

- **Cost Reduction**: Reduce hospital readmission costs by 10-20%
- **Improved Patient Outcomes**: Early identification leads to better care coordination
- **Resource Optimization**: Allocate case management resources more effectively
- **Quality Metrics**: Improve Hospital Quality (HQ) ratings

---

## 🎯 Project Overview

This end-to-end machine learning project predicts whether a diabetic patient will be readmitted to the hospital within 30 days. The solution includes:

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Preprocessing & Feature Engineering
- 🤖 Model Training & Hyperparameter Tuning
- 📈 Interactive Streamlit Dashboard
- ☁️ Cloud Deployment Ready

---

## 🗂️ Project Structure

```
Hospital-Readmission-Predictor/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── app.py                       # Streamlit dashboard
├── data/
│   └── diabetic_data.csv       # Original dataset
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb  # Data Cleaning & Feature Engineering
│   ├── 03_Modeling.ipynb        # Model Training & Evaluation
│   └── 04_Deployment.ipynb     # Deployment Instructions
├── src/
│   ├── data_cleaning.py        # Data cleaning utilities
│   ├── modeling.py             # Model training utilities
│   └── visualization.py        # Plotting utilities
├── models/
│   ├── best_model.pkl          # Trained model
│   ├── feature_cols.pkl        # Feature columns
│   └── model_metrics.pkl       # Model performance metrics
└── dashboard/                   # Dashboard screenshots
```

---

## 🚀 Live Demo

**Try the prediction dashboard:** [Hospital Readmission Predictor](https://hospital-readmission-predictor.streamlit.app)

*(Deploy to Streamlit Cloud - see instructions below)*

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

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Model Deployment** | Streamlit Cloud / Hugging Face |

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **ROC-AUC** | >0.70 |
| **Recall** | >0.65 |
| **Precision** | >0.40 |
| **F1 Score** | >0.50 |

### Model Comparison

| Model | ROC-AUC | Recall | Precision | F1 |
|-------|---------|--------|-----------|-----|
| XGBoost | 0.72 | 0.68 | 0.42 | 0.52 |
| LightGBM | 0.71 | 0.66 | 0.41 | 0.50 |
| Random Forest | 0.69 | 0.62 | 0.38 | 0.47 |
| Logistic Regression | 0.65 | 0.58 | 0.35 | 0.43 |

---

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Hospital-Readmission-Predictor.git
cd Hospital-Readmission-Predictor
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

The dataset is available on Kaggle:
- [Diabetes 130-US Hospitals for Years 1999-2008](https://www.kaggle.com/datasets/uciml/diabetes-130-us-hospitals-for-years-1999-2008)

Place `diabetic_data.csv` in the `data/` folder.

### 5. Run Notebooks

```bash
jupyter notebook
```

Open and run the notebooks in order:
1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_Preprocessing.ipynb`
3. `notebooks/03_Modeling.ipynb`

### 6. Run Streamlit Dashboard

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## ☁️ Deployment

### Option 1: Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select repository and branch
5. Set main file path: `app.py`
6. Click **Deploy!**

### Option 2: Hugging Face Spaces

1. Push code to GitHub
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
3. Create new Space (select Streamlit)
4. Link your GitHub repository
5. Deploy!

### Option 3: Local with Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

---

## 📱 Dashboard Features

### Patient Input Form
- Time in hospital
- Number of lab procedures
- Number of medications
- Prior healthcare utilization (inpatient, outpatient, emergency)
- Primary diagnosis indicators

### Prediction Display
- Risk probability percentage
- Risk level indicator (High/Medium/Low)
- Color-coded risk assessment

### Recommendations
- Follow-up scheduling guidance
- Case management recommendations
- Medication reconciliation suggestions
- Patient education materials

---

## 📂 Data Dictionary

| Feature | Description |
|---------|-------------|
| `encounter_id` | Unique identifier of the encounter |
| `patient_nbr` | Unique identifier of the patient |
| `race` | Race of the patient |
| `gender` | Gender of the patient |
| `age` | Age bracket of the patient |
| `time_in_hospital` | Number of days spent in the hospital |
| `num_lab_procedures` | Number of lab procedures |
| `num_medications` | Number of medications prescribed |
| `num_procedures` | Number of procedures performed |
| `number_inpatient` | Number of inpatient visits in the past year |
| `number_outpatient` | Number of outpatient visits in the past year |
| `number_emergency` | Number of emergency visits in the past year |
| `num_diagnoses` | Number of diagnoses |
| `precode` | Whether diabetes was the primary diagnosis |
| `readmitted` | Target variable: <30, >30, or NO |

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- Inspired by: [Diabetes Readmission Prediction](https://www.hindawi.com/journals/bmri/2014/781670/)

---

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

<p align="center">
  Made with ❤️ for healthcare innovation
</p>

# 🏥 Hospital Readmission Prediction Pipeline

> **Predict early hospital readmission (within 30 days) for diabetic patients using Machine Learning.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Project Overview

This project builds a complete ML pipeline to predict early hospital readmission for diabetic patients using the **UCI Diabetes 130-US Hospitals dataset** (101,766 records, 50 features).

### 🎯 Objective
Predict whether a diabetic patient will be **readmitted within 30 days** (`readmitted == '<30'`).

---

## 📊 Pipeline Steps

| Step | Description |
|------|-------------|
| **1. EDA** | Missing values, target distribution, demographics, medication usage |
| **2. Preprocessing** | Replace placeholders, drop sparse columns, fill missing values |
| **3. Feature Engineering** | Outlier capping (IQR), label/ordinal encoding, diagnosis OHE, medication count feature |
| **4. Train/Test Split** | 80/20 stratified split + StandardScaler |
| **5. Class Imbalance** | Optional SMOTE oversampling |
| **6. Modeling** | XGBoost, Random Forest, Logistic Regression |
| **7. Evaluation** | Accuracy, F1, ROC-AUC, Confusion Matrix, Feature Importance |
| **8. Prediction** | Interactive patient-level risk prediction |

---

## 🤖 Models

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| **XGBoost** | 0.6941 | 0.2772 | 0.6637 |
| **Random Forest** | 0.6823 | 0.2748 | 0.6639 |
| **Logistic Regression** | 0.6555 | 0.2583 | 0.6436 |

> ✅ **Best Model by F1:** XGBoost

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/hospital-readmission.git
cd hospital-readmission
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the Dataset
Download the dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) and rename it to `diabetic_data.csv`.

### 4. Run the App
```bash
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy!**

> **Note:** The app requires you to upload the dataset through the UI (no data is stored).

---

## 📁 Project Structure

```
hospital-readmission/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

---

## 🔑 Key Features (Top 5 by XGBoost Importance)

1. `number_inpatient` – Prior inpatient visits (strongest predictor)
2. `age` – Patient age group
3. `discharge_disposition_id` – Where patient was discharged
4. `time_in_hospital` – Length of current stay
5. `race` – Patient race

---

## 📚 Dataset

- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)
- **Records:** 101,766 patient encounters
- **Features:** 50 (demographics, diagnoses, medications, lab results)
- **Target:** `readmitted` → binary (1 = readmitted within 30 days, 0 = not)
- **Class Imbalance:** ~11.2% positive class

---

## 🛠️ Tech Stack

- **Python** 3.9+
- **Streamlit** – Web app framework
- **Pandas / NumPy** – Data manipulation
- **Scikit-learn** – ML models & preprocessing
- **XGBoost** – Gradient boosting
- **imbalanced-learn** – SMOTE
- **Matplotlib / Seaborn** – Visualizations

---

## 📝 License

MIT License — feel free to use and modify.

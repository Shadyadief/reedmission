# 🏥 Hospital Readmission Prediction Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A full ML pipeline to predict **early hospital readmission (within 30 days)** for diabetic patients using the UCI Diabetes 130-US Hospitals dataset (101,766 records).

---

## 📌 Project Overview

| Item | Details |
|------|---------|
| **Dataset** | Diabetic Patients — 101,766 records, 50 features |
| **Target** | Early readmission within 30 days (`readmitted == '<30'`) |
| **Class Imbalance** | ~11.2% positive (early readmit) vs 88.8% negative |
| **Best Model** | XGBoost (scale_pos_weight) — ROC AUC: 0.664, Recall: 0.526 |

---

## 🗂️ Project Structure

```
hospital_readmission/
│
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Data cleaning & feature engineering
│   ├── train.py            # Model training pipeline
│   └── evaluate.py         # Metrics & evaluation utilities
│
├── data/
│   └── README.md           # Instructions to download dataset
│
├── models/
│   └── .gitkeep            # Trained models saved here
│
└── notebooks/
    └── hospital_pipeline.ipynb   # Full EDA + modeling notebook
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/hospital-readmission.git
cd hospital-readmission
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the dataset
Download `diabetic_data.csv` from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) and place it in the `data/` folder.

### 4. Train the models
```bash
python src/train.py
```

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🧪 Pipeline Steps

1. **EDA** — Missing values matrix, target distribution, demographic overview
2. **Preprocessing** — Replace placeholders (`?`) with NaN, drop irrelevant/sparse columns
3. **Feature Engineering** — IQR outlier capping, label/ordinal encoding, diagnosis grouping + OHE, new feature: `num_medications_taken`
4. **Balancing** — SMOTE oversampling (72,326 → balanced classes)
5. **Models Trained:**
   - XGBoost (scale_pos_weight)
   - XGBoost + SMOTE
   - Random Forest (balanced)
   - Logistic Regression (balanced)
   - Soft Voting Ensemble (XGB + RF + LR)
   - Stacking Ensemble (XGB + RF → LR meta)

---

## 📊 Model Results

| Model | Accuracy | F1 Score | ROC AUC | Recall (class 1) |
|-------|----------|----------|---------|-----------------|
| XGBoost | 0.694 | 0.277 | 0.664 | **0.526** |
| XGBoost+SMOTE | 0.888 | 0.017 | 0.676 | 0.009 |
| Random Forest | 0.682 | 0.275 | 0.664 | 0.539 |
| Logistic Regression | 0.656 | 0.258 | 0.644 | 0.538 |
| Voting Ensemble | 0.873 | 0.138 | 0.662 | 0.091 |
| Stacking Ensemble | 0.886 | 0.052 | 0.668 | 0.028 |

> ⚠️ **Clinical note:** In healthcare, **recall for class 1 (early readmit)** is the most critical metric — missing a high-risk patient is more costly than a false alarm.

---

## 🔑 Top Predictive Features (XGBoost + SMOTE)

1. `number_inpatient` (0.233)
2. `age` (0.096)
3. `discharge_disposition_id` (0.089)
4. `time_in_hospital` (0.075)
5. `race` (0.074)

---

## 🖥️ Streamlit App Features

- 📁 Upload your own CSV data
- 🔮 Real-time single-patient prediction
- 📊 Interactive charts: ROC curve, feature importance, confusion matrix
- ⚙️ Model selector (XGBoost / Random Forest / Logistic Regression)

---

## 📦 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** 🎉

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

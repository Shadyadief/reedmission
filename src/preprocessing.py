"""
preprocessing.py
Data cleaning, feature engineering, and encoding for the
Hospital Readmission Prediction Pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ── Constants ────────────────────────────────────────────────────────────────

COLS_TO_DROP = [
    'encounter_id', 'patient_nbr',
    'examide', 'citoglipton',
    'max_glu_serum', 'A1Cresult',
    'payer_code', 'medical_specialty',
    'weight',
    'acetohexamide', 'tolbutamide',
    'troglitazone', 'tolazamide',
    'glimepiride-pioglitazone',
    'metformin-rosiglitazone',
    'metformin-pioglitazone',
    'glipizide-metformin', 'glyburide-metformin',
]

MED_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'insulin',
]

MED_MAP = {'Down': -1, 'No': 0, 'Steady': 1, 'Up': 2}

RACE_MAP = {
    'Caucasian': 1, 'AfricanAmerican': 2,
    'Other': 3, 'Asian': 4, 'Hispanic': 5,
}

GENDER_MAP = {'Male': 1, 'Female': 2}

AGE_ORDER = [
    '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
    '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)',
]

CAP_COLS = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses',
]

DIAG_COLS = ['diag_1', 'diag_2', 'diag_3']


# ── Helpers ───────────────────────────────────────────────────────────────────

def map_diag(code):
    """Map ICD-9 diagnosis code to broad clinical category."""
    try:
        code = float(code)
    except (ValueError, TypeError):
        return 'Other'
    if code == 250:
        return 'Diabetes'
    elif 390 <= code < 460:
        return 'Circulatory'
    elif 460 <= code < 520:
        return 'Respiratory'
    elif 520 <= code < 580:
        return 'Digestive'
    elif 580 <= code < 630:
        return 'Genitourinary'
    elif 800 <= code < 1000:
        return 'Injury'
    return 'Other'


def cap_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Cap outliers using the IQR method (in-place)."""
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = True):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df          : Raw input DataFrame
    scaler      : Optional pre-fitted StandardScaler (for inference)
    fit_scaler  : Whether to fit a new scaler (True for training)

    Returns
    -------
    X_scaled    : np.ndarray  — scaled feature matrix
    y           : pd.Series   — binary target (1 = early readmit)
    scaler      : fitted StandardScaler
    feature_cols: list of column names
    """
    df = df.copy()

    # 1. Replace placeholders
    df.replace(['?', 'Unknown/Invalid'], np.nan, inplace=True)

    # 2. Drop irrelevant / sparse columns
    cols_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df.drop(columns=cols_drop, inplace=True)

    # 3. Fill remaining missing values
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[col] = df[col].fillna('Other')
    if 'race' in df.columns:
        df['race'] = df['race'].fillna('Other')
    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

    # 4. Outlier capping
    cap_cols_present = [c for c in CAP_COLS if c in df.columns]
    df = cap_outliers_iqr(df, cap_cols_present)

    # 5. Encode categorical features
    if 'race' in df.columns:
        df['race'] = df['race'].map(RACE_MAP).fillna(3)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map(GENDER_MAP).fillna(2)
    if 'age' in df.columns:
        age_map = {v: i for i, v in enumerate(AGE_ORDER)}
        df['age'] = df['age'].map(age_map).fillna(0)

    med_cols_present = [c for c in MED_COLS if c in df.columns]
    for col in med_cols_present:
        df[col] = df[col].map(MED_MAP).fillna(0)

    if 'change' in df.columns:
        df['change'] = df['change'].map({'No': 0, 'Ch': 1}).fillna(0)
    if 'diabetesMed' in df.columns:
        df['diabetesMed'] = df['diabetesMed'].map({'No': 0, 'Yes': 1}).fillna(0)

    # 6. Diagnosis grouping + OHE
    for col in DIAG_COLS:
        if col in df.columns:
            df[col] = df[col].apply(map_diag)
    df = pd.get_dummies(df, columns=[c for c in DIAG_COLS if c in df.columns])

    # 7. New feature: total active medications
    existing_meds = [c for c in med_cols_present if c in df.columns]
    df['num_medications_taken'] = df[existing_meds].clip(lower=0).sum(axis=1)

    # 8. Target encoding
    if 'readmitted' in df.columns:
        df['readmitted'] = df['readmitted'].apply(
            lambda x: 1 if x == '<30' else 0
        )
        y = df.pop('readmitted')
    else:
        y = None

    # 9. Scale
    feature_cols = df.columns.tolist()
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
    else:
        X_scaled = scaler.transform(df)

    return X_scaled, y, scaler, feature_cols

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
    roc_curve, f1_score
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

st.set_page_config(
    page_title="🏥 Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2196F3;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.3rem;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 4px solid #2196F3;
        padding-left: 0.7rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Helper Functions ───────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(df_raw):
    df = df_raw.copy()

    # Replace placeholders
    df = df.replace(['?', 'Unknown/Invalid'], np.nan)

    # Drop irrelevant/sparse columns
    cols_to_drop = [
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
        'glipizide-metformin', 'glyburide-metformin'
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Fill missing
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[col] = df[col].fillna('Other')
    if 'race' in df.columns:
        df['race'] = df['race'].fillna('Other')
    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

    # Outlier capping (IQR)
    num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'number_diagnoses']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # Encode
    if 'race' in df.columns:
        df['race'] = df['race'].map({'Caucasian': 1, 'AfricanAmerican': 2,
                                     'Other': 3, 'Asian': 4, 'Hispanic': 5}).fillna(3)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 2}).fillna(1)

    age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                 '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    if 'age' in df.columns:
        df['age'] = df['age'].map({v: i for i, v in enumerate(age_order)}).fillna(5)

    med_map = {'Down': -1, 'No': 0, 'Steady': 1, 'Up': 2}
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
                'rosiglitazone', 'acarbose', 'miglitol', 'insulin']
    for col in med_cols:
        if col in df.columns:
            df[col] = df[col].map(med_map).fillna(0)

    if 'change' in df.columns:
        df['change'] = df['change'].map({'No': 0, 'Ch': 1}).fillna(0)
    if 'diabetesMed' in df.columns:
        df['diabetesMed'] = df['diabetesMed'].map({'No': 0, 'Yes': 1}).fillna(0)

    # Target
    if 'readmitted' in df.columns:
        df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

    # Diagnosis grouping + OHE
    def map_diag(code):
        try:
            code = float(code)
        except:
            return 'Other'
        if code == 250:      return 'Diabetes'
        elif 390 <= code < 460: return 'Circulatory'
        elif 460 <= code < 520: return 'Respiratory'
        elif 520 <= code < 580: return 'Digestive'
        elif 580 <= code < 630: return 'Genitourinary'
        elif 800 <= code < 1000: return 'Injury'
        else:                   return 'Other'

    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[col] = df[col].apply(map_diag)
    df = pd.get_dummies(df, columns=[c for c in ['diag_1', 'diag_2', 'diag_3'] if c in df.columns])

    # Feature: total meds taken
    existing_meds = [c for c in med_cols if c in df.columns]
    df['num_medications_taken'] = df[existing_meds].clip(lower=0).sum(axis=1)

    return df


def train_models(X_train, y_train, X_test, y_test):
    results = {}

    # XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        colsample_bytree=0.7, learning_rate=0.1, max_depth=6,
        n_estimators=300, scale_pos_weight=scale_pos_weight,
        subsample=0.8, random_state=42,
        eval_metric='logloss', use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    results['XGBoost'] = {
        'model': xgb_model, 'y_pred': y_pred, 'y_prob': y_prob,
        'acc': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc': roc_auc_score(y_test, y_prob),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    results['RandomForest'] = {
        'model': rf_model, 'y_pred': y_pred, 'y_prob': y_prob,
        'acc': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc': roc_auc_score(y_test, y_prob),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

    # Logistic Regression
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000,
                                  random_state=42, solver='lbfgs')
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    y_prob = lr_model.predict_proba(X_test)[:, 1]
    results['LogisticRegression'] = {
        'model': lr_model, 'y_pred': y_pred, 'y_prob': y_prob,
        'acc': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc': roc_auc_score(y_test, y_prob),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

    return results


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital-2.png", width=80)
    st.title("🏥 Control Panel")
    st.markdown("---")

    page = st.radio("📋 Navigation", [
        "🏠 Overview",
        "📊 EDA",
        "🤖 Train & Evaluate",
        "🔮 Predict"
    ])

    st.markdown("---")
    st.markdown("**Dataset:** Diabetic Patients")
    st.markdown("**Target:** Early Readmission (<30 days)")
    st.markdown("**Records:** 101,766")

# ─── Load Data ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏥 Hospital Readmission Prediction Pipeline</div>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📂 Upload diabetic_data.csv",
    type=["csv"],
    help="Upload the diabetic patients dataset (CSV format)"
)

if uploaded_file is None:
    st.info("👆 Please upload **diabetic_data.csv** to get started. "
            "You can download the dataset from the UCI ML Repository "
            "(Diabetes 130-US hospitals dataset).")
    st.stop()

df_raw = pd.read_csv(uploaded_file)
st.success(f"✅ Data loaded: **{df_raw.shape[0]:,}** rows × **{df_raw.shape[1]}** columns")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="section-title">📋 Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{df_raw.shape[0]:,}")
    c2.metric("Features", df_raw.shape[1])
    c3.metric("Early Readmissions",
              f"{(df_raw['readmitted'] == '<30').sum():,}")
    c4.metric("Readmission Rate",
              f"{(df_raw['readmitted'] == '<30').mean()*100:.1f}%")

    st.markdown('<div class="section-title">🔍 Raw Data Sample</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.markdown('<div class="section-title">📊 Column Info</div>', unsafe_allow_html=True)
    info_df = pd.DataFrame({
        'Column': df_raw.columns,
        'Type': df_raw.dtypes.values,
        'Non-Null': df_raw.notnull().sum().values,
        'Null %': (df_raw.isnull().mean() * 100).round(2).values
    })
    st.dataframe(info_df, use_container_width=True)

    st.markdown('<div class="section-title">📈 Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.describe().round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown('<div class="section-title">🔍 Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Target", "👥 Demographics", "💊 Medications",
        "📉 Distributions", "❓ Missing Values"
    ])

    with tab1:
        st.markdown("### Target Variable Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        vc = df_raw['readmitted'].value_counts()
        axes[0].bar(vc.index, vc.values,
                    color=['#2196F3', '#FF9800', '#F44336'])
        axes[0].set_title('Readmission Categories')
        axes[0].set_xlabel('Readmitted')
        axes[0].set_ylabel('Count')
        for i, v in enumerate(vc.values):
            axes[0].text(i, v + 200, f'{v:,}\n({v/len(df_raw)*100:.1f}%)',
                         ha='center', fontsize=9, fontweight='bold')

        target_binary = df_raw['readmitted'].apply(
            lambda x: 'Early (<30 days)' if x == '<30' else 'Not Early')
        vc2 = target_binary.value_counts()
        axes[1].pie(vc2.values, labels=vc2.index, autopct='%1.1f%%',
                    colors=['#F44336', '#2196F3'], startangle=90,
                    explode=[0.05, 0], shadow=True)
        axes[1].set_title('Early vs Non-Early Readmission')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown("### Demographic Overview")
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        race_counts = df_raw['race'].replace('?', np.nan).value_counts()
        axes[0].barh(race_counts.index, race_counts.values,
                     color=sns.color_palette('Set2', len(race_counts)))
        axes[0].set_title('Race Distribution')
        axes[0].set_xlabel('Count')

        gender_counts = df_raw['gender'].replace('Unknown/Invalid', np.nan).value_counts()
        axes[1].pie(gender_counts.values, labels=gender_counts.index,
                    autopct='%1.1f%%', colors=['#E91E63', '#2196F3'], startangle=90)
        axes[1].set_title('Gender Distribution')

        age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                     '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
        age_counts = df_raw['age'].value_counts().reindex(age_order)
        axes[2].bar(age_counts.index, age_counts.values,
                    color=sns.color_palette('Blues_d', len(age_counts)))
        axes[2].set_title('Age Group Distribution')
        axes[2].set_xlabel('Age Group')
        axes[2].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.markdown("### Medication Usage")
        med_cols_raw = ['metformin', 'glipizide', 'glyburide', 'pioglitazone',
                        'glimepiride', 'insulin', 'rosiglitazone', 'repaglinide',
                        'nateglinide', 'acarbose', 'miglitol', 'chlorpropamide']
        med_summary = {col: (df_raw[col] != 'No').sum()
                       for col in med_cols_raw if col in df_raw.columns}
        med_df = pd.Series(med_summary).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        import matplotlib.cm as cm
        colors = cm.RdYlGn(np.linspace(0.2, 0.9, len(med_df)))
        bars = ax.barh(med_df.index, med_df.values, color=colors, edgecolor='black')
        for bar, val in zip(bars, med_df.values):
            ax.text(val + 150, bar.get_y() + bar.get_height() / 2.,
                    f'{val:,} ({val/len(df_raw)*100:.1f}%)', va='center', fontsize=9)
        ax.set_title('Medication Usage (Patients on Any Dose)')
        ax.set_xlabel('Number of Patients')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.markdown("### Numerical Feature Distributions")
        num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_medications',
                    'num_procedures', 'number_diagnoses', 'number_inpatient']
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            if col in df_raw.columns:
                axes[i].hist(df_raw[col], bins=30, color='#3F51B5',
                             edgecolor='white', alpha=0.85)
                axes[i].axvline(df_raw[col].mean(), color='red',
                                linestyle='--', linewidth=2)
                axes[i].set_title(col.replace('_', ' ').title())
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab5:
        st.markdown("### Missing Values")
        missing = df_raw.isnull().sum()
        missing_pct = (missing / len(df_raw) * 100).round(2)
        missing_df = pd.DataFrame({'Count': missing, 'Percentage (%)': missing_pct})
        missing_df = missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False)
        if missing_df.empty:
            st.success("✅ No missing values found!")
        else:
            st.dataframe(missing_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: TRAIN & EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train & Evaluate":
    st.markdown('<div class="section-title">🤖 Model Training & Evaluation</div>',
                unsafe_allow_html=True)

    st.info("⚙️ Data will be preprocessed automatically before training.")

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    with col2:
        use_smote = st.checkbox("Apply SMOTE (Balance Classes)", value=False,
                                help="SMOTE oversamples the minority class")

    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        with st.spinner("⏳ Preprocessing data..."):
            df_clean = load_and_preprocess(df_raw)

        st.success(f"✅ Preprocessed: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")

        X = df_clean.drop('readmitted', axis=1)
        y = df_clean['readmitted']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X_train_sc, y_train = smote.fit_resample(X_train_sc, y_train)
                st.info(f"🔄 SMOTE applied: {pd.Series(y_train).value_counts().to_dict()}")
            except ImportError:
                st.warning("⚠️ imbalanced-learn not installed. Skipping SMOTE.")

        with st.spinner("🏋️ Training models..."):
            results = train_models(X_train_sc, y_train, X_test_sc, y_test)

        st.session_state['results'] = results
        st.session_state['X_test'] = X_test_sc
        st.session_state['y_test'] = y_test.values
        st.session_state['scaler'] = scaler
        st.session_state['feature_cols'] = list(X.columns)

        # Metrics Table
        st.markdown('<div class="section-title">📊 Model Comparison</div>',
                    unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['acc'] for m in results],
            'F1 Score': [results[m]['f1'] for m in results],
            'ROC AUC': [results[m]['roc'] for m in results],
            'Recall (class 1)': [results[m]['report']['1']['recall'] for m in results],
            'Precision (class 1)': [results[m]['report']['1']['precision'] for m in results],
        }).set_index('Model').round(4)
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='#C8E6C9'),
                     use_container_width=True)

        # ROC Curves
        st.markdown('<div class="section-title">📈 ROC Curves</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        colors_roc = ['#2196F3', '#4CAF50', '#FF9800']
        for (name, res), color in zip(results.items(), colors_roc):
            fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
            ax.plot(fpr, tpr, color=color, linewidth=2.5,
                    label=f"{name} (AUC = {res['roc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (AUC = 0.500)')
        ax.set_title('ROC Curves – All Models', fontsize=14, fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        # Confusion Matrices
        st.markdown('<div class="section-title">🔲 Confusion Matrices</div>',
                    unsafe_allow_html=True)
        fig, axes = plt.subplots(1, len(results), figsize=(16, 5))
        if len(results) == 1:
            axes = [axes]
        for ax, (name, res) in zip(axes, results.items()):
            cm = confusion_matrix(y_test, res['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Readmit', 'Early Readmit'],
                        yticklabels=['No Readmit', 'Early Readmit'],
                        annot_kws={'size': 13, 'weight': 'bold'})
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Feature Importance
        st.markdown('<div class="section-title">⭐ Feature Importance (XGBoost)</div>',
                    unsafe_allow_html=True)
        best_xgb = results['XGBoost']['model']
        feat_imp = pd.Series(best_xgb.feature_importances_, index=X.columns)
        top20 = feat_imp.sort_values(ascending=True).tail(20)
        fig, ax = plt.subplots(figsize=(10, 7))
        import matplotlib.cm as cm
        colors_fi = cm.RdYlGn(np.linspace(0.2, 0.9, len(top20)))
        ax.barh(top20.index, top20.values, color=colors_fi, edgecolor='black')
        ax.set_title('Top 20 Feature Importances – XGBoost')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    elif 'results' in st.session_state:
        st.info("✅ Models already trained! Results are stored in session. "
                "Retrain anytime by clicking the button above.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown('<div class="section-title">🔮 Patient Readmission Prediction</div>',
                unsafe_allow_html=True)

    if 'results' not in st.session_state:
        st.warning("⚠️ Please train models first (go to **Train & Evaluate** page).")
        st.stop()

    st.markdown("Fill in patient details to predict early readmission risk.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.selectbox("Age Group", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                                          '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
                            index=6)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        race = st.selectbox("Race", ['Caucasian', 'AfricanAmerican', 'Other', 'Asian', 'Hispanic'])
        time_in_hospital = st.slider("Days in Hospital", 1, 14, 4)

    with col2:
        num_lab_procedures = st.slider("# Lab Procedures", 1, 120, 43)
        num_procedures = st.slider("# Procedures", 0, 6, 1)
        num_medications = st.slider("# Medications", 1, 81, 16)
        number_diagnoses = st.slider("# Diagnoses", 1, 16, 7)

    with col3:
        number_inpatient = st.slider("# Inpatient Visits", 0, 20, 0)
        number_emergency = st.slider("# Emergency Visits", 0, 20, 0)
        number_outpatient = st.slider("# Outpatient Visits", 0, 40, 0)
        insulin = st.selectbox("Insulin", ['No', 'Steady', 'Up', 'Down'])

    model_choice = st.selectbox("🤖 Choose Model",
                                 list(st.session_state['results'].keys()))

    if st.button("🔮 Predict Readmission Risk", type="primary", use_container_width=True):
        age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                     '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
        age_enc = age_order.index(age)
        gender_enc = 1 if gender == 'Male' else 2
        race_enc = {'Caucasian': 1, 'AfricanAmerican': 2, 'Other': 3,
                    'Asian': 4, 'Hispanic': 5}[race]
        med_map = {'Down': -1, 'No': 0, 'Steady': 1, 'Up': 2}
        insulin_enc = med_map[insulin]

        # Build a row matching training features
        feature_cols = st.session_state['feature_cols']
        row = pd.Series(0.0, index=feature_cols)

        # Fill known fields
        mapping = {
            'age': age_enc, 'gender': gender_enc, 'race': race_enc,
            'time_in_hospital': time_in_hospital,
            'num_lab_procedures': num_lab_procedures,
            'num_procedures': num_procedures,
            'num_medications': num_medications,
            'number_diagnoses': number_diagnoses,
            'number_inpatient': number_inpatient,
            'number_emergency': number_emergency,
            'number_outpatient': number_outpatient,
            'insulin': insulin_enc,
        }
        for k, v in mapping.items():
            if k in row.index:
                row[k] = v

        X_input = row.values.reshape(1, -1)
        scaler = st.session_state['scaler']
        model = st.session_state['results'][model_choice]['model']

        X_scaled = scaler.transform(X_input)
        prob = model.predict_proba(X_scaled)[0][1]
        pred = model.predict(X_scaled)[0]

        st.markdown("---")
        risk_color = "#F44336" if prob > 0.5 else "#4CAF50"
        risk_label = "⚠️ HIGH RISK" if prob > 0.5 else "✅ LOW RISK"

        st.markdown(f"""
        <div style="background: {risk_color}22; border: 2px solid {risk_color};
                    border-radius: 12px; padding: 1.5rem; text-align: center;">
            <h2 style="color: {risk_color}; margin: 0;">{risk_label}</h2>
            <h3 style="color: #333; margin: 0.5rem 0;">
                Readmission Probability: <b>{prob*100:.1f}%</b>
            </h3>
            <p style="color: #555; margin: 0;">Model: {model_choice}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Risk Gauge")
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh(['Risk'], [prob], color=risk_color, height=0.4)
        ax.barh(['Risk'], [1 - prob], left=[prob], color='#E0E0E0', height=0.4)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
        ax.set_xlim(0, 1)
        ax.text(prob / 2, 0, f'{prob*100:.1f}%', ha='center', va='center',
                color='white', fontweight='bold', fontsize=13)
        ax.axis('off')
        ax.set_title(f'Readmission Risk Score ({model_choice})', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

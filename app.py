"""
app.py — Hospital Readmission Prediction  |  Streamlit Web App
Run: streamlit run app.py
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from preprocessing import preprocess

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #2196F3;
    }
    .metric-label { font-size: 13px; color: #666; margin: 0; }
    .metric-value { font-size: 26px; font-weight: 600; color: #1a1a1a; margin: 0; }
    .risk-high   { background: #fff0f0; border-left-color: #F44336; }
    .risk-low    { background: #f0fff4; border-left-color: #4CAF50; }
    .section-header {
        font-size: 18px; font-weight: 600; color: #1a1a1a;
        margin: 1.5rem 0 0.5rem; padding-bottom: 4px;
        border-bottom: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=60)
st.sidebar.title("🏥 Readmission Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 EDA & Insights", "🔮 Predict", "📈 Model Comparison"],
)

MODEL_FILES = {
    "XGBoost": "models/xgboost.pkl",
    "XGBoost + SMOTE": "models/xgboost_smote.pkl",
    "Random Forest": "models/random_forest.pkl",
    "Logistic Regression": "models/logistic_regression.pkl",
    "Voting Ensemble": "models/voting_ensemble.pkl",
    "Stacking Ensemble": "models/stacking_ensemble.pkl",
}

HARDCODED_METRICS = pd.DataFrame({
    'Model': ['XGBoost', 'XGBoost+SMOTE', 'RandomForest',
              'LogisticRegression', 'VotingEnsemble', 'StackingEnsemble'],
    'Accuracy':           [0.6941, 0.8884, 0.6823, 0.6555, 0.8733, 0.8862],
    'F1 Score':           [0.2772, 0.0173, 0.2748, 0.2583, 0.1384, 0.0524],
    'ROC AUC':            [0.6637, 0.6757, 0.6639, 0.6436, 0.6619, 0.6676],
    'Recall (class 1)':   [0.5258, 0.0088, 0.5394, 0.5376, 0.0911, 0.0282],
    'Precision (class 1)':[0.1882, 0.4878, 0.1843, 0.1700, 0.2871, 0.3699],
}).set_index('Model')

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_scaler():
    if os.path.exists('models/scaler.pkl'):
        return joblib.load('models/scaler.pkl')
    return None


@st.cache_resource
def load_feature_cols():
    if os.path.exists('models/feature_cols.pkl'):
        return joblib.load('models/feature_cols.pkl')
    return None


def models_available():
    return any(os.path.exists(p) for p in MODEL_FILES.values())


# ═══════════════════════════════════════════════════════════════
#  PAGE: Overview
# ═══════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("🏥 Hospital Readmission Prediction Pipeline")
    st.markdown(
        "Predict **early hospital readmission within 30 days** for diabetic patients "
        "using an ensemble ML pipeline trained on 101,766 records."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <p class="metric-label">Dataset Records</p>
            <p class="metric-value">101,766</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <p class="metric-label">Features</p>
            <p class="metric-value">50 → 51</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <p class="metric-label">Best ROC AUC</p>
            <p class="metric-value">0.676</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <p class="metric-label">Positive Class</p>
            <p class="metric-value">11.2%</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<p class="section-header">Pipeline Steps</p>', unsafe_allow_html=True)
        steps = [
            ("1. Data Loading", "101,766 records × 50 columns from UCI repository"),
            ("2. EDA", "Missing values matrix, target distribution, demographics"),
            ("3. Preprocessing", "Replace placeholders, drop sparse columns, fill NaNs"),
            ("4. Feature Engineering", "IQR capping, encoding, diagnosis grouping + OHE"),
            ("5. SMOTE Balancing", "9,086 → 72,326 minority class samples"),
            ("6. Model Training", "6 models: XGBoost, RF, LR, Voting, Stacking"),
            ("7. Evaluation", "Accuracy, F1, ROC AUC, Confusion Matrices, PR Curves"),
        ]
        for title, desc in steps:
            st.markdown(f"**{title}** — {desc}")

    with col_right:
        st.markdown('<p class="section-header">Top Features (XGBoost + SMOTE)</p>', unsafe_allow_html=True)
        feat_data = {
            'Feature': ['number_inpatient', 'age', 'discharge_disposition_id',
                        'time_in_hospital', 'race', 'num_procedures', 'gender'],
            'Importance': [0.2332, 0.0962, 0.0892, 0.0748, 0.0736, 0.0633, 0.0615],
        }
        feat_df = pd.DataFrame(feat_data)
        fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Teal',
                     title='Top 7 Feature Importances')
        fig.update_layout(height=340, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.info(
        "⚠️ **Clinical note:** In healthcare, **recall for the early-readmit class** "
        "is the most critical metric — missing a high-risk patient costs more than a false alarm. "
        "XGBoost (scale_pos_weight) achieves the best recall: **0.526**."
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE: EDA & Insights
# ═══════════════════════════════════════════════════════════════

elif page == "📊 EDA & Insights":
    st.title("📊 EDA & Insights")

    tab1, tab2, tab3 = st.tabs(["Target Distribution", "Demographics", "Medication Usage"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            raw_data = {'NO': 54864, '>30': 35545, '<30': 11357}
            fig = px.bar(
                x=list(raw_data.keys()), y=list(raw_data.values()),
                color=list(raw_data.keys()),
                color_discrete_map={'NO': '#2196F3', '>30': '#FF9800', '<30': '#F44336'},
                title='Readmission Categories (Raw)',
                labels={'x': 'Readmitted', 'y': 'Count'},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.pie(
                values=[90409, 11357],
                names=['Not Early', 'Early (<30 days)'],
                color_discrete_sequence=['#F44336', '#2196F3'],
                title='Early vs Non-Early Readmission',
                hole=0.4,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Class imbalance:** Not Early: 90,409 | Early (<30 days): 11,357")
        st.markdown("""
        | Column | Missing Count | Missing % |
        |--------|--------------|-----------|
        | max_glu_serum | 96,420 | 94.75% |
        | A1Cresult | 84,748 | 83.28% |
        """)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            race_data = {
                'Caucasian': 76099, 'AfricanAmerican': 19210,
                'Hispanic': 2037, 'Other': 1506, 'Asian': 641,
            }
            fig = px.bar(
                x=list(race_data.values()), y=list(race_data.keys()),
                orientation='h', title='Race Distribution',
                color=list(race_data.values()),
                color_continuous_scale='Viridis',
                labels={'x': 'Count', 'y': 'Race'},
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.pie(
                values=[53.76, 46.24],
                names=['Female', 'Male'],
                color_discrete_sequence=['#E91E63', '#2196F3'],
                title='Gender Distribution',
            )
            st.plotly_chart(fig2, use_container_width=True)

        age_data = {
            '[0-10)': 22, '[10-20)': 167, '[20-30)': 893, '[30-40)': 2592,
            '[40-50)': 7694, '[50-60)': 14964, '[60-70)': 23698,
            '[70-80)': 26068, '[80-90)': 19521, '[90-100)': 6147,
        }
        fig3 = px.bar(
            x=list(age_data.keys()), y=list(age_data.values()),
            title='Age Group Distribution',
            color=list(age_data.values()),
            color_continuous_scale='Blues',
            labels={'x': 'Age Group', 'y': 'Count'},
        )
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        med_data = {
            'insulin': 54383, 'metformin': 19988, 'glipizide': 12686,
            'glyburide': 10650, 'pioglitazone': 7328, 'rosiglitazone': 6365,
            'glimepiride': 5191, 'repaglinide': 1539, 'nateglinide': 703,
            'acarbose': 308, 'chlorpropamide': 86, 'miglitol': 38,
        }
        labels = [f"{k}\n({v/101766*100:.1f}%)" for k, v in med_data.items()]
        fig = px.bar(
            x=list(med_data.values()), y=labels,
            orientation='h', title='Medication Usage (Patients on Any Dose)',
            color=list(med_data.values()),
            color_continuous_scale='Greens',
            labels={'x': 'Number of Patients', 'y': 'Medication'},
        )
        fig.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: Predict
# ═══════════════════════════════════════════════════════════════

elif page == "🔮 Predict":
    st.title("🔮 Patient Readmission Risk Predictor")

    tabs = st.tabs(["🧑‍⚕️ Single Patient", "📂 Batch CSV Upload"])

    # ── Single patient ────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("Fill in the patient details below:")

        col1, col2, col3 = st.columns(3)
        with col1:
            age_val = st.selectbox("Age Group", [
                '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)',
            ], index=6)
            gender_val = st.selectbox("Gender", ['Male', 'Female'])
            race_val = st.selectbox("Race", [
                'Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other',
            ])

        with col2:
            time_in_hosp = st.slider("Time in Hospital (days)", 1, 14, 4)
            num_lab = st.slider("# Lab Procedures", 1, 120, 44)
            num_meds = st.slider("# Medications", 1, 80, 15)
            num_diag = st.slider("# Diagnoses", 1, 16, 7)

        with col3:
            num_inpat = st.slider("# Inpatient Visits (prior year)", 0, 20, 0)
            num_emerg = st.slider("# Emergency Visits (prior year)", 0, 20, 0)
            num_outpat = st.slider("# Outpatient Visits (prior year)", 0, 40, 0)
            insulin_val = st.selectbox("Insulin", ['No', 'Steady', 'Up', 'Down'])
            metformin_val = st.selectbox("Metformin", ['No', 'Steady', 'Up', 'Down'])

        model_choice = st.selectbox("Select Model", list(MODEL_FILES.keys()))

        if st.button("🔮 Predict Risk", type="primary"):
            model = load_model(MODEL_FILES[model_choice])
            scaler = load_scaler()
            feature_cols = load_feature_cols()

            if model is None or scaler is None:
                st.warning(
                    "⚠️ No trained models found. "
                    "Run `python src/train.py --data data/diabetic_data.csv` first, "
                    "or the app will show a demo prediction."
                )
                # Demo result
                prob = np.random.uniform(0.1, 0.6)
                pred = 1 if prob > 0.35 else 0
            else:
                AGE_MAP = {v: i for i, v in enumerate([
                    '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                    '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)',
                ])}
                MED_MAP = {'Down': -1, 'No': 0, 'Steady': 1, 'Up': 2}
                RACE_MAP = {'Caucasian': 1, 'AfricanAmerican': 2, 'Other': 3, 'Asian': 4, 'Hispanic': 5}
                GENDER_MAP = {'Male': 1, 'Female': 2}

                patient = {
                    'age': AGE_MAP.get(age_val, 6),
                    'gender': GENDER_MAP.get(gender_val, 1),
                    'race': RACE_MAP.get(race_val, 1),
                    'time_in_hospital': time_in_hosp,
                    'num_lab_procedures': num_lab,
                    'num_medications': num_meds,
                    'number_diagnoses': num_diag,
                    'number_inpatient': num_inpat,
                    'number_emergency': num_emerg,
                    'number_outpatient': num_outpat,
                    'insulin': MED_MAP.get(insulin_val, 0),
                    'metformin': MED_MAP.get(metformin_val, 0),
                }
                df_pat = pd.DataFrame([patient])
                for col in feature_cols:
                    if col not in df_pat.columns:
                        df_pat[col] = 0
                df_pat = df_pat[feature_cols]
                X = scaler.transform(df_pat)
                pred = model.predict(X)[0]
                prob = model.predict_proba(X)[0][1]

            st.markdown("---")
            risk_class = "risk-high" if pred == 1 else "risk-low"
            risk_label = "⚠️ High Risk — Early Readmission Likely" if pred == 1 else "✅ Low Risk — No Early Readmission Expected"

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(
                    f'<div class="metric-card {risk_class}">'
                    f'<p class="metric-label">Prediction</p>'
                    f'<p class="metric-value">{risk_label}</p></div>',
                    unsafe_allow_html=True,
                )
            with col_r2:
                st.markdown(
                    f'<div class="metric-card {risk_class}">'
                    f'<p class="metric-label">Readmission Probability</p>'
                    f'<p class="metric-value">{prob:.1%}</p></div>',
                    unsafe_allow_html=True,
                )

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode='gauge+number',
                value=round(prob * 100, 1),
                title={'text': 'Readmission Risk (%)'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#F44336' if pred == 1 else '#4CAF50'},
                    'steps': [
                        {'range': [0, 35], 'color': '#e8f5e9'},
                        {'range': [35, 65], 'color': '#fff8e1'},
                        {'range': [65, 100], 'color': '#ffebee'},
                    ],
                    'threshold': {'line': {'color': '#333', 'width': 3}, 'value': 35},
                },
                number={'suffix': '%'},
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Batch upload ──────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("Upload a CSV file matching the original dataset format.")
        uploaded = st.file_uploader("Choose CSV file", type=['csv'])

        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.write(f"Loaded: **{df_up.shape[0]} rows × {df_up.shape[1]} columns**")
            st.dataframe(df_up.head())

            model_choice_b = st.selectbox("Model for Batch", list(MODEL_FILES.keys()), key='batch_model')

            if st.button("Run Batch Prediction", type="primary"):
                model_b = load_model(MODEL_FILES[model_choice_b])
                scaler_b = load_scaler()
                feature_cols_b = load_feature_cols()

                if model_b is None or scaler_b is None:
                    st.warning("Train models first (`python src/train.py`).")
                else:
                    has_target = 'readmitted' in df_up.columns
                    X_b, y_b, _, _ = preprocess(df_up, scaler=scaler_b, fit_scaler=False)
                    preds = model_b.predict(X_b)
                    probs = model_b.predict_proba(X_b)[:, 1]

                    df_up['Prediction'] = preds
                    df_up['Risk Probability'] = probs.round(4)
                    st.success(f"✅ Predicted {len(preds)} patients")
                    st.dataframe(df_up[['Prediction', 'Risk Probability']].head(20))

                    csv_out = df_up.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        csv_out, "predictions.csv", "text/csv",
                    )


# ═══════════════════════════════════════════════════════════════
#  PAGE: Model Comparison
# ═══════════════════════════════════════════════════════════════

elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison & Analysis")

    st.markdown("### Metrics Summary")
    st.dataframe(HARDCODED_METRICS.style.highlight_max(axis=0, color='#c8e6c9').format("{:.4f}"))

    st.markdown("---")

    # Bar chart
    metrics_to_plot = ['Accuracy', 'F1 Score', 'ROC AUC', 'Recall (class 1)']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    fig = go.Figure()
    for metric, color in zip(metrics_to_plot, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=HARDCODED_METRICS.index,
            y=HARDCODED_METRICS[metric],
            marker_color=color,
        ))
    fig.update_layout(
        barmode='group', title='Model Performance Comparison',
        yaxis=dict(range=[0, 1.05]),
        xaxis_tickangle=-30, height=450,
        legend=dict(orientation='h', y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Feature Importance (XGBoost + SMOTE — Top 20)")
    feat_importance = {
        'number_inpatient': 0.2332, 'age': 0.0962, 'discharge_disposition_id': 0.0892,
        'time_in_hospital': 0.0748, 'race': 0.0736, 'num_procedures': 0.0633,
        'gender': 0.0615, 'number_diagnoses': 0.0420, 'admission_type_id': 0.0366,
        'admission_source_id': 0.0242, 'insulin': 0.0173, 'metformin': 0.0172,
        'diag_1_Circulatory': 0.0153, 'num_medications_taken': 0.0134,
        'diag_3_Other': 0.0129, 'diag_2_Other': 0.0119, 'diabetesMed': 0.0064,
        'change': 0.0063, 'diag_1_Other': 0.0057, 'diag_3_Circulatory': 0.0056,
    }
    feat_df = pd.DataFrame.from_dict(feat_importance, orient='index', columns=['Importance'])
    feat_df = feat_df.sort_values('Importance')
    fig2 = px.bar(
        feat_df, x='Importance', y=feat_df.index, orientation='h',
        color='Importance', color_continuous_scale='Teal',
        title='Top 20 Feature Importances — XGBoost + SMOTE',
    )
    fig2.update_layout(height=580, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### Key Takeaways")
    st.markdown("""
    - 🏆 **Best Recall (class 1):** Random Forest Balanced (0.539) and XGBoost (0.526) — best for catching high-risk patients
    - 🎯 **Best ROC AUC:** XGBoost + SMOTE (0.676)
    - ⚖️ **Best F1 Score:** XGBoost (0.277)
    - ⚠️ SMOTE and ensemble methods boost accuracy but can collapse minority-class recall — always inspect **class-specific metrics**
    - 🔑 Top predictor: **number_inpatient** (prior inpatient visits) — the strongest signal for future readmission
    """)

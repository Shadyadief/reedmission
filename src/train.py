"""
train.py
Train all models in the Hospital Readmission Prediction Pipeline
and save them to the models/ directory.

Usage:
    python src/train.py --data data/diabetic_data.csv
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

from preprocessing import preprocess


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred)
    roc = roc_auc_score(y_te, y_prob)
    rep = classification_report(y_te, y_pred, output_dict=True)

    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC AUC  : {roc:.4f}")
    print(classification_report(y_te, y_pred))

    return {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'acc': acc, 'f1': f1, 'roc': roc, 'report': rep,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_path: str, models_dir: str = 'models'):
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading data from {data_path} …")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")

    # Preprocess
    X, y, scaler, feature_cols = preprocess(df, fit_scaler=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(models_dir, 'feature_cols.pkl'))
    print(f"  Features: {len(feature_cols)}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # SMOTE balanced set
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE — {dict(zip(*np.unique(y_res, return_counts=True)))}")

    results = {}
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # 1. XGBoost (scale_pos_weight)
    xgb_model = xgb.XGBClassifier(
        colsample_bytree=0.7, learning_rate=0.1, max_depth=6,
        n_estimators=300, scale_pos_weight=scale_pos_weight,
        subsample=0.8, random_state=42,
        eval_metric='logloss', use_label_encoder=False,
    )
    results['XGBoost'] = evaluate_model(
        'XGBoost (scale_pos_weight)', xgb_model,
        X_train, y_train, X_test, y_test,
    )
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost.pkl'))

    # 2. XGBoost + SMOTE
    xgb_smote = xgb.XGBClassifier(
        colsample_bytree=0.7, learning_rate=0.1, max_depth=6,
        n_estimators=300, subsample=0.8,
        random_state=42, eval_metric='logloss', use_label_encoder=False,
    )
    results['XGBoost+SMOTE'] = evaluate_model(
        'XGBoost + SMOTE', xgb_smote,
        X_res, y_res, X_test, y_test,
    )
    joblib.dump(xgb_smote, os.path.join(models_dir, 'xgboost_smote.pkl'))

    # 3. Random Forest (balanced)
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    results['RandomForest'] = evaluate_model(
        'Random Forest (Balanced)', rf_model,
        X_train, y_train, X_test, y_test,
    )
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))

    # 4. Logistic Regression (balanced)
    lr_model = LogisticRegression(
        class_weight='balanced', max_iter=1000,
        random_state=42, solver='lbfgs',
    )
    results['LogisticRegression'] = evaluate_model(
        'Logistic Regression (Balanced)', lr_model,
        X_train, y_train, X_test, y_test,
    )
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression.pkl'))

    # 5. Soft Voting Ensemble
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', xgb.XGBClassifier(
                colsample_bytree=0.7, learning_rate=0.1, max_depth=6,
                n_estimators=300, subsample=0.8,
                random_state=42, eval_metric='logloss', use_label_encoder=False,
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight='balanced', random_state=42, n_jobs=-1,
            )),
            ('lr', LogisticRegression(
                class_weight='balanced', max_iter=1000,
                random_state=42, solver='lbfgs',
            )),
        ],
        voting='soft',
    )
    results['VotingEnsemble'] = evaluate_model(
        'Soft Voting Ensemble (XGB + RF + LR)', voting_clf,
        X_res, y_res, X_test, y_test,
    )
    joblib.dump(voting_clf, os.path.join(models_dir, 'voting_ensemble.pkl'))

    # 6. Stacking Ensemble
    stacking_clf = StackingClassifier(
        estimators=[
            ('xgb', xgb.XGBClassifier(
                colsample_bytree=0.7, learning_rate=0.1, max_depth=6,
                n_estimators=200, subsample=0.8,
                random_state=42, eval_metric='logloss', use_label_encoder=False,
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100, max_depth=8,
                class_weight='balanced', random_state=42, n_jobs=-1,
            )),
        ],
        final_estimator=LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42,
        ),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
    )
    results['StackingEnsemble'] = evaluate_model(
        'Stacking Ensemble (XGB + RF → LR meta)', stacking_clf,
        X_res, y_res, X_test, y_test,
    )
    joblib.dump(stacking_clf, os.path.join(models_dir, 'stacking_ensemble.pkl'))

    # Summary table
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['acc'] for m in results],
        'F1 Score': [results[m]['f1'] for m in results],
        'ROC AUC':  [results[m]['roc'] for m in results],
        'Recall (class 1)': [results[m]['report']['1']['recall'] for m in results],
        'Precision (class 1)': [results[m]['report']['1']['precision'] for m in results],
    }).set_index('Model')

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print(metrics_df.round(4).to_string())

    metrics_df.to_csv(os.path.join(models_dir, 'metrics_summary.csv'))
    print(f"\n✅ All models saved to '{models_dir}/'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hospital readmission models')
    parser.add_argument('--data', default='data/diabetic_data.csv', help='Path to CSV data')
    parser.add_argument('--models', default='models', help='Directory to save models')
    args = parser.parse_args()
    main(args.data, args.models)

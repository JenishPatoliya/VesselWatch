"""
=============================================================================
  VesselWatch V3 — Main Pipeline Runner
=============================================================================
  Run this in Google Colab. It imports from the modular pipeline/ files.

  Before running, upload the pipeline/ folder to your Colab or
  add it to sys.path.
=============================================================================
"""

# ── Setup (run this cell first in Colab) ──────────────────
from google.colab import drive
drive.mount('/content/drive')

import sys
import os
import json
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings('ignore')

# Ensure the project root is in sys.path so 'pipeline' is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pipeline.config import ML_COLS, OUTPUT_PATH, PROCESSED_PATH
from pipeline.data_loader import load_raw_data, clean_data
from pipeline.feature_engineering import compute_features
from pipeline.labels import create_improved_labels
from pipeline.models import (
    prepare_ml_data, train_isolation_forest, train_dbscan,
    train_lstm_autoencoder, build_stack_features,
    train_supervised_stack, find_optimal_threshold
)
from pipeline.evaluation import cross_validate, evaluate_final, plot_results

print('✅ All modules loaded')


# ══════════════════════════════════════════════════════════
# STEP 1 — Load & Clean Data
# ══════════════════════════════════════════════════════════

train_df, test_df, common_vessels = load_raw_data()

print('\nCleaning data...')
train_df = clean_data(train_df)
test_df = clean_data(test_df)
print(f'Train: {len(train_df):,} | Test: {len(test_df):,}')


# ══════════════════════════════════════════════════════════
# STEP 2 — Feature Engineering (32 features)
# ══════════════════════════════════════════════════════════

print('\nComputing train features...')
train_features = compute_features(train_df)
print(f'Train vessels: {len(train_features):,}')

print('\nComputing test features...')
test_features = compute_features(test_df)
test_features = test_features[
    test_features['MMSI'].isin(common_vessels)].copy()
print(f'Test vessels: {len(test_features):,}')


# ══════════════════════════════════════════════════════════
# STEP 3 — Create Pseudo-Labels
# ══════════════════════════════════════════════════════════

train_features['true_anomaly'] = create_improved_labels(train_features)
test_features['true_anomaly'] = create_improved_labels(test_features)

print(f'\nTrain anomalies: {train_features["true_anomaly"].sum():,} / '
      f'{len(train_features):,}')
print(f'Test anomalies:  {test_features["true_anomaly"].sum():,} / '
      f'{len(test_features):,}')


# ══════════════════════════════════════════════════════════
# STEP 4 — Prepare & Scale Features
# ══════════════════════════════════════════════════════════

X_train_scaled, X_test_scaled, scaler = prepare_ml_data(
    train_features, test_features)
y_train = train_features['true_anomaly'].values
y_test = test_features['true_anomaly'].values


# ══════════════════════════════════════════════════════════
# STEP 5 — Train Unsupervised Models
# ══════════════════════════════════════════════════════════

iso = train_isolation_forest(
    X_train_scaled, X_test_scaled, train_features, test_features)
train_dbscan(train_features, test_features)
train_lstm_autoencoder(X_train_scaled, X_test_scaled, test_features)


# ══════════════════════════════════════════════════════════
# STEP 6 — Train Supervised Stacking Model
# ══════════════════════════════════════════════════════════

X_train_stack = build_stack_features(X_train_scaled, train_features)
X_test_stack = build_stack_features(X_test_scaled, test_features)

gbc, rfc, ensemble_proba = train_supervised_stack(
    X_train_stack, y_train, X_test_stack)


# ══════════════════════════════════════════════════════════
# STEP 7 — Find Optimal Threshold
# ══════════════════════════════════════════════════════════

optimal_threshold, precisions, recalls, thresholds = \
    find_optimal_threshold(y_test, ensemble_proba)

final_pred = (ensemble_proba >= optimal_threshold).astype(int)
test_features['final_risk_score'] = ensemble_proba
test_features['final_flag'] = final_pred


# ══════════════════════════════════════════════════════════
# STEP 8 — Cross-Validation
# ══════════════════════════════════════════════════════════

cv_metrics = cross_validate(X_train_stack, y_train, optimal_threshold)


# ══════════════════════════════════════════════════════════
# STEP 9 — Final Evaluation & Visualization
# ══════════════════════════════════════════════════════════

p_val, r_val, f1_val, auc_val, cm = evaluate_final(
    y_test, final_pred, ensemble_proba, optimal_threshold)

plot_results(y_test, ensemble_proba, final_pred, cm, auc_val,
             optimal_threshold, gbc, precisions, recalls)


# ══════════════════════════════════════════════════════════
# STEP 10 — SHAP & Save Results
# ══════════════════════════════════════════════════════════

print('\nCalculating SHAP values...')
feature_names = ML_COLS + ['iso_risk_score', 'rendezvous_flag']
explainer = shap.TreeExplainer(gbc)
shap_values = explainer.shap_values(X_test_stack)

shap_df = pd.DataFrame(shap_values, columns=feature_names)
shap_df['MMSI'] = test_features['MMSI'].values
shap_df['VesselName'] = test_features['VesselName'].values
shap_df['final_risk_score'] = test_features['final_risk_score'].values

top_shap = shap_df.nlargest(50, 'final_risk_score')
top_shap.to_csv(OUTPUT_PATH + 'shap_explanations.csv', index=False)
top_shap.to_parquet(PROCESSED_PATH + 'shap_explanations_v3.parquet',
                    index=False)

# Save final results
if 'rendezvous_risk' not in test_features.columns:
    test_features['rendezvous_risk'] = \
        test_features['rendezvous_flag'].astype(float)

test_features.to_csv(OUTPUT_PATH + 'final_results.csv', index=False)
test_features.to_parquet(PROCESSED_PATH + 'final_results_v3.parquet',
                         index=False)

# Save summary
summary = {
    'version': 'V3',
    'precision': round(float(p_val), 4),
    'recall': round(float(r_val), 4),
    'f1_score': round(float(f1_val), 4),
    'roc_auc': round(float(auc_val), 4),
    'threshold': round(float(optimal_threshold), 4),
    'test_vessels': int(len(test_features)),
    'flagged': int(final_pred.sum()),
    'TP': int(cm[1][1]), 'FP': int(cm[0][1]),
    'TN': int(cm[0][0]), 'FN': int(cm[1][0]),
}
with open(OUTPUT_PATH + 'validation_summary_v3.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('\n' + '=' * 60)
print('  ✅ ALL RESULTS SAVED')
print('=' * 60)
print(f'Precision:   {p_val:.2%}')
print(f'Recall:      {r_val:.2%}')
print(f'F1:          {f1_val:.2%}')
print(f'ROC-AUC:     {auc_val:.4f}')
print(f'Threshold:   {optimal_threshold:.4f}')

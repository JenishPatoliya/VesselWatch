"""
VesselWatch V3 — Evaluation & Visualization
Cross-validation, confusion matrix, ROC/PR curves, feature importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report
)

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'imbalanced-learn', '-q'])
    from imblearn.over_sampling import SMOTE

from .config import ML_COLS, GBC_PARAMS, PROCESSED_PATH


def cross_validate(X_train_stack, y_train, optimal_threshold):
    """5-fold stratified cross-validation on train set."""
    print('\n' + '=' * 60)
    print('  5-FOLD CROSS-VALIDATION')
    print('=' * 60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics = {'precision': [], 'recall': [], 'f1': [], 'auc': []}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_stack, y_train)):
        X_tr, X_val = X_train_stack[tr_idx], X_train_stack[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        smote_cv = SMOTE(random_state=42,
                         k_neighbors=min(5, sum(y_tr == 1) - 1))
        X_tr_bal, y_tr_bal = smote_cv.fit_resample(X_tr, y_tr)

        gbc_cv = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42)
        gbc_cv.fit(X_tr_bal, y_tr_bal)

        val_proba = gbc_cv.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= optimal_threshold).astype(int)

        p = precision_score(y_val, val_pred, zero_division=0)
        r = recall_score(y_val, val_pred, zero_division=0)
        f = f1_score(y_val, val_pred, zero_division=0)
        auc = roc_auc_score(y_val, val_proba) if len(set(y_val)) > 1 else 0

        cv_metrics['precision'].append(p)
        cv_metrics['recall'].append(r)
        cv_metrics['f1'].append(f)
        cv_metrics['auc'].append(auc)

        print(f'Fold {fold + 1}: P={p:.2%} R={r:.2%} F1={f:.2%} AUC={auc:.4f}')

    print(f'\n--- CV Summary ---')
    for k, v in cv_metrics.items():
        print(f'{k.capitalize():>10}: {np.mean(v):.2%} ± {np.std(v):.2%}')

    return cv_metrics


def evaluate_final(y_test, final_pred, ensemble_proba, optimal_threshold):
    """Final out-of-sample evaluation with full metrics."""
    print('\n' + '=' * 60)
    print('  FINAL OUT-OF-SAMPLE VALIDATION')
    print('=' * 60)

    p_val = precision_score(y_test, final_pred, zero_division=0)
    r_val = recall_score(y_test, final_pred, zero_division=0)
    f1_val = f1_score(y_test, final_pred, zero_division=0)
    auc_val = roc_auc_score(y_test, ensemble_proba)
    cm = confusion_matrix(y_test, final_pred)

    print(f'\nPrecision: {p_val:.2%}')
    print(f'Recall:    {r_val:.2%}')
    print(f'F1 Score:  {f1_val:.2%}')
    print(f'ROC-AUC:   {auc_val:.4f}')
    print(f'Threshold: {optimal_threshold:.4f}')
    print(f'\nConfusion Matrix:')
    print(f'TN: {cm[0][0]:,}  FP: {cm[0][1]:,}')
    print(f'FN: {cm[1][0]:,}  TP: {cm[1][1]:,}')

    report = classification_report(y_test, final_pred,
                                   target_names=["Normal", "Anomaly"])
    print(f'\n{report}')

    return p_val, r_val, f1_val, auc_val, cm


def plot_results(y_test, ensemble_proba, final_pred, cm, auc_val,
                 optimal_threshold, gbc, precisions_curve=None,
                 recalls_curve=None):
    """Generate all validation charts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#111827')

    # 1. Confusion Matrix
    ax1 = axes[0, 0]
    ax1.set_facecolor('#111827')
    ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Normal', 'Anomaly'], color='white')
    ax1.set_yticklabels(['Normal', 'Anomaly'], color='white')
    ax1.set_xlabel('Predicted', color='white')
    ax1.set_ylabel('Actual', color='white')
    ax1.set_title('Confusion Matrix (V3)', color='#00d4ff', fontsize=13)
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f'{labels[i][j]}\n{cm[i, j]:,}',
                     ha='center', va='center',
                     color='white', fontsize=12, fontweight='bold')

    # 2. ROC Curve
    ax2 = axes[0, 1]
    ax2.set_facecolor('#111827')
    fpr, tpr, _ = roc_curve(y_test, ensemble_proba)
    ax2.plot(fpr, tpr, color='#00d4ff', linewidth=2,
             label=f'V3 (AUC={auc_val:.4f})')
    ax2.plot([0, 1], [0, 1], color='#4a5568', linestyle='--', label='Random')
    ax2.fill_between(fpr, tpr, alpha=0.1, color='#00d4ff')
    ax2.set_xlabel('False Positive Rate', color='white')
    ax2.set_ylabel('True Positive Rate', color='white')
    ax2.set_title('ROC Curve', color='#00d4ff', fontsize=13)
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#1e2d45', labelcolor='white')
    ax2.grid(True, alpha=0.2, color='#1e2d45')

    # 3. Precision-Recall Curve
    ax3 = axes[1, 0]
    ax3.set_facecolor('#111827')
    if precisions_curve is not None and recalls_curve is not None:
        ax3.plot(recalls_curve, precisions_curve, color='#ff8c00', linewidth=2)
    ax3.set_xlabel('Recall', color='white')
    ax3.set_ylabel('Precision', color='white')
    ax3.set_title('Precision-Recall Curve', color='#00d4ff', fontsize=13)
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, color='#1e2d45')

    # 4. Feature Importance
    ax4 = axes[1, 1]
    ax4.set_facecolor('#111827')
    feature_names = ML_COLS + ['iso_risk_score', 'rendezvous_flag']
    importances = gbc.feature_importances_
    top_n = 15
    top_idx = np.argsort(importances)[-top_n:]
    ax4.barh(range(top_n), importances[top_idx], color='#3b82f6')
    ax4.set_yticks(range(top_n))
    ax4.set_yticklabels([feature_names[i] for i in top_idx],
                         color='white', fontsize=9)
    ax4.set_xlabel('Importance', color='white')
    ax4.set_title('Top 15 Feature Importances', color='#00d4ff', fontsize=13)
    ax4.tick_params(colors='white')

    plt.suptitle('VesselWatch V3 — Validation Results',
                 color='#00d4ff', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PROCESSED_PATH + 'validation_results_v3.png',
                dpi=150, bbox_inches='tight', facecolor='#111827')
    plt.show()
    print('✅ Validation charts saved')

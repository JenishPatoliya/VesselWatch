"""
VesselWatch V3 — ML Models
Isolation Forest, DBSCAN, LSTM Autoencoder, and Supervised Stacking.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    IsolationForest, GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_recall_curve

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'imbalanced-learn', '-q'])
    from imblearn.over_sampling import SMOTE

from .config import (
    ML_COLS, ISO_FOREST_PARAMS, GBC_PARAMS, RFC_PARAMS,
    DBSCAN_EPS, DBSCAN_MIN_SAMPLES, RENDEZVOUS_PORT_DIST_KM
)


def prepare_ml_data(train_features, test_features):
    """Scale features using train statistics only."""
    for col in ML_COLS:
        if col not in train_features.columns:
            train_features[col] = 0
        if col not in test_features.columns:
            test_features[col] = 0

    X_train = train_features[ML_COLS].replace(
        [np.inf, -np.inf], 0).fillna(0)
    X_test = test_features[ML_COLS].replace(
        [np.inf, -np.inf], 0).fillna(0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f'Feature matrix shape: {X_train_scaled.shape}')
    return X_train_scaled, X_test_scaled, scaler


def train_isolation_forest(X_train_scaled, X_test_scaled,
                           train_features, test_features):
    """Train Isolation Forest and compute risk scores."""
    print('\nTraining Isolation Forest...')
    iso = IsolationForest(**ISO_FOREST_PARAMS)
    iso.fit(X_train_scaled)

    # Train scores
    train_scores = iso.decision_function(X_train_scaled)
    train_risk = 1 - (train_scores - train_scores.min()) / \
                 (train_scores.max() - train_scores.min())
    train_features['iso_risk_score'] = train_risk

    # Test scores
    test_scores = iso.decision_function(X_test_scaled)
    test_risk = 1 - (test_scores - test_scores.min()) / \
                (test_scores.max() - test_scores.min())
    test_features['iso_risk_score'] = test_risk

    print(f'✅ ISO Forest | Train flagged: {(train_risk > 0.5).sum()} | '
          f'Test flagged: {(test_risk > 0.5).sum()}')
    return iso


def train_dbscan(train_features, test_features):
    """Run DBSCAN rendezvous detection on both sets."""
    print('Running DBSCAN...')
    for feat_df in [train_features, test_features]:
        moving = feat_df[feat_df['speed_mean'] >= 0.5].copy()
        coords = np.radians(moving[['last_lat', 'last_lon']].values)
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                    metric='haversine', n_jobs=-1)
        clusters = db.fit_predict(coords)
        moving['cluster_id'] = clusters
        moving['rendezvous_flag'] = (
            (moving['cluster_id'] != -1) &
            (moving['dist_from_port_km'] > RENDEZVOUS_PORT_DIST_KM)
        ).astype(int)
        feat_df['rendezvous_flag'] = 0
        feat_df.loc[moving.index, 'rendezvous_flag'] = \
            moving['rendezvous_flag'].values

    print(f'✅ DBSCAN | Train: {train_features["rendezvous_flag"].sum()} | '
          f'Test: {test_features["rendezvous_flag"].sum()}')


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # Decoder
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden_repeated = hidden.permute(1, 0, 2)
        output, _ = self.decoder(hidden_repeated)
        return output


def train_lstm_autoencoder(X_train_scaled, X_test_scaled, test_features):
    """Train LSTM Autoencoder using PyTorch (demo — excluded from ensemble)."""
    print('Training LSTM Autoencoder (PyTorch)...')
    
    # Convert numpy arrays to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
    
    input_dim = X_train_scaled.shape[1]
    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=64)
    
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Simple training loop (50 epochs, batch size 256 equivalent)
    model.train()
    dataset_size = len(X_train_tensor)
    batch_size = 256
    
    for epoch in range(50):
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_train_tensor[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            
    # Evaluation (Calculating reconstruction error)
    model.eval()
    with torch.no_grad():
        X_recon = model(X_test_tensor)
        recon_err = torch.mean((X_test_tensor - X_recon) ** 2, dim=(1, 2)).numpy()
        
    # Scale error to a 0-1 risk score
    lstm_risk = (recon_err - recon_err.min()) / (recon_err.max() - recon_err.min() + 1e-8)
    test_features['lstm_risk_score'] = lstm_risk
    print(f'✅ LSTM recon error mean: {recon_err.mean():.6f}')
    return model


def build_stack_features(X_scaled, feat_df):
    """Combine scaled features with unsupervised model scores."""
    return np.column_stack([
        X_scaled,
        feat_df['iso_risk_score'].values.reshape(-1, 1),
        feat_df['rendezvous_flag'].values.reshape(-1, 1),
    ])


def train_supervised_stack(X_train_stack, y_train, X_test_stack):
    """Train GradientBoosting + RandomForest ensemble with SMOTE."""
    print('\n' + '=' * 60)
    print('  TRAINING SUPERVISED STACKING MODEL')
    print('=' * 60)

    print(f'Stacked feature shape: {X_train_stack.shape}')
    print(f'Class distribution: Normal={sum(y_train == 0):,} | '
          f'Anomaly={sum(y_train == 1):,}')

    # SMOTE class balancing
    print('\nApplying SMOTE...')
    smote = SMOTE(random_state=42,
                  k_neighbors=min(5, sum(y_train == 1) - 1))
    X_balanced, y_balanced = smote.fit_resample(X_train_stack, y_train)
    print(f'After SMOTE: Normal={sum(y_balanced == 0):,} | '
          f'Anomaly={sum(y_balanced == 1):,}')

    # GradientBoosting
    print('\nTraining GradientBoosting...')
    gbc = GradientBoostingClassifier(**GBC_PARAMS)
    gbc.fit(X_balanced, y_balanced)

    # RandomForest
    print('Training RandomForest...')
    rfc = RandomForestClassifier(**RFC_PARAMS)
    rfc.fit(X_train_stack, y_train)

    # Ensemble probabilities
    gbc_proba = gbc.predict_proba(X_test_stack)[:, 1]
    rfc_proba = rfc.predict_proba(X_test_stack)[:, 1]
    ensemble_proba = gbc_proba * 0.6 + rfc_proba * 0.4

    print(f'\n✅ Supervised models trained')
    return gbc, rfc, ensemble_proba


def find_optimal_threshold(y_test, ensemble_proba):
    """Find the threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(
        y_test, ensemble_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    print(f'\nOptimal threshold: {optimal_threshold:.4f}')
    print(f'  Precision: {precisions[optimal_idx]:.2%}')
    print(f'  Recall:    {recalls[optimal_idx]:.2%}')
    print(f'  F1:        {f1_scores[optimal_idx]:.2%}')

    return optimal_threshold, precisions, recalls, thresholds

"""
VesselWatch V3 — Configuration & Constants
"""

# ── Raw Data Paths ────────────────────────────────────────
RAW_PATH = '/content/drive/MyDrive/VesselWatch/data/raw/'
PROCESSED_PATH = '/content/drive/MyDrive/VesselWatch/data/processed/'
OUTPUT_PATH = '/content/drive/MyDrive/VesselWatch/'

# ── Data Files ────────────────────────────────────────────
FILES = [
    'AIS_2023_01_11.csv', 'AIS_2023_01_12.csv',
    'AIS_2023_01_13.csv', 'AIS_2023_01_14.csv',
    'AIS_2023_01_15.csv', 'AIS_2023_01_16.csv',
    'AIS_2023_01_17.csv',
]

USE_COLS = [
    'MMSI', 'BaseDateTime', 'LAT', 'LON',
    'SOG', 'COG', 'Heading', 'VesselName',
    'VesselType', 'Length'
]

# ── Vessel Type Mapping ───────────────────────────────────
VESSEL_TYPE_MAP = {
    30: 'Fishing', 31: 'Towing', 32: 'Towing',
    33: 'Dredging', 34: 'Diving', 35: 'Military',
    36: 'Sailing', 37: 'Pleasure', 51: 'SAR',
    52: 'Tug', 60: 'Passenger', 61: 'Passenger',
    62: 'Passenger', 63: 'Passenger', 69: 'Passenger',
    70: 'Cargo', 71: 'Cargo', 72: 'Cargo',
    73: 'Cargo', 79: 'Cargo', 80: 'Tanker',
    81: 'Tanker', 82: 'Tanker', 83: 'Tanker', 89: 'Tanker'
}

# ── ML Feature Columns ───────────────────────────────────
ML_COLS = [
    # Original speed features
    'speed_mean', 'speed_std', 'speed_min', 'speed_max',
    'speed_variance',
    # Loitering
    'total_distance_km', 'total_time_hrs', 'loitering_score',
    # Gaps (expanded)
    'max_gap_hrs', 'total_gaps', 'gap_flag',
    'position_jump_km', 'avg_gap_hrs', 'gap_duration_std',
    # Port distance
    'dist_from_port_km',
    # Behavioral
    'behavioral_score', 'speed_consistency',
    # Direction changes
    'cog_change_mean', 'cog_change_max',
    # Interaction features
    'gap_x_jump', 'gap_x_port_dist', 'speed_range',
    'distance_per_hour', 'gap_ratio', 'ping_density', 'jump_per_gap',
    # Vessel-type Z-scores
    'max_gap_hrs_zscore', 'position_jump_km_zscore',
    'dist_from_port_km_zscore', 'speed_mean_zscore',
    'loitering_score_zscore',
]

# ── Model Hyperparameters ─────────────────────────────────
ISO_FOREST_PARAMS = {
    'n_estimators': 300,
    'contamination': 0.12,
    'random_state': 42,
    'n_jobs': -1,
}

GBC_PARAMS = {
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': 42,
}

RFC_PARAMS = {
    'n_estimators': 300,
    'max_depth': 8,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
}

# ── Thresholds ────────────────────────────────────────────
AIS_GAP_THRESHOLD_HRS = 6      # Minimum gap to be considered suspicious
DBSCAN_EPS = 0.009              # ~1km radius for rendezvous detection
DBSCAN_MIN_SAMPLES = 2
RENDEZVOUS_PORT_DIST_KM = 100   # Must be this far from port for rendezvous

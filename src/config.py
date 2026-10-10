"""
Configuration settings and constants for the dropout prediction project.

RANDOM_STATE is the single seed for the split, mutual information, L1 logistic
regression, XGBoost, stratified CV, and the Optuna sampler.
"""

from pathlib import Path

# File Paths
DATA_PATH = Path("data/data.csv")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "dropout_xgb.joblib"
FEATURES_PATH = MODELS_DIR / "selected_features.json"

# Column Definitions
TARGET_COL = "Target"
TARGET_CLASSES = ["Graduate", "Dropout"]
TARGET_POSITIVE_CLASS = "Dropout"
# Index 0 is Graduate and index 1 is Dropout.
CLASS_NAMES = ["Graduate", "Dropout"]

# Machine Learning Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CV_SPLITS = 5
DEFAULT_N_TRIALS = 20

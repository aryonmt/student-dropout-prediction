"""
Configuration settings and constants for the dropout prediction project.
"""

from pathlib import Path

# File Paths
DATA_PATH = Path("data/data.csv")

# Column Definitions
TARGET_COL = "Target"
TARGET_CLASSES = ["Graduate", "Dropout"]
TARGET_POSITIVE_CLASS = "Dropout"

# Machine Learning Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20

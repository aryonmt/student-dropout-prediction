"""
Data loading and preprocessing module.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Union
from src import config


def load_and_clean_data(file_path: Union[str, Path] = config.DATA_PATH) -> pd.DataFrame:
    """
    Loads data from CSV, cleans column names, and filters valid target classes.

    Args:
        file_path: The path to the CSV data file.

    Returns:
        A cleaned pandas DataFrame containing only the valid target classes.
    """
    df = pd.read_csv(file_path, sep=";", engine="python")
    df.columns = [c.strip().strip('"').replace("\t", "").strip() for c in df.columns]

    df = df[df[config.TARGET_COL].isin(config.TARGET_CLASSES)].copy()
    return df


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separates features from the target variable, converts the target to binary,
    and ensures all features are numeric without NaN values.

    Args:
        df: The cleaned pandas DataFrame.

    Returns:
        A tuple containing the features DataFrame (X) and target Series (y).
    """
    y = (df[config.TARGET_COL] == config.TARGET_POSITIVE_CLASS).astype(int)
    feature_cols = [c for c in df.columns if c != config.TARGET_COL]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = df[feature_cols].notna().all(axis=1)
    X = df.loc[mask, feature_cols].copy()
    y = y.loc[mask]

    return X, y


def split_dataset(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets with stratification.

    Args:
        X: Features DataFrame.
        y: Target Series.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

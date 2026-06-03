"""
Feature selection and importance module.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression

from src import config


def calculate_mutual_information(
    X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    """
    Calculates Mutual Information scores for features using ONLY training data.

    Args:
        X_train: Training features DataFrame.
        y_train: Training target Series.

    Returns:
        A DataFrame containing features sorted by their MI scores.
    """
    mi_scores = mutual_info_classif(
        X_train, y_train, random_state=config.RANDOM_STATE, n_neighbors=5
    )

    mi_df = (
        pd.DataFrame({"Feature": X_train.columns, "MI_Score": mi_scores})
        .sort_values("MI_Score", ascending=False)
        .reset_index(drop=True)
    )

    return mi_df


def get_l1_selected_features(
    X_train: pd.DataFrame, y_train: pd.Series, c_param: float = 0.01
) -> list[str]:
    """
    Selects features using L1 regularization (Lasso) via Logistic Regression.
    Features with non-zero coefficients are selected.

    Args:
        X_train: Training features DataFrame (should be scaled).
        y_train: Training target Series.
        c_param: Inverse of regularization strength (smaller values specify stronger regularization).

    Returns:
        A list of selected feature names.
    """
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=c_param,
        random_state=config.RANDOM_STATE,
        max_iter=1000,
    )
    model.fit(X_train, y_train)

    # Extract features that have non-zero coefficients
    coefs = np.abs(model.coef_[0])
    selected_mask = coefs > 0
    selected_features = X_train.columns[selected_mask].tolist()

    return selected_features


def combine_feature_strategies(
    mi_df: pd.DataFrame, l1_features: list[str], top_k_mi: int = 12
) -> list[str]:
    """
    Creates a final robust feature set by combining top MI features and L1 features.

    Args:
        mi_df: DataFrame output from calculate_mutual_information.
        l1_features: List of features selected by L1.
        top_k_mi: Number of top MI features to consider.

    Returns:
        A deduplicated list of consensus features.
    """
    top_mi_features = mi_df.head(top_k_mi)["Feature"].tolist()

    # Union of selected features
    combined_features = list(set(top_mi_features + l1_features))

    return combined_features

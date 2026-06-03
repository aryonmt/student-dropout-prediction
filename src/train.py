"""
Model training and hyperparameter tuning module using Optuna.
"""

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config
from xgboost import XGBClassifier


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Optuna objective function for tuning XGBoost hyperparameters.
    Optimizes for ROC-AUC using Stratified K-Fold Cross Validation.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "scale_pos_weight": float((y == 0).sum() / (y == 1).sum()),
        "random_state": config.RANDOM_STATE,
        "eval_metric": "auc",
        "verbosity": 0,
    }

    model = XGBClassifier(**params)

    # Robust evaluation using Stratified K-Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    # n_jobs=-1 ensures we use all CPU cores
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    return float(scores.mean())


def tune_and_train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 30
) -> XGBClassifier:
    """
    Runs Optuna study to find best hyperparameters and trains the final model.

    Args:
        X_train: Training features DataFrame.
        y_train: Training target Series.
        n_trials: Number of hyperparameter combinations to try.

    Returns:
        Trained XGBoost model with the best discovered hyperparameters.
    """
    # Suppress verbose Optuna logs to keep the terminal clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    print(f"Best CV ROC-AUC Score: {study.best_trial.value:.4f}")

    # Build final model parameters
    best_params = study.best_trial.params
    best_params["scale_pos_weight"] = float((y_train == 0).sum() / (y_train == 1).sum())
    best_params["random_state"] = config.RANDOM_STATE
    best_params["eval_metric"] = "auc"

    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    return final_model

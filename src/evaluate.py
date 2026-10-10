"""
Model evaluation and Explainable AI (SHAP) module.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, learning_curve
from xgboost import XGBClassifier

from src import config


def evaluate_performance(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """
    Evaluates the trained model on the test set and calculates key metrics.

    The positive class is 1 (Dropout).

    Args:
        model: Trained XGBoost model.
        X_test: Test features DataFrame.
        y_test: Test target Series.

    Returns:
        Accuracy, Balanced_Accuracy, Precision_Dropout, Recall_Dropout,
        F1_Dropout, F1_Macro, and ROC_AUC.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "Precision_Dropout": float(
            precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        ),
        "Recall_Dropout": float(
            recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        ),
        "F1_Dropout": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "F1_Macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_test, y_proba)),
    }

    print("\n" + "=" * 40)
    print("FINAL TEST SET PERFORMANCE")
    print("=" * 40)
    for metric_name, value in metrics.items():
        print(f"{metric_name:20}: {value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=config.CLASS_NAMES
        )
    )

    return metrics


def plot_confusion_matrix(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str | Path = config.REPORTS_DIR / "confusion_matrix.png",
) -> None:
    """
    Generates and saves the confusion matrix plot.

    Args:
        model: Trained XGBoost model.
        X_test: Test features DataFrame.
        y_test: Test target Series.
        save_path: Path to save the generated plot.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
    )
    plt.title("Confusion Matrix", fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Ensure directory exists before saving
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def explain_model_with_shap(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    save_path: str | Path = config.REPORTS_DIR / "shap_summary.png",
) -> None:
    """
    Uses SHAP (SHapley Additive exPlanations) to explain feature impacts.

    Args:
        model: Trained XGBoost model.
        X_test: Test features DataFrame.
        save_path: Path to save the SHAP summary plot.

    Returns:
        None
    """
    print("\nGenerating SHAP explanations (this might take a moment)...")

    # TreeExplainer is highly optimized for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)

    plt.title(
        "SHAP Feature Importance (Impact on Model Output)", pad=20, fontweight="bold"
    )

    # Ensure directory exists before saving
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to: {save_path}")


def plot_roc_curve(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str | Path = config.REPORTS_DIR / "roc_curve.png",
) -> None:
    """
    Generates and saves the Receiver Operating Characteristic (ROC) curve.

    Args:
        model: Trained XGBoost model.
        X_test: Test features DataFrame.
        y_test: Test target Series.
        save_path: Path to save the generated plot.

    Returns:
        None
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"ROC Curve saved to: {save_path}")


def plot_learning_curve(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_path: str | Path = config.REPORTS_DIR / "learning_curve.png",
) -> None:
    """
    Generates and saves the learning curve to diagnose overfitting or underfitting.

    Args:
        model: Trained XGBoost model.
        X_train: Training features DataFrame.
        y_train: Training target Series.
        save_path: Path to save the generated plot.

    Returns:
        None
    """
    print("\nGenerating Learning Curve (this might take a moment)...")

    cv = StratifiedKFold(
        n_splits=config.N_CV_SPLITS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))

    plt.plot(
        train_sizes,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="Training Score (ROC-AUC)",
    )
    plt.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )

    plt.plot(
        train_sizes,
        test_mean,
        color="green",
        linestyle="--",
        marker="s",
        markersize=5,
        label="Cross-Validation Score (ROC-AUC)",
    )
    plt.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )

    plt.title("Learning Curve", fontweight="bold")
    plt.xlabel("Number of Training Samples", fontweight="bold")
    plt.ylabel("ROC-AUC Score", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Learning Curve saved to: {save_path}")

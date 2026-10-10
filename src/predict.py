"""
Save, load, and apply the trained dropout classifier.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src import config, data_loader


def save_training_artifact(
    model: XGBClassifier,
    feature_names: list[str],
    model_path: str | Path = config.MODEL_PATH,
    features_path: str | Path = config.FEATURES_PATH,
) -> None:
    """
    Persist the trained model and the feature names it expects.

    Args:
        model: Trained XGBoost classifier.
        feature_names: Feature columns in the order used to fit the model.
        model_path: Destination path for the joblib model.
        features_path: Destination path for the JSON feature-name list.

    Returns:
        None
    """
    model_path = Path(model_path)
    features_path = Path(features_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    features_path.write_text(
        json.dumps(list(feature_names)) + "\n",
        encoding="utf-8",
    )


def load_training_artifact(
    model_path: str | Path = config.MODEL_PATH,
    features_path: str | Path = config.FEATURES_PATH,
) -> tuple[XGBClassifier, list[str]]:
    """
    Load a previously saved model and its feature names.

    Args:
        model_path: Path to the joblib model file.
        features_path: Path to the JSON feature-name list.

    Returns:
        The loaded classifier and the feature names it expects.

    Raises:
        FileNotFoundError: If the model file or the feature file is missing.
    """
    model_path = Path(model_path)
    features_path = Path(features_path)
    if not model_path.is_file() or not features_path.is_file():
        raise FileNotFoundError(
            "Training artifacts are missing. Run python main.py first."
        )
    model = joblib.load(model_path)
    feature_names = json.loads(features_path.read_text(encoding="utf-8"))
    return model, list(feature_names)


def score_students(
    frame: pd.DataFrame,
    model: XGBClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Score each student for dropout risk.

    Every name in feature_names must be present. Those columns are coerced
    with pd.to_numeric(errors='coerce'). A Target column, when present, is
    copied into the result and is not used as a feature.

    Args:
        frame: Student records to score.
        model: Trained XGBoost classifier.
        feature_names: Required feature columns in model order.

    Returns:
        A copy of the frame with dropout_probability and predicted_status.
        predicted_status is Dropout when the probability is at least 0.5,
        and Graduate otherwise.

    Raises:
        ValueError: If a required feature is absent or any required value is missing.
    """
    missing_columns = [name for name in feature_names if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"Input is missing required features: {missing_columns}")

    numeric = pd.DataFrame(
        {
            name: pd.to_numeric(frame[name], errors="coerce")
            for name in feature_names
        },
        index=frame.index,
    )
    if numeric.isna().any().any():
        raise ValueError("One or more required feature values are missing.")

    scored = frame.copy()
    probability = model.predict_proba(numeric)[:, 1]
    scored["dropout_probability"] = probability
    scored["predicted_status"] = [
        "Dropout" if value >= 0.5 else "Graduate" for value in probability
    ]
    return scored


def main() -> None:
    """
    Score a student CSV from the command line.

    Reads --input with the student CSV reader, scores every row with the saved
    artifact, and writes --output with index=False.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Score students with the saved dropout model."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    args = parser.parse_args()

    frame = data_loader.read_student_csv(args.input)
    model, feature_names = load_training_artifact()
    scored = score_students(frame, model, feature_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

"""
Main execution pipeline for the Dropout Prediction project.
"""

from src import config, data_loader, evaluate, features, train


def main() -> None:
    print("=" * 50)
    print("🚀 Starting Dropout Prediction Pipeline")
    print("=" * 50)

    # 1. Load and prepare data
    print("\n[1/5] Loading and cleaning data...")
    df = data_loader.load_and_clean_data()
    X, y = data_loader.prepare_features_and_target(df)

    # 2. Split dataset
    print(f"\n[2/5] Splitting dataset (Test Size: {config.TEST_SIZE})...")
    X_train, X_test, y_train, y_test = data_loader.split_dataset(X, y)
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 3. Feature Selection (Strictly on Training Data)
    print("\n[3/5] Performing Feature Selection to prevent Data Leakage...")
    mi_df = features.calculate_mutual_information(X_train, y_train)
    l1_features = features.get_l1_selected_features(X_train, y_train)

    final_features = features.combine_feature_strategies(mi_df, l1_features)
    print(f"Selected {len(final_features)} robust features out of {len(X.columns)}.")

    # Filter datasets to only keep selected features
    X_train_selected = X_train[final_features]
    X_test_selected = X_test[final_features]

    # 4. Train and Tune Model (Optuna)
    print("\n[4/5] Tuning hyperparameters and training XGBoost model...")
    best_model = train.tune_and_train_xgboost(X_train_selected, y_train, n_trials=20)

    # 5. Evaluate and Explain
    print("\n[5/5] Evaluating performance and generating explanations...")
    evaluate.evaluate_performance(best_model, X_test_selected, y_test)
    evaluate.plot_confusion_matrix(best_model, X_test_selected, y_test)
    evaluate.explain_model_with_shap(best_model, X_train_selected, X_test_selected)
    evaluate.plot_roc_curve(best_model, X_test_selected, y_test)

    print("\n✅ Pipeline completed successfully. Check the 'reports' folder for plots.")


if __name__ == "__main__":
    main()

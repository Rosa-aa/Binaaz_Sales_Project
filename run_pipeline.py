"""
run_pipeline.py
----------------
End-to-end execution: raw CSV -> cleaned data -> engineered features ->
model comparison -> tuned XGBoost -> evaluation.

Usage:
    python run_pipeline.py
"""

from config import DATA_PATH
from data_cleaning import clean_pipeline
from feature_engineering import feature_engineering_pipeline
from train import (
    drop_leakage_columns, split_features_target, train_test_split_data,
    build_preprocessor, cast_bool_columns, get_candidate_models, train_best_model,
)
from evaluate import (
    compare_models, evaluate_final_model, cross_validate_model,
    get_feature_importance, check_overfitting,
)


def main():
    # 1) Clean the raw data
    df_clean = clean_pipeline(DATA_PATH)

    # 2) Outlier removal, feature engineering, encoding
    df_model = feature_engineering_pipeline(df_clean)

    # 3) Drop leakage columns, split into X/y
    df_model_no_leak = drop_leakage_columns(df_model)
    X, y = split_features_target(df_model_no_leak)

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    X_train, X_test = cast_bool_columns(X_train, X_test)

    # 5) Preprocessing pipeline (impute + scale numeric, impute + one-hot categorical)
    preprocessor = build_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 6) Compare 5 candidate models
    models = get_candidate_models()
    results_df = compare_models(models, X_train_transformed, y_train, X_test_transformed, y_test)
    print(results_df)

    # 7) Train the final, tuned XGBoost model
    best_model = train_best_model(X_train_transformed, y_train)

    # 8) Evaluate
    evaluate_final_model(best_model, X_test_transformed, y_test)
    cross_validate_model(best_model, X_train_transformed, y_train)
    print(get_feature_importance(best_model, preprocessor))
    check_overfitting(best_model, X_train_transformed, y_train, X_test_transformed, y_test)


if __name__ == "__main__":
    main()

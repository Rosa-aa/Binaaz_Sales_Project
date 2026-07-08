"""
evaluate.py
-----------
Model evaluation: comparing all 5 candidate models, scoring the final tuned
model, cross-validation, feature importance, and an overfitting check.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def compare_models(models: dict, X_train_transformed, y_train, X_test_transformed, y_test) -> pd.DataFrame:
    """
    Trains every model in `models`, scores it on the test set (MAE, RMSE, R2),
    and returns a comparison DataFrame sorted by R2 (best first).
    """
    results = []
    for name, model in models.items():
        print("Training:", name)
        model.fit(X_train_transformed, y_train)
        y_pred = model.predict(X_test_transformed)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append([name, mae, rmse, r2])

    results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
    return results_df.sort_values("R2", ascending=False)


def evaluate_final_model(model, X_test_transformed, y_test) -> dict:
    """Prints and returns MAE / RMSE / R2 for the final tuned model on the test set."""
    y_pred = model.predict(X_test_transformed)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def cross_validate_model(model, X_train_transformed, y_train, cv: int = 5) -> dict:
    """Runs k-fold cross-validation (R2 scoring) and prints the mean/std across folds."""
    cv_scores = cross_val_score(model, X_train_transformed, y_train, cv=cv, scoring="r2")
    print("CV Mean R2:", cv_scores.mean())
    print("CV Std:", cv_scores.std())
    return {"mean_r2": cv_scores.mean(), "std_r2": cv_scores.std(), "scores": cv_scores}


def get_feature_importance(model, preprocessor, top_n: int = 10) -> pd.DataFrame:
    """Returns the top-N most important features according to the fitted tree-based model."""
    importances = model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    return feat_imp.head(top_n)


def check_overfitting(model, X_train_transformed, y_train, X_test_transformed, y_test) -> dict:
    """Compares train vs. test R2 to flag potential overfitting."""
    y_train_pred = model.predict(X_train_transformed)
    y_test_pred = model.predict(X_test_transformed)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("Train R2:", train_r2)
    print("Test R2:", test_r2)
    if train_r2 - test_r2 > 0.1:
        print("Warning: noticeable gap between train and test R2 — possible overfitting.")

    return {"train_r2": train_r2, "test_r2": test_r2}

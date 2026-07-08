"""
train.py
--------
Building the preprocessing pipeline, splitting the data, comparing five
regression models, and training the final tuned XGBoost model.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from config import LEAKAGE_COLS, TARGET_COL, TEST_SIZE, RANDOM_STATE, BEST_MODEL_PARAMS


def drop_leakage_columns(df_model: pd.DataFrame) -> pd.DataFrame:
    """Drops columns derived from the target (or otherwise unsafe) right before modeling."""
    return df_model.drop(columns=LEAKAGE_COLS, errors="ignore")


def split_features_target(df_model_no_leak: pd.DataFrame):
    """Splits the DataFrame into X (features) and y (target = qiymət)."""
    X = df_model_no_leak.drop(columns=[TARGET_COL])
    y = df_model_no_leak[TARGET_COL]
    return X, y


def train_test_split_data(X: pd.DataFrame, y: pd.Series):
    """80/20 train/test split (random, not time-based — this dataset has no meaningful time ordering)."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that:
      - imputes + scales numeric columns
      - imputes + one-hot encodes categorical columns
    Boolean columns (from one-hot encoding of şəhər) should be cast to int first — see cast_bool_columns().
    """
    num_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X_train.select_dtypes(exclude=["int64", "float64"]).columns

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features),
    ])


def cast_bool_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Casts boolean columns (created by pd.get_dummies) to int, so the preprocessor treats them as numeric."""
    bool_cols = X_train.select_dtypes(include="bool").columns
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[bool_cols] = X_train[bool_cols].astype(int)
    X_test[bool_cols] = X_test[bool_cols].astype(int)
    return X_train, X_test


def get_candidate_models() -> dict:
    """Returns the 5 regression models compared in this project, with their notebook-defined hyperparameters."""
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=12),
        "Random Forest": RandomForestRegressor(
            n_estimators=50, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE, n_estimators=100, learning_rate=0.1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            tree_method="hist", subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE,
        ),
    }


def train_best_model(X_train_transformed, y_train) -> XGBRegressor:
    """Trains the final, hyperparameter-tuned XGBoost model (see config.BEST_MODEL_PARAMS)."""
    model = XGBRegressor(**BEST_MODEL_PARAMS)
    model.fit(X_train_transformed, y_train)
    return model

"""
feature_engineering.py
------------------------
Outlier removal, new feature creation, and categorical encoding.

Pipeline stages:
    1. outlier_thresholds / remove_outliers — IQR-based cleaning of price, area, etc.
    2. add_engineered_features              — price_per_m2, floor_ratio, ground/top floor flags
    3. select_model_columns                 — drops non-feature columns (free text, images, etc.)
    4. encode_features                      — label / target / one-hot encoding
"""

import numpy as np
import pandas as pd

from config import (
    OUTLIER_CHECK_COLS, VIEW_COUNT_QUANTILE_CAP, NON_FEATURE_COLS,
    LABEL_ENCODE_COLS, TARGET_ENCODE_COLS, ONE_HOT_COLS, TARGET_COL,
)


def outlier_thresholds(df: pd.DataFrame, col: str, q1: float = 0.25, q3: float = 0.75):
    """Returns the (lower, upper) IQR-based outlier bounds for a column."""
    Q1 = df[col].quantile(q1)
    Q3 = df[col].quantile(q3)
    IQR = Q3 - Q1
    return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR


def report_outliers(df: pd.DataFrame, cols: list = OUTLIER_CHECK_COLS) -> None:
    """Prints how many outliers each column in `cols` contains, without removing anything."""
    for col in cols:
        low, up = outlier_thresholds(df, col)
        n = ((df[col] < low) | (df[col] > up)).sum()
        print(f"{col}: {n} outliers (lower={low:.0f}, upper={up:.0f})")


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows outside the IQR bounds for 'qiymət' (price), and rows above
    the 99th percentile of 'baxış_sayı' (view count).
    """
    df = df.copy()
    low_q, up_q = outlier_thresholds(df, "qiymət")
    df = df[(df["qiymət"] >= low_q) & (df["qiymət"] <= up_q)]

    q99_views = df["baxış_sayı"].quantile(VIEW_COUNT_QUANTILE_CAP)
    df = df[df["baxış_sayı"] <= q99_views]

    print("Shape after outlier removal:", df.shape)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates four new features:
      - qiymət_per_m2:    price / area
      - mərtəbə_nisbəti:  floor / total_floors
      - zemin_mərtəbə:    1 if ground floor, else 0
      - son_mərtəbə:      1 if top floor, else 0
    """
    df = df.copy()
    df["qiymət_per_m2"] = df["qiymət"] / df["sahə"].replace(0, np.nan)
    df["mərtəbə_nisbəti"] = df["mərtəbə"] / df["ümumi_mərtəbə"].replace(0, np.nan)
    df["zemin_mərtəbə"] = (df["mərtəbə"] == 1).astype(int)
    df["son_mərtəbə"] = (df["mərtəbə"] == df["ümumi_mərtəbə"]).astype(int)

    print("New features added. Shape:", df.shape)
    return df


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drops columns that can't be used as model features (free text, image URLs, owner name, address)."""
    df_model = df.drop(columns=NON_FEATURE_COLS, errors="ignore").copy()
    print("df_model shape:", df_model.shape)
    return df_model


def encode_features(df_model: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns using three strategies:
      - Label Encoding for low-cardinality columns
      - Target (mean) Encoding for high-cardinality columns (yer, agentlik_adı)
      - One-Hot Encoding for şəhər (city)
    """
    from sklearn.preprocessing import LabelEncoder

    df_model = df_model.copy()
    le = LabelEncoder()

    for col in LABEL_ENCODE_COLS:
        if col in df_model.columns:
            df_model[col] = le.fit_transform(df_model[col].astype(str))

    for col in TARGET_ENCODE_COLS:
        if col in df_model.columns:
            means = df_model.groupby(col)[TARGET_COL].mean()
            df_model[col] = df_model[col].map(means).astype(float)

    for col in ONE_HOT_COLS:
        if col in df_model.columns:
            df_model = pd.get_dummies(df_model, columns=[col], drop_first=True)

    print("Remaining object columns after encoding:",
          df_model.select_dtypes(include="object").columns.tolist())
    print("Shape:", df_model.shape)
    return df_model


def feature_engineering_pipeline(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Runs outlier removal, feature creation, column selection, and encoding in order."""
    report_outliers(df_clean)
    df_clean = remove_outliers(df_clean)
    df_clean = add_engineered_features(df_clean)
    df_model = select_model_columns(df_clean)
    df_model = encode_features(df_model)
    return df_model

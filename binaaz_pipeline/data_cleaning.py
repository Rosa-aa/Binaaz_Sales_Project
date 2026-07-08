"""
data_cleaning.py
-----------------
Loading and cleaning the raw bina.az listings CSV.

Pipeline stages:
    1. load_data                — reads the raw CSV
    2. drop_duplicate_columns   — removes duplicate/low-quality columns
    3. parse_structured_fields  — splits combined text fields (area, floor, etc.)
    4. clean_unit_price         — parses the price-per-m2 text field into a number
    5. rename_columns           — renames all columns to Azerbaijani
    6. handle_missing_values    — imputes remaining missing values
"""

import numpy as np
import pandas as pd

from config import (
    DUPLICATE_COLS, LOW_QUALITY_COLS, DATETIME_RAW_COLS,
    COLUMN_RENAME_MAP, POST_RENAME_DROP_COLS,
)


def load_data(path: str) -> pd.DataFrame:
    """Reads the raw bina.az CSV export."""
    return pd.read_csv(path)


def find_similar_columns(df: pd.DataFrame, threshold: float = 0.95) -> list:
    """
    Compares every pair of columns and returns those whose values match more
    than `threshold` of the time — a quick way to spot duplicate columns
    stored under two different names before deciding what to drop.
    """
    cols = df.columns.tolist()
    similar_pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1, col2 = cols[i], cols[j]
            try:
                similarity = (df[col1] == df[col2]).mean()
                if similarity > threshold:
                    similar_pairs.append((col1, col2, round(similarity, 4)))
            except Exception:
                pass
    return similar_pairs


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drops columns identified as duplicates, low-quality, or raw datetime fields not used in modeling."""
    df = df.drop(columns=DUPLICATE_COLS, errors="ignore")
    df = df.drop(columns=LOW_QUALITY_COLS, errors="ignore")

    # mortgage / İpoteka carry the same information — keep only one
    if "mortgage" in df.columns and "İpoteka" in df.columns:
        df = df.drop(columns=["mortgage"], errors="ignore")

    return df


def parse_structured_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits text fields that encode multiple values into separate numeric columns:
      - 'Sahə' ("145 m²")   -> area
      - 'Otaq sayı' ("4")   -> rooms
      - 'Mərtəbə' ("7/9")   -> floor, total_floors
    """
    df = df.copy()

    df["area"] = df["Sahə"].str.extract(r"(\d+)").astype(float)
    df["rooms"] = pd.to_numeric(df["Otaq sayı"], errors="coerce")

    floor_split = df["Mərtəbə"].str.extract(r"(\d+)\s*/\s*(\d+)")
    df["floor"] = pd.to_numeric(floor_split[0], errors="coerce")
    df["total_floors"] = pd.to_numeric(floor_split[1], errors="coerce")

    df = df.drop(columns=["Sahə", "Otaq sayı", "Mərtəbə"])
    return df


def drop_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parses then drops raw scrape-datetime columns — not used as model features."""
    df = df.copy()
    df["scrape_vaxtı"] = pd.to_datetime(df["datetime_scrape_x"], errors="coerce")
    df["scrape_vaxtı_2"] = pd.to_datetime(df["datetime_scrape_y"], errors="coerce")

    df = df.drop(columns=DATETIME_RAW_COLS, errors="ignore")
    df = df.drop(columns=["scrape_vaxtı", "scrape_vaxtı_2"], errors="ignore")
    return df


def clean_unit_price(df: pd.DataFrame) -> pd.DataFrame:
    """Strips 'AZN/m²' and spaces from the unit_price text field and converts it to numeric."""
    df = df.copy()
    df["unit_price"] = (
        df["unit_price"].astype(str)
        .str.replace("AZN/m²", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renames every column to Azerbaijani using config.COLUMN_RENAME_MAP, then drops leftover unused columns."""
    df = df.rename(columns=COLUMN_RENAME_MAP)
    df = df.drop(columns=POST_RENAME_DROP_COLS, errors="ignore")
    return df


def fix_duplicates_after_rename(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes inconsistent category values (e.g. 'var' -> 'Təmirli'),
    removes duplicate column names created by the rename step, and drops
    fully duplicate rows.
    """
    df = df.copy()
    if "təmir_statusu" in df.columns:
        df["təmir_statusu"] = df["təmir_statusu"].replace({"var": "Təmirli"})

    df = df.loc[:, ~df.columns.duplicated()]

    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows. Shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills remaining missing values:
      - categorical columns -> 'bilinmir' (unknown)
      - numeric columns     -> column median
      - datetime columns    -> column mode
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    datetime_cols = df.select_dtypes(include="datetime64[ns]").columns

    df[cat_cols] = df[cat_cols].fillna("bilinmir")
    df[num_cols] = df[num_cols].apply(lambda col: col.fillna(col.median()))
    for col in datetime_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    print("Remaining NaN:", df.isna().sum().sum())
    print("Shape:", df.shape)
    return df


def clean_pipeline(path: str) -> pd.DataFrame:
    """Runs the full cleaning pipeline in order and returns the cleaned DataFrame."""
    df = load_data(path)
    df = drop_duplicate_columns(df)
    df = parse_structured_fields(df)
    df = drop_datetime_columns(df)
    df = clean_unit_price(df)
    df = rename_columns(df)
    df = fix_duplicates_after_rename(df)
    df = handle_missing_values(df)
    return df

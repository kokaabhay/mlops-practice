import pandas as pd

from .config import ROOT


def validate_dataframe(df, text_col="text", label_col="label"):
    """Basic data checks. Raises ValueError if the data is unusable."""
    if len(df) == 0:
        raise ValueError("Dataset is empty")
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[text_col].isna().any() or df[label_col].isna().any():
        raise ValueError("Null values found")
    if (df[text_col].astype(str).str.strip() == "").any():
        raise ValueError("Empty text values found")
    return df


def load_split(cfg, split):
    """split is one of: train, val, test"""
    path = ROOT / cfg["data"]["raw_dir"] / cfg["data"][f"{split}_file"]
    df = pd.read_csv(path)
    return validate_dataframe(df, cfg["data"]["text_column"], cfg["data"]["label_column"])
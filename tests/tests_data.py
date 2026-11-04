import pandas as pd
import pytest

from src.config import ROOT
from src.data import validate_dataframe


def test_valid_dataframe_passes(tiny_df):
    assert len(validate_dataframe(tiny_df)) == len(tiny_df)


def test_missing_column_fails(tiny_df):
    with pytest.raises(ValueError, match="Missing columns"):
        validate_dataframe(tiny_df.drop(columns=["label"]))


def test_null_values_fail(tiny_df):
    tiny_df.loc[0, "text"] = None
    with pytest.raises(ValueError, match="Null"):
        validate_dataframe(tiny_df)


def test_empty_text_fails(tiny_df):
    tiny_df.loc[0, "text"] = "   "
    with pytest.raises(ValueError, match="Empty text"):
        validate_dataframe(tiny_df)


def test_empty_dataset_fails():
    with pytest.raises(ValueError, match="empty"):
        validate_dataframe(pd.DataFrame(columns=["text", "label"]))


RAW = ROOT / "data" / "raw" / "ledgar_train.csv"


@pytest.mark.skipif(not RAW.exists(), reason="raw data not present (run dvc pull)")
def test_real_training_data_is_valid():
    validate_dataframe(pd.read_csv(RAW))
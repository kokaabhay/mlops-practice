import pandas as pd
import pytest

from src.config import load_config

TEMPLATES = {
    "termination": "either party may terminate this agreement upon thirty days written notice",
    "governing law": "this agreement shall be governed by the laws of the state of new york",
    "indemnification": "the company shall indemnify and hold harmless the other party from claims",
}


@pytest.fixture
def tiny_df():
    rows = []
    for i in range(20):
        for label, text in TEMPLATES.items():
            rows.append({"text": f"{text} section {i}", "label": label})
    return pd.DataFrame(rows)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    # run in a temp dir so mlflow.db / mlruns never touch the repo
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    cfg["features"].update({"max_features": 1000, "min_df": 1, "ngram_range": [1, 2]})
    cfg["model"]["max_iter"] = 200
    cfg["training"]["max_train_samples"] = None
    return cfg
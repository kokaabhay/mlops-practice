from mlflow.tracking import MlflowClient

from src.promote import promote_if_better
from src.train import train


def test_first_model_becomes_champion(cfg, tiny_df):
    train(cfg, tiny_df, tiny_df)
    assert promote_if_better(cfg) is True

    client = MlflowClient()
    champ = client.get_model_version_by_alias(
        cfg["mlflow"]["registered_model_name"], cfg["mlflow"]["champion_alias"]
    )
    assert champ.version == "1"


def test_equal_model_does_not_replace_champion(cfg, tiny_df):
    train(cfg, tiny_df, tiny_df)
    promote_if_better(cfg)
    train(cfg, tiny_df, tiny_df)  # same score, version 2
    promote_if_better(cfg)

    client = MlflowClient()
    champ = client.get_model_version_by_alias(
        cfg["mlflow"]["registered_model_name"], cfg["mlflow"]["champion_alias"]
    )
    assert champ.version == "1"


def test_model_below_threshold_is_rejected(cfg, tiny_df):
    train(cfg, tiny_df, tiny_df)
    cfg["promotion"]["min_val_macro_f1"] = 1.01  # impossible bar
    assert promote_if_better(cfg) is False
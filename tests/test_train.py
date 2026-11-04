from mlflow.tracking import MlflowClient

from src.train import build_pipeline, evaluate, train


def test_pipeline_learns_tiny_dataset(cfg, tiny_df):
    pipe = build_pipeline(cfg)
    pipe.fit(tiny_df["text"], tiny_df["label"])
    metrics = evaluate(pipe, tiny_df["text"], tiny_df["label"])
    assert metrics["macro_f1"] > 0.9  # quality gate on the smoke dataset


def test_train_logs_and_registers_model(cfg, tiny_df):
    result = train(cfg, tiny_df, tiny_df, tiny_df)
    assert result["val_metrics"]["macro_f1"] > 0.9

    client = MlflowClient()
    name = cfg["mlflow"]["registered_model_name"]
    versions = client.search_model_versions(f"name='{name}'")
    assert len(versions) == 1
    metrics = client.get_run(result["run_id"]).data.metrics
    assert "val_macro_f1" in metrics and "test_macro_f1" in metrics
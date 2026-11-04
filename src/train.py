import argparse

import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from .config import load_config
from .data import load_split


def build_pipeline(cfg):
    f, m = cfg["features"], cfg["model"]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=f["max_features"],
                    ngram_range=tuple(f["ngram_range"]),
                    min_df=f["min_df"],
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=m["C"],
                    max_iter=m["max_iter"],
                    random_state=m["random_state"],
                ),
            ),
        ]
    )


def evaluate(pipeline, texts, labels):
    preds = pipeline.predict(texts)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def train(cfg, train_df, val_df, test_df=None, register=True):
    tc = cfg["data"]["text_column"]
    lc = cfg["data"]["label_column"]

    n = cfg["training"]["max_train_samples"]
    if n:
        train_df = train_df.sample(
            n=min(n, len(train_df)), random_state=cfg["model"]["random_state"]
        )

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    pipeline = build_pipeline(cfg)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "max_features": cfg["features"]["max_features"],
                "ngram_range": str(cfg["features"]["ngram_range"]),
                "min_df": cfg["features"]["min_df"],
                "C": cfg["model"]["C"],
                "max_iter": cfg["model"]["max_iter"],
                "n_train": len(train_df),
                "n_classes": train_df[lc].nunique(),
            }
        )
        mlflow.log_dict(cfg, "config.json")

        pipeline.fit(train_df[tc], train_df[lc])

        val_metrics = evaluate(pipeline, val_df[tc], val_df[lc])
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        test_metrics = None
        if test_df is not None:
            test_metrics = evaluate(pipeline, test_df[tc], test_df[lc])
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=cfg["mlflow"]["registered_model_name"] if register else None,
        )

    return {"run_id": run.info.run_id, "val_metrics": val_metrics, "test_metrics": test_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    if args.max_train_samples:
        cfg["training"]["max_train_samples"] = args.max_train_samples

    train_df = load_split(cfg, "train")
    val_df = load_split(cfg, "val")
    test_df = load_split(cfg, "test")

    result = train(cfg, train_df, val_df, test_df, register=not args.no_register)
    print("run_id:", result["run_id"])
    print("validation:", result["val_metrics"])
    print("test:", result["test_metrics"])


if __name__ == "__main__":
    main()
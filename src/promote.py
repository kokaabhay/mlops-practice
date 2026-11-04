"""Promote the latest registered version to 'champion' if it beats the current one.

Exits with code 1 if the latest version fails the minimum quality gate.
"""
import sys

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import load_config

METRIC = "val_macro_f1"


def promote_if_better(cfg):
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = MlflowClient()
    name = cfg["mlflow"]["registered_model_name"]
    alias = cfg["mlflow"]["champion_alias"]
    min_f1 = cfg["promotion"]["min_val_macro_f1"]

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError("No registered versions found. Run training first.")
    latest = max(versions, key=lambda v: int(v.version))
    candidate = client.get_run(latest.run_id).data.metrics[METRIC]

    if candidate < min_f1:
        print(f"Version {latest.version} FAILED gate: {METRIC}={candidate:.4f} < {min_f1}")
        return False

    try:
        champion = client.get_model_version_by_alias(name, alias)
        champion_score = client.get_run(champion.run_id).data.metrics.get(METRIC, float("-inf"))
    except MlflowException:
        champion, champion_score = None, float("-inf")

    if champion is not None and champion.version == latest.version:
        print(f"Version {latest.version} is already the champion.")
        return True

    if candidate > champion_score:
        client.set_registered_model_alias(name, alias, latest.version)
        print(f"Promoted version {latest.version} to '{alias}' ({METRIC}={candidate:.4f})")
        return True

    print(
        f"Kept version {champion.version} as '{alias}': "
        f"candidate {candidate:.4f} did not beat {champion_score:.4f}"
    )
    return True


if __name__ == "__main__":
    ok = promote_if_better(load_config())
    sys.exit(0 if ok else 1)
"""Classify contract clauses from the command line using the champion model.

Usage:
    python -m src.predict "Either party may terminate this agreement on 30 days notice."
    python -m src.predict            # interactive mode, empty line to quit
"""
import argparse

import mlflow
import mlflow.sklearn
import numpy as np

from src.config import load_config


def load_champion():
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    uri = f"models:/{cfg['mlflow']['registered_model_name']}@{cfg['mlflow']['champion_alias']}"
    return mlflow.sklearn.load_model(uri)


def predict(model, clauses, top_k=3):
    proba = model.predict_proba(clauses)
    results = []
    for row in proba:
        order = np.argsort(row)[::-1][:top_k]
        results.append([(str(model.classes_[i]), float(row[i])) for i in order])
    return results


def show(clause, preds):
    print(f"\nClause: {clause[:100]}{'...' if len(clause) > 100 else ''}")
    for label, score in preds:
        print(f"  {score:6.1%}  {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clauses", nargs="*", help="one or more clauses (quote each)")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    model = load_champion()

    if args.clauses:
        for clause, preds in zip(args.clauses, predict(model, args.clauses, args.top_k)):
            show(clause, preds)
        return

    print("Paste a clause and press Enter (empty line to quit).")
    while True:
        clause = input("\n> ").strip()
        if not clause:
            break
        show(clause, predict(model, [clause], args.top_k)[0])


if __name__ == "__main__":
    main()
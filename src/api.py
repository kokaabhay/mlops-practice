import os
from typing import Annotated

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field, StringConstraints

from src.config import load_config

Clause = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10000)]

_model = None


def model_uri():
    cfg = load_config()
    default = f"models:/{cfg['mlflow']['registered_model_name']}@{cfg['mlflow']['champion_alias']}"
    return os.getenv("MODEL_URI", default)


def get_model():
    global _model
    if _model is None:
        cfg = load_config()
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        _model = mlflow.sklearn.load_model(model_uri())
    return _model


class PredictRequest(BaseModel):
    clauses: list[Clause] = Field(min_length=1, max_length=100)
    top_k: int = Field(default=3, ge=1, le=10)


class Prediction(BaseModel):
    label: str
    score: float


class ClauseResult(BaseModel):
    label: str
    score: float
    top_k: list[Prediction]


class PredictResponse(BaseModel):
    model_uri: str
    results: list[ClauseResult]


app = FastAPI(title="Legal Clause Classifier")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model=Depends(get_model)):
    proba = model.predict_proba(req.clauses)
    classes = model.classes_
    results = []
    for row in proba:
        order = np.argsort(row)[::-1][: req.top_k]
        top = [Prediction(label=str(classes[i]), score=float(row[i])) for i in order]
        results.append(ClauseResult(label=top[0].label, score=top[0].score, top_k=top))
    return PredictResponse(model_uri=model_uri(), results=results)
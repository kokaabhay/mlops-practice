# 1. environment (Windows: .venv\Scripts\activate)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# 2. tests first, to confirm the setup works
git init
pytest -q

# 3. get the data, then version it with DVC
python -m src.prepare_data
dvc init
dvc remote add -d localstore ../dvc-store     # a folder outside the repo
dvc add data/raw
git add data/raw.dvc data/.gitignore .dvc .gitignore
dvc push
git add .
git commit -m "Initial project"

# 4. train: quick run first, then full
python -m src.train --max-train-samples 5000
python -m src.train
python -m src.promote

# 5. look at experiments and registry
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000

# 6. serve the champion model
uvicorn src.api:app --reload                          # try http://127.0.0.1:8000/docs
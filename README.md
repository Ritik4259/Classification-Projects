# ML Classification Suite  
A modular machine learning system implementing Logistic Regression, Decision Trees, and Support Vector Machines (SVM) across multiple datasets to analyze and compare model performance.

---

## Project Overview
The **ML Classification Suite** is a unified machine learning framework designed to explore how different classical ML algorithms perform on varied datasets. It follows a structured pipeline covering:

- Data loading  
- Preprocessing  
- Model training  
- Evaluation  
- Visualization  
- Comparative analysis  

This project was built entirely by me as part of my ongoing learning in machine learning.

---

## Problem Statement
Different datasets have different structures, levels of complexity, and decision boundaries.  
No single ML model performs best on every dataset.

This project solves the problem of understanding **which model works best for which dataset** by implementing and comparing three widely used classification algorithms.

---

## Tech Stack

**Language:**  
- Python 3.8+

**Libraries:**  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-Learn  

**Tools:**  
- Jupyter Notebook / Google Colab  

---

## System Architecture
```

                ┌───────────────────────┐
                │      Dataset Loader   │
                │  (student, diabetes)  │
                └──────────┬────────────┘
                           │
                ┌──────────▼────────────┐
                │     Data Preprocessor │
                │ (cleaning, encoding)  │
                └──────────┬────────────┘
                           │
          ┌────────────────┼─────────────────┐
          ▼                ▼                 ▼
┌────────────────┐  ┌───────────────┐  ┌────────────────┐
│ Logistic Reg.  │  │ Decision Tree │  │       SVM      │
└──────┬─────────┘  └──────┬────────┘  └─────────┬──────┘
       │                   │                     │
       └──────────────┬────┴───────────┬────────┘
                      ▼                ▼
             ┌────────────────┐  ┌──────────────────┐
             │ Model Metrics  │  │ Visualization    │
             └────────────────┘  └──────────────────┘

```
---

## Features

### Multi-dataset support  
- Diabetes dataset  
- Student performance dataset  
- Student scores dataset  

### Three machine learning models  
- Logistic Regression  
- Decision Tree Classifier  
- Support Vector Machine  

### Modular, reusable pipeline  
Each notebook includes:
- Data preprocessing  
- Model training  
- Evaluation  
- Visualizations  

### Visualization & comparison  
- Accuracy scores  
- Confusion matrix  
- Classification report  

---

## Models Used

| Model | Use-Case Strength | Trade-Off |
|-------|------------------|-----------|
| Logistic Regression | Linearly separable data | Limited on complex datasets |
| Decision Tree | Handles non-linear data | Overfits without pruning |
| SVM | High accuracy on scaled data | Computationally heavy |

---

## Metrics & Observations
- Logistic Regression performed best on linearly separable datasets.  
- Decision Trees performed well but showed signs of overfitting.  
- SVM achieved strong performance on the diabetes dataset after scaling.  

This highlights the importance of **choosing the right model for the right data**.

---

Ritik — below is a ready-to-copy **“Setup & Run”** section (plus the requested extras) you can paste straight into your `README.md`. I kept it concrete (commands, file names) and added optional DB / deployment guidance so the doc stays useful whether you run locally or deploy later.

---

## Setup & Run (Step-by-Step)

### Prerequisites

* Python 3.8+ installed
* `git` installed (optional)
* Recommended: virtual environment tool (`venv` or `conda`)
* Jupyter Notebook / Google Colab to run notebooks

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd Classification-Projects-main
```

### 2. Create and activate a virtual environment (recommended)

Using `venv`:

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

Or using `conda`:

```bash
conda create -n ml-suite python=3.9 -y
conda activate ml-suite
```

### 3. Install dependencies

Make a `requirements.txt` (example below) and then:

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`**

```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyterlab
joblib
fastapi     # optional, if you deploy an API
uvicorn     # optional, if you deploy an API
```

### 4. Prepare datasets

The `datasets/` folder already contains:

* `diabetes_dataset.csv`
* `Student_Performance.csv`
* `student_scores.csv`

If you use new datasets, place them inside `datasets/` and update the notebook file paths.

### 5. Run notebooks (local)

Start Jupyter Lab / Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

Open one of:

* `LogisticRegression.ipynb`
* `DecisionTree.ipynb`
* `SVM.ipynb`

Each notebook contains sequential cells for data loading → preprocessing → training → evaluation → visualization. Run cells top-to-bottom.

### 6. Run evaluation script (optional)

You can add a small script `evaluate.py` (example below) to compute common metrics for saved models:

```python
# evaluate.py (example)
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model = joblib.load("models/logreg_v1.joblib")
df = pd.read_csv("datasets/diabetes_dataset.csv")
# preprocess same as notebook...
X_test, y_test = ...  # follow notebook preprocessing
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Run:

```bash
python evaluate.py
```

## Key Components & ML Interfaces

### Core ML components 

* **Dataset loaders** — functions / cells that read CSVs from `datasets/`
* **Preprocessing pipeline** — missing-value handling, encoding categorical variables, feature scaling (StandardScaler) for SVM
* **Model training** — `sklearn.linear_model.LogisticRegression`, `sklearn.tree.DecisionTreeClassifier`, `sklearn.svm.SVC`
* **Model persistence** — save trained models via `joblib.dump(model, "models/<name>.joblib")`
* **Evaluation utilities** — confusion matrix, classification report, accuracy, ROC-AUC functions in notebooks

### API (via FastAPI)

These are **recommended** endpoints to implement when you convert notebooks into a service:

* `POST /predict`

  * Payload: `{"model":"logreg","features":[...]}`
  * Response: `{"prediction": 1, "probability": 0.87}`

* `GET /models`

  * Response: `{"available_models": ["logreg_v1","dt_v1","svm_v1"]}`

* `POST /batch-predict`

  * Payload: CSV or JSON array of feature rows
  * Response: JSON array of predictions

* `GET /metrics` (admin)

  * Returns latest evaluation metrics (accuracy, precision, recall, f1) for deployed model

### Example minimal FastAPI server skeleton

```python
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("models/logreg_v1.joblib")

@app.post("/predict")
def predict(data: dict):
    features = np.array(data["features"]).reshape(1, -1)
    pred = int(model.predict(features)[0])
    prob = float(model.predict_proba(features).max())
    return {"prediction": pred, "probability": prob}
```

### Database / Schemas

If you want to log requests, predictions and metrics, a simple SQLite/Postgres schema example:

**Table: `predictions`**

* `id` (PK, integer)
* `timestamp` (datetime)
* `model_name` (text)
* `input_json` (text)
* `predicted_label` (integer)
* `predicted_prob` (real)
* `ground_truth` (integer, nullable)

**Table: `metrics`**

* `id` (PK)
* `model_name`
* `run_timestamp`
* `accuracy`
* `precision`
* `recall`
* `f1_score`
* `notes` (text)

You can implement DB using SQLAlchemy or an ORM.

---

## Example deployment stack 

* Backend: FastAPI served by `uvicorn`
* Container: Docker (create `Dockerfile`)
* Hosting: Render / Heroku / Railway / AWS Elastic Beanstalk / Google Cloud Run
* Model files stored in repository or in object storage (S3) for larger models


## Impact & Metrics (How to measure & report)

Use these metrics to summarize model impact and performance in your README / demo:

### Primary metrics

* **Accuracy** — overall correctness
* **Precision** — correctness of positive predictions
* **Recall** — how many positives were found
* **F1-score** — harmonic mean of precision & recall
* **Confusion Matrix** — view class-wise errors
* **ROC AUC** — for binary classifiers

### How to generate the report (example)

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
```

### Example performance summary (explainable, not fabricated)

* “On `diabetes_dataset.csv`, after standard scaling and train/test split (80/20), the SVM (RBF) achieved higher ROC-AUC than Logistic Regression. See `notebooks/` for exact numbers and plots.”
* Provide a table in README with actual numbers produced by your notebooks:

  * Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC

### Scale & assumptions

* These models are lightweight (classical ML) and suitable for small-to-medium tabular datasets (<100k rows).
* For production at scale, consider:

  * model batching & caching
  * async request handling
  * feature store or persistent preprocessing pipeline
  * horizontal scaling of API containers

---

## What’s Next (Limitations & Planned Improvements)

**Limitations**

* Notebooks are not yet modularized as importable Python packages.
* No production-grade API / auth / monitoring in current repo.
* No automated hyperparameter tuning (GridSearch / RandomSearch) yet.
* Limited explainability (no SHAP/LIME analysis included).

**Planned improvements**

1. Convert notebooks into modular Python scripts / package (`ml_suite/`) with CLI.
2. Add hyperparameter tuning with `GridSearchCV` and cross-validation.
3. Add additional models: Random Forest, XGBoost, LightGBM.
4. Add model explainability (SHAP) and model card.
5. Build a FastAPI + Docker deployment with logging, metrics, and health checks.
6. Create a simple frontend (Streamlit) for interactive dataset upload and predictions.
7. CI/CD: tests, linting, and automatic model validation before deploy.



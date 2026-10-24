import json
import os

import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression

from config.settings import POSTGRES_CONN_STRING

SEED = 42
MAX_SAMPLES = 100000
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/airflow/data/models/2022_2023")
BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "best_params.json")

# Identifiants : ne doivent pas servir de variables explicatives
DROP_COLS = ["id_accident", "annee", "id_usager", "id_vehicule"]

# nom -> (classe, paramètres FIXES). Les hyperparamètres optimisés viennent de best_params.json
MODELS = {
    "rf_model": (RandomForestClassifier, {"random_state": SEED, "n_jobs": -1}),
    "gb_model": (GradientBoostingClassifier, {"random_state": SEED}),
    "extratrees_model": (ExtraTreesClassifier, {"random_state": SEED, "n_jobs": -1}),
    "adaboost_model": (AdaBoostClassifier, {"random_state": SEED}),
    "logreg_model": (LogisticRegression, {"max_iter": 1000}),
}


def get_dataset():
    engine = create_engine(POSTGRES_CONN_STRING)
    query = """
    SELECT f.*, l.*, u.*, v.*
    FROM fact_accidents f
    LEFT JOIN dim_lieux l ON f.id_accident = l.id_accident
    LEFT JOIN dim_usagers u ON f.id_accident = u.id_accident
    LEFT JOIN dim_vehicules v ON f.id_accident = v.id_accident
    """
    df = pd.read_sql(query, engine)
    df = df.dropna(subset=["gravite"])

    # Groupe = accident (le JOIN produit plusieurs lignes par accident)
    groups = df.loc[:, "id_accident"]
    if isinstance(groups, pd.DataFrame):  # colonne dupliquée par le SELECT f.*, l.*, ...
        groups = groups.iloc[:, 0]

    y = df["gravite"]
    X = df.drop(columns=DROP_COLS + ["gravite"], errors="ignore")

    cat_cols = X.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))

    if len(X) > MAX_SAMPLES:
        idx = X.sample(n=MAX_SAMPLES, random_state=SEED).index
        X, y, groups = X.loc[idx], y.loc[idx], groups.loc[idx]
    return X, y, groups


def split_data(X, y, groups):
    """Split train/test par accident. Utilisé aussi par tune_models.py
    pour que le réglage ne touche JAMAIS le jeu de test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], groups.iloc[tr]


def load_best_params():
    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH) as f:
            return json.load(f)
    print("⚠️ best_params.json introuvable : paramètres par défaut utilisés.")
    return {}


def train_and_save(name, cls, fixed, tuned, X_train, X_test, y_train, y_test):
    model = cls(**fixed, **tuned)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {"accuracy": accuracy_score(y_test, y_pred)}
    for avg in ("macro", "macro"):
        metrics[f"precision_{avg}"] = precision_score(y_test, y_pred, average=avg, zero_division=0)
        metrics[f"recall_{avg}"] = recall_score(y_test, y_pred, average=avg, zero_division=0)
        metrics[f"f1_{avg}"] = f1_score(y_test, y_pred, average=avg, zero_division=0)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
    with open(os.path.join(MODEL_DIR, f"{name}_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write(f"params: {json.dumps(tuned)}\n")

    print(f"✅ {name} accuracy={metrics['accuracy']:.3f} f1_macro={metrics['f1_macro']:.3f}")


if __name__ == "__main__":
    X, y, groups = get_dataset()  # chargé UNE seule fois
    X_train, X_test, y_train, y_test, _ = split_data(X, y, groups)
    best = load_best_params()
    for name, (cls, fixed) in MODELS.items():
        train_and_save(name, cls, fixed, best.get(name, {}), X_train, X_test, y_train, y_test)

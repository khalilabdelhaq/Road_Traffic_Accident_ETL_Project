import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config.settings import POSTGRES_CONN_STRING

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression

MAX_SAMPLES = 100000


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
    y = df["gravite"]
    X = df.drop(columns=["gravite", "id_accident", "annee"], errors="ignore")
    # Encodage compact
    cat_cols = X.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))

    # Sous-échantillonnage
    if len(X) > MAX_SAMPLES:
        X_sampled = X.sample(n=MAX_SAMPLES, random_state=42)
        y = y.loc[X_sampled.index]
        X = X_sampled
    return X, y


def train_and_save(model_cls, name):
    X, y = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = model_cls()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    MODEL_DIR = "/opt/airflow/data/models/2022_2023"
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

    with open(os.path.join(MODEL_DIR, f"{name}_metrics.txt"), "w") as f:
        f.write(f"accuracy: {acc:.4f}\n")
        f.write(f"precision: {prec:.4f}\n")
        f.write(f"recall: {rec:.4f}\n")
        f.write(f"f1_score: {f1:.4f}\n")

    print(f"✅ {name} accuracy: {acc:.3f}")


train_models = [
    ("rf_model", RandomForestClassifier),
    ("gb_model", GradientBoostingClassifier),
    ("extratrees_model", ExtraTreesClassifier),
    ("adaboost_model", AdaBoostClassifier),
    ("logreg_model", LogisticRegression),
]

for name, cls in train_models:
    train_and_save(cls, name)

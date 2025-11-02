import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config.settings import POSTGRES_CONN_STRING

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression


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
    return pd.get_dummies(X), y


def train_and_save(model_cls, name):
    X, y = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = model_cls()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    model_dir = "/opt/airflow/data/models/2022_2023"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f"{name}.pkl"))

    with open(os.path.join(model_dir, f"{name}_accuracy.txt"), "w") as f:
        f.write(str(acc))

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

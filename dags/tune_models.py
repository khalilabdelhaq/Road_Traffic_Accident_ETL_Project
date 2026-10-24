"""Optimisation des hyperparamètres (RandomizedSearchCV, CV stratifiée par accident, macro-F1).
Écrit best_params.json, tuning_summary.json et un CSV de cv_results par modèle."""
import json
import os
import time

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from train_models import (
    BEST_PARAMS_PATH, MODEL_DIR, MODELS, SEED, get_dataset, split_data,
)

N_ITER = int(os.environ.get("TUNING_N_ITER", 20))
CV_FOLDS = int(os.environ.get("TUNING_CV_FOLDS", 5))
TUNING_SAMPLES = int(os.environ.get("TUNING_SAMPLES", 30000))  # 0 = tout le train
FORCE = os.environ.get("FORCE_TUNING", "0") == "1"
SCORING = "f1_macro"

SEARCH_SPACES = {
    "rf_model": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 20, 30, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"],
    },
    "extratrees_model": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 20, 30, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"],
    },
    "gb_model": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 1.0],
        "min_samples_leaf": [1, 5, 10],
    },
    "adaboost_model": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    },
    "logreg_model": {
        "C": [0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"],
    },
}


def grid_size(space):
    n = 1
    for v in space.values():
        n *= len(v)
    return n


def main():
    if os.path.exists(BEST_PARAMS_PATH) and not FORCE:
        print(f"⏩ {BEST_PARAMS_PATH} existe déjà, réglage ignoré (FORCE_TUNING=1 pour relancer).")
        return

    X, y, groups = get_dataset()
    X_train, _, y_train, _, g_train = split_data(X, y, groups)  # test set jamais utilisé ici

    if TUNING_SAMPLES and len(X_train) > TUNING_SAMPLES:
        idx = X_train.sample(n=TUNING_SAMPLES, random_state=SEED).index
        X_train, y_train, g_train = X_train.loc[idx], y_train.loc[idx], g_train.loc[idx]
    print(f"Réglage sur {len(X_train)} lignes, CV={CV_FOLDS} plis, scoring={SCORING}")

    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    best_params, summary = {}, {}
    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, (cls, fixed) in MODELS.items():
        space = SEARCH_SPACES[name]
        n_iter = min(N_ITER, grid_size(space))
        t0 = time.time()
        search = RandomizedSearchCV(
            cls(**fixed), space, n_iter=n_iter, scoring=SCORING, cv=cv,
            random_state=SEED, refit=False,
            n_jobs=1 if "n_jobs" in fixed else -1,  # évite le sur-parallélisme
        )
        search.fit(X_train, y_train, groups=g_train)

        best_params[name] = search.best_params_
        summary[name] = {
            "search_space": space,
            "n_iter": n_iter,
            "best_params": search.best_params_,
            "best_cv_f1_macro_mean": float(search.best_score_),
            "best_cv_f1_macro_std": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
            "seconds": round(time.time() - t0, 1),
        }
        pd.DataFrame(search.cv_results_).to_csv(
            os.path.join(MODEL_DIR, f"{name}_cv_results.csv"), index=False
        )
        print(f"✅ {name}: {search.best_params_} | CV macro-F1={search.best_score_:.4f}")

    meta = {"scoring": SCORING, "cv_folds": CV_FOLDS, "tuning_rows": len(X_train),
            "seed": SEED, "models": summary}
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    with open(os.path.join(MODEL_DIR, "tuning_summary.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

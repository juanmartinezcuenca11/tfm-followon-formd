"""
10_red_neuronal.py
Tercera familia de modelos: red neuronal (perceptrón multicapa, MLP).

Usa el MLP de scikit-learn sobre el MISMO panel y el MISMO split temporal que
los modelos anteriores, con un preprocesado equivalente al del baseline
(imputación + escalado + one-hot), ya que las redes no manejan categóricas
nativas. Se añade su fila a la comparación de modelos.

Salida: models/red_neuronal.pkl, models/metricas_red.json, models/comparacion_final.json
Uso:    python scripts/10_red_neuronal.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

N_COHORTES_TEST = 2
MIN_FRECUENCIA_CAT = 0.01

FEATURES_NUM = [
    "TOTALOFFERINGAMOUNT_NUM", "TOTALAMOUNTSOLD_NUM", "RATIO_COLOCADO",
    "MINIMUMINVESTMENTACCEPTED_NUM", "TOTALNUMBERALREADYINVESTED_NUM",
    "N_DIRECTIVOS",
]
FEATURES_BIN = [
    "OBJETIVO_INDEFINIDO", "PAGO_COMISIONES", "REVELA_INGRESOS",
    "ISEQUITYTYPE_BIN", "ISDEBTTYPE_BIN", "ISSECURITYTOBEACQUIREDTYPE_BIN",
    "ISOPTIONTOACQUIRETYPE_BIN",
]
FEATURES_CAT = ["INDUSTRYGROUPTYPE", "STATEORCOUNTRY", "ENTITYTYPE",
                "REVENUERANGE", "ANTIGUEDAD"]
TARGET = "FOLLOWON"


def log_transform(X):
    return np.log1p(np.clip(X, a_min=0, a_max=None))


def evaluar(y_true, y_prob, etiqueta: str) -> dict:
    m = {
        "n": int(len(y_true)),
        "tasa_base": round(float(np.mean(y_true)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
    }
    print(f"  {etiqueta}: n={m['n']:,} | base={m['tasa_base']:.1%} | "
          f"ROC-AUC={m['roc_auc']:.3f} | PR-AUC={m['pr_auc']:.3f} | "
          f"Brier={m['brier']:.4f}")
    return m


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")

    cohortes = sorted(panel["cohorte"].unique())
    cohortes_test = cohortes[-N_COHORTES_TEST:]
    train = panel[~panel["cohorte"].isin(cohortes_test)].copy()
    test = panel[panel["cohorte"].isin(cohortes_test)].copy()
    print(f"Split temporal -> train: {cohortes[0]}-{cohortes_test[0]-1} "
          f"({len(train):,}) | test: {list(cohortes_test)} ({len(test):,})\n")

    # Agrupar categorías raras (umbral aprendido en train)
    for col in FEATURES_CAT:
        frec = train[col].fillna("DESCONOCIDO").value_counts(normalize=True)
        raras = set(frec[frec < MIN_FRECUENCIA_CAT].index)
        for df in (train, test):
            df[col] = df[col].fillna("DESCONOCIDO")
            df.loc[df[col].isin(raras), col] = "OTROS"

    X_cols = FEATURES_NUM + FEATURES_BIN + FEATURES_CAT

    numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(log_transform, feature_names_out="one-to-one")),
        ("escalar", StandardScaler()),
    ])
    categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preproceso = ColumnTransformer([
        ("num", numericas, FEATURES_NUM),
        ("bin", "passthrough", FEATURES_BIN),
        ("cat", categoricas, FEATURES_CAT),
    ])

    pipeline = Pipeline([
        ("preproceso", preproceso),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),   # dos capas ocultas
            activation="relu",
            alpha=1e-3,                    # regularización L2
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=100,
            early_stopping=True,           # separa validación interna
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )),
    ])

    print("Entrenando red neuronal (puede tardar unos minutos)...")
    pipeline.fit(train[X_cols], train[TARGET])
    print(f"Iteraciones hasta converger: {pipeline.named_steps['mlp'].n_iter_}\n")

    print("Resultados red neuronal:")
    metricas = {
        "train": evaluar(train[TARGET], pipeline.predict_proba(train[X_cols])[:, 1], "train"),
        "test": evaluar(test[TARGET], pipeline.predict_proba(test[X_cols])[:, 1], "test "),
    }
    print("\nTest por cohorte:")
    metricas["test_por_cohorte"] = {}
    for c in cohortes_test:
        sub = test[test["cohorte"] == c]
        prob = pipeline.predict_proba(sub[X_cols])[:, 1]
        metricas["test_por_cohorte"][str(c)] = evaluar(sub[TARGET], prob, f"  {c}")

    # ---- Comparación final de las tres familias -----------------------------
    comp = {"red_neuronal_test": metricas["test"]}
    for nombre, fichero in [("baseline", "metricas_baseline.json"),
                            ("catboost", "metricas_catboost.json")]:
        ruta = MODELS_DIR / fichero
        if ruta.exists():
            comp[nombre + "_test"] = json.loads(ruta.read_text())["test"]

    if "baseline_test" in comp and "catboost_test" in comp:
        print("\n" + "=" * 60)
        print("COMPARACIÓN FINAL EN TEST (2021-2022)")
        print("=" * 60)
        print(f"{'Modelo':<22}{'ROC-AUC':>10}{'PR-AUC':>10}{'Brier':>10}")
        filas = [("Regresión logística", "baseline_test"),
                 ("CatBoost", "catboost_test"),
                 ("Red neuronal", "red_neuronal_test")]
        for etiqueta, clave in filas:
            m = comp[clave]
            print(f"{etiqueta:<22}{m['roc_auc']:>10.3f}{m['pr_auc']:>10.3f}{m['brier']:>10.4f}")
        print("(ROC-AUC y PR-AUC: más alto mejor; Brier: más bajo mejor)")

    with open(MODELS_DIR / "red_neuronal.pkl", "wb") as fh:
        pickle.dump({"pipeline": pipeline, "features": X_cols}, fh)
    (MODELS_DIR / "metricas_red.json").write_text(json.dumps(metricas, indent=2))
    (MODELS_DIR / "comparacion_final.json").write_text(json.dumps(comp, indent=2))
    print(f"\nGuardado: red_neuronal.pkl, metricas_red.json, comparacion_final.json")


if __name__ == "__main__":
    main()

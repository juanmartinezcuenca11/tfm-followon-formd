"""
05_baseline_logistico.py
Primer modelo: regresión logística sobre el panel de primeras rondas.

Diseño:
  - Split TEMPORAL por cohortes (nunca aleatorio): entrenamos con el pasado
    y evaluamos con cohortes posteriores, como se usaría en producción.
    Un split aleatorio filtraría información del futuro (mismo ciclo macro
    en train y test) e inflaría las métricas.
  - Preprocesado en Pipeline de sklearn: imputación + escalado para
    numéricas, agrupación de categorías raras + one-hot para categóricas.
    Todo se ajusta SOLO con train.
  - Métricas: ROC-AUC y PR-AUC (la clase positiva no es mayoritaria),
    más desglose por cohorte de test para ver estabilidad temporal.

Salida: models/baseline_logistico.pkl, models/metricas_baseline.json
Uso:    python scripts/05_baseline_logistico.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# -----------------------------------------------------------------------------
# Parámetros
# -----------------------------------------------------------------------------
N_COHORTES_TEST = 2        # últimas N cohortes completas como test
MIN_FRECUENCIA_CAT = 0.01  # categorías por debajo del 1% se agrupan en 'OTROS'

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
FEATURES_CAT = [
    "INDUSTRYGROUPTYPE", "STATEORCOUNTRY", "ENTITYTYPE", "REVENUERANGE", "ANTIGUEDAD",
]
TARGET = "FOLLOWON"


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def log_transform(X):
    """log(1+x) para importes con colas largas (definida a nivel de módulo
    para que el pickle del pipeline sea cargable desde la app)."""
    return np.log1p(np.clip(X, a_min=0, a_max=None))


def agrupar_raras(serie: pd.Series, frecuencia_min: float) -> pd.Series:
    """Agrupa categorías poco frecuentes (umbral calculado sobre TRAIN)."""
    frec = serie.value_counts(normalize=True)
    raras = frec[frec < frecuencia_min].index
    return serie.where(~serie.isin(raras), "OTROS")


def construir_pipeline() -> Pipeline:
    from sklearn.preprocessing import FunctionTransformer

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
    return Pipeline([
        ("preproceso", preproceso),
        ("modelo", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


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


# -----------------------------------------------------------------------------
# Entrenamiento y evaluación
# -----------------------------------------------------------------------------
def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")

    # ---- Split temporal ------------------------------------------------------
    cohortes = sorted(panel["cohorte"].unique())
    if len(cohortes) < 3:
        raise SystemExit("Se necesitan >=3 cohortes para un split temporal. "
                         "Ejecuta el panel sobre el dataset completo.")
    cohortes_test = cohortes[-N_COHORTES_TEST:]
    train = panel[~panel["cohorte"].isin(cohortes_test)].copy()
    test = panel[panel["cohorte"].isin(cohortes_test)].copy()
    print(f"Split temporal -> train: cohortes {cohortes[0]}-{cohortes[-N_COHORTES_TEST-1]} "
          f"({len(train):,}) | test: {cohortes_test} ({len(test):,})\n")

    # Agrupación de categorías raras: umbral aprendido en train, aplicado a ambos
    for col in FEATURES_CAT:
        frec = train[col].fillna("DESCONOCIDO").value_counts(normalize=True)
        raras = set(frec[frec < MIN_FRECUENCIA_CAT].index)
        for df in (train, test):
            df[col] = df[col].fillna("DESCONOCIDO")
            df.loc[df[col].isin(raras), col] = "OTROS"

    X_cols = FEATURES_NUM + FEATURES_BIN + FEATURES_CAT
    pipeline = construir_pipeline()
    pipeline.fit(train[X_cols], train[TARGET])

    # ---- Evaluación ----------------------------------------------------------
    print("Resultados:")
    metricas = {
        "train": evaluar(train[TARGET], pipeline.predict_proba(train[X_cols])[:, 1], "train"),
        "test": evaluar(test[TARGET], pipeline.predict_proba(test[X_cols])[:, 1], "test "),
    }

    print("\nTest por cohorte (estabilidad temporal):")
    metricas["test_por_cohorte"] = {}
    for c in cohortes_test:
        sub = test[test["cohorte"] == c]
        if sub[TARGET].nunique() > 1:
            metricas["test_por_cohorte"][str(c)] = evaluar(
                sub[TARGET], pipeline.predict_proba(sub[X_cols])[:, 1], f"  {c}")

    # ---- Coeficientes: la interpretabilidad del baseline ---------------------
    nombres = pipeline.named_steps["preproceso"].get_feature_names_out()
    coefs = pd.Series(pipeline.named_steps["modelo"].coef_[0], index=nombres)
    print("\nTop 10 coeficientes |positivos| (favorecen follow-on):")
    print(coefs.sort_values(ascending=False).head(10).round(3).to_string())
    print("\nTop 10 coeficientes negativos:")
    print(coefs.sort_values().head(10).round(3).to_string())

    # ---- Persistencia ---------------------------------------------------------
    with open(MODELS_DIR / "baseline_logistico.pkl", "wb") as fh:
        pickle.dump({"pipeline": pipeline, "features": X_cols,
                     "cohortes_test": [int(c) for c in cohortes_test]}, fh)
    with open(MODELS_DIR / "metricas_baseline.json", "w") as fh:
        json.dump(metricas, fh, indent=2)
    print(f"\nGuardado: {MODELS_DIR/'baseline_logistico.pkl'} y metricas_baseline.json")


if __name__ == "__main__":
    main()

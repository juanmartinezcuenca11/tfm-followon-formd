"""
06_catboost.py
Segundo modelo: CatBoost sobre el mismo panel y el MISMO split temporal que el
baseline, para una comparación justa.

Diseño:
  - Categóricas en NATIVO (sin one-hot): es la ventaja de CatBoost frente a la
    logística, sobre todo con STATEORCOUNTRY e INDUSTRYGROUPTYPE (muchos niveles).
  - Split temporal idéntico al 05: train = cohortes antiguas, validación = última
    cohorte de train (para early stopping), test = 2 cohortes más recientes.
  - Rebalanceo con auto_class_weights='Balanced' (equivalente al class_weight
    del baseline) para que la comparación sea manzana con manzana.
  - Mismas métricas (ROC-AUC, PR-AUC, Brier) y comparación lado a lado.

Salida: models/catboost.cbm, models/metricas_catboost.json, models/comparacion.json
Uso:    python scripts/06_catboost.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# -----------------------------------------------------------------------------
# Parámetros
# -----------------------------------------------------------------------------
N_COHORTES_TEST = 2   # mismas que el baseline: las 2 cohortes más recientes

# Mismas familias de features que el baseline. CatBoost no necesita el ratio
# ni transformaciones: se le pasan crudas y él las trata.
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
    "INDUSTRYGROUPTYPE", "STATEORCOUNTRY", "ENTITYTYPE", "REVENUERANGE","ANTIGUEDAD",
]
TARGET = "FOLLOWON"


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

    X_cols = FEATURES_NUM + FEATURES_BIN + FEATURES_CAT

    # CatBoost exige que las categóricas no tengan NaN: las pasamos a texto y
    # rellenamos los vacíos con una etiqueta explícita.
    for col in FEATURES_CAT:
        panel[col] = panel[col].fillna("DESCONOCIDO").astype(str)

    # ---- Split temporal idéntico al baseline --------------------------------
    cohortes = sorted(panel["cohorte"].unique())
    cohortes_test = cohortes[-N_COHORTES_TEST:]
    cohorte_val = cohortes[-N_COHORTES_TEST - 1]  # última de train = validación

    es_test = panel["cohorte"].isin(cohortes_test)
    es_val = panel["cohorte"].eq(cohorte_val)
    train = panel[~es_test & ~es_val]
    val = panel[es_val]
    test = panel[es_test]
    print(f"Split temporal -> train: {cohortes[0]}-{cohorte_val - 1} ({len(train):,}) | "
          f"val: {cohorte_val} ({len(val):,}) | test: {cohortes_test} ({len(test):,})\n")

    idx_cat = [X_cols.index(c) for c in FEATURES_CAT]
    pool_train = Pool(train[X_cols], train[TARGET], cat_features=idx_cat)
    pool_val = Pool(val[X_cols], val[TARGET], cat_features=idx_cat)
    pool_test = Pool(test[X_cols], test[TARGET], cat_features=idx_cat)

    # ---- Modelo --------------------------------------------------------------
    modelo = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",   # equivalente al class_weight del baseline
        eval_metric="AUC",
        random_seed=42,
        early_stopping_rounds=100,        # para si val AUC no mejora en 100 rondas
        verbose=200,
    )
    modelo.fit(pool_train, eval_set=pool_val)
    print(f"\nMejor iteración (early stopping): {modelo.get_best_iteration()}\n")

    # ---- Evaluación ----------------------------------------------------------
    print("Resultados CatBoost:")
    metricas = {
        "train": evaluar(train[TARGET], modelo.predict_proba(pool_train)[:, 1], "train"),
        "test": evaluar(test[TARGET], modelo.predict_proba(pool_test)[:, 1], "test "),
    }
    print("\nTest por cohorte:")
    metricas["test_por_cohorte"] = {}
    for c in cohortes_test:
        sub = test[test["cohorte"] == c]
        prob = modelo.predict_proba(Pool(sub[X_cols], cat_features=idx_cat))[:, 1]
        metricas["test_por_cohorte"][str(c)] = evaluar(sub[TARGET], prob, f"  {c}")

    # ---- Importancia de variables -------------------------------------------
    imp = pd.Series(modelo.get_feature_importance(pool_train), index=X_cols)
    print("\nImportancia de variables (top 15):")
    print(imp.sort_values(ascending=False).head(15).round(2).to_string())

    # ---- Comparación formal contra el baseline ------------------------------
    comparacion = {"catboost_test": metricas["test"]}
    ruta_base = MODELS_DIR / "metricas_baseline.json"
    if ruta_base.exists():
        base = json.loads(ruta_base.read_text())["test"]
        comparacion["baseline_test"] = base
        print("\n" + "=" * 52)
        print("COMPARACIÓN EN TEST (2021-2022)")
        print("=" * 52)
        print(f"{'Métrica':<12}{'Baseline':>12}{'CatBoost':>12}{'Δ':>12}")
        for met in ["roc_auc", "pr_auc", "brier"]:
            b, c = base[met], metricas["test"][met]
            signo = "+" if c - b >= 0 else ""
            print(f"{met:<12}{b:>12.3f}{c:>12.3f}{signo + f'{c - b:.3f}':>12}")
        print("(en Brier, más bajo es mejor)")
    else:
        print("\n(No encuentro metricas_baseline.json; ejecuta antes el 05 "
              "para la comparación)")

    # ---- Persistencia --------------------------------------------------------
    modelo.save_model(str(MODELS_DIR / "catboost.cbm"))
    (MODELS_DIR / "metricas_catboost.json").write_text(json.dumps(metricas, indent=2))
    (MODELS_DIR / "comparacion.json").write_text(json.dumps(comparacion, indent=2))
    print(f"\nGuardado: catboost.cbm, metricas_catboost.json, comparacion.json")


if __name__ == "__main__":
    main()

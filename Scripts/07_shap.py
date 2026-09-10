"""
07_shap.py
Interpretabilidad del modelo CatBoost mediante SHAP.

Pasa de "qué variables importan" (importancia global) a "en qué DIRECCIÓN y con
qué FORMA influye cada variable", y a "por qué ESTA ronda concreta tendrá o no
follow-on" (base del deal memo de la app).

Genera:
  - reports/shap_resumen.png       : beeswarm global (dirección de cada feature)
  - reports/shap_barras.png        : importancia media |SHAP|
  - reports/shap_ratio.png         : efecto de RATIO_COLOCADO
  - reports/shap_casos.csv         : descomposición SHAP de ejemplos concretos
  - reports/shap_values.parquet    : valores SHAP del test (para la app)

Uso: python scripts/07_shap.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin ventana gráfica: guardamos a fichero
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

N_COHORTES_TEST = 2

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
FEATURES_CAT = ["INDUSTRYGROUPTYPE", "STATEORCOUNTRY", "ENTITYTYPE", "REVENUERANGE", "ANTIGUEDAD"]
TARGET = "FOLLOWON"


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    X_cols = FEATURES_NUM + FEATURES_BIN + FEATURES_CAT
    for col in FEATURES_CAT:
        panel[col] = panel[col].fillna("DESCONOCIDO").astype(str)

    # Mismo split: evaluamos SHAP sobre el test (datos no vistos)
    cohortes = sorted(panel["cohorte"].unique())
    test = panel[panel["cohorte"].isin(cohortes[-N_COHORTES_TEST:])].copy()

    modelo = CatBoostClassifier()
    modelo.load_model(str(MODELS_DIR / "catboost.cbm"))

    idx_cat = [X_cols.index(c) for c in FEATURES_CAT]
    pool_test = Pool(test[X_cols], cat_features=idx_cat)

    # ---- Cálculo de valores SHAP --------------------------------------------
    # CatBoost calcula SHAP de forma exacta y eficiente (TreeSHAP integrado).
    print("Calculando valores SHAP sobre el test...")
    shap_raw = modelo.get_feature_importance(pool_test, type="ShapValues")
    # La última columna es el valor base (esperado); el resto, una por feature
    base_value = shap_raw[0, -1]
    shap_values = shap_raw[:, :-1]
    print(f"Valor base (log-odds medio): {base_value:.3f}")

    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(len(test), base_value),
        data=test[X_cols].values,
        feature_names=X_cols,
    )

    # ---- Gráfico 1: beeswarm global (dirección de cada variable) ------------
    plt.figure()
    shap.summary_plot(shap_values, test[X_cols], feature_names=X_cols, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_resumen.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ---- Gráfico 2: importancia media |SHAP| --------------------------------
    plt.figure()
    shap.summary_plot(shap_values, test[X_cols], feature_names=X_cols,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_barras.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ---- Gráficos 3-4: efecto (forma) de dos variables numéricas clave ------
    for feat, fichero in [("RATIO_COLOCADO", "shap_ratio.png")]:
        j = X_cols.index(feat)
        plt.figure(figsize=(7, 5))
        plt.scatter(test[feat].values, shap_values[:, j], s=6, alpha=0.3)
        plt.axhline(0, color="grey", lw=0.8)
        plt.xlabel(feat)
        plt.ylabel(f"Impacto SHAP en log-odds de follow-on")
        plt.title(f"Efecto de {feat}")
        if feat == "RATIO_COLOCADO":
            plt.xlim(0, 1.5)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / fichero, dpi=140)
        plt.close()

    # ---- Casos concretos: descomposición para el deal memo ------------------
    prob = modelo.predict_proba(pool_test)[:, 1]
    test = test.assign(prob=prob)
    ejemplos = pd.concat([
        test.nlargest(3, "prob"),   # 3 con mayor probabilidad
        test.nsmallest(3, "prob"),  # 3 con menor probabilidad
    ])
    filas = []
    for pos, (_, row) in zip(
        [test.index.get_loc(i) for i in ejemplos.index], ejemplos.iterrows()
    ):
        contrib = pd.Series(shap_values[pos], index=X_cols)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)
        filas.append({
            "ENTITYNAME": row.get("ENTITYNAME", ""),
            "sector": row["INDUSTRYGROUPTYPE"],
            "estado": row["STATEORCOUNTRY"],
            "prob_followon": round(row["prob"], 3),
            "followon_real": int(row[TARGET]),
            "top_factores": "; ".join(f"{k}={v:+.2f}" for k, v in top.items()),
        })
    pd.DataFrame(filas).to_csv(REPORTS_DIR / "shap_casos.csv", index=False)

    # ---- Valores SHAP completos para la app ---------------------------------
    df_shap = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in X_cols])
    df_shap["base_value"] = base_value
    df_shap["prob"] = prob
    df_shap.to_parquet(REPORTS_DIR / "shap_values.parquet", index=False)

    print("\nGuardado en reports/:")
    print("  shap_resumen.png, shap_barras.png, shap_ratio.png")
    print("  shap_casos.csv (ejemplos para el deal memo)")
    print("  shap_values.parquet (para la app)")
    print("\nEjemplos de descomposición (mira reports/shap_casos.csv):")
    print(pd.DataFrame(filas)[["sector", "estado", "prob_followon",
                               "followon_real"]].to_string(index=False))


if __name__ == "__main__":
    main()

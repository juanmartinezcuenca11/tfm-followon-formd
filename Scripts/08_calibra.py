"""
08_calibra.py
Calibra las probabilidades del CatBoost.

Por qué: el modelo se entrenó con auto_class_weights='Balanced', lo que mejora
el ranking pero INFLA las probabilidades (un 0.9 del modelo no significa 90% de
probabilidad real). Para la app, que muestra una probabilidad al usuario,
necesitamos que un 0.2 signifique de verdad "1 de cada 5".

Método: calibración isotónica sobre una partición de validación temporal que el
modelo no usó para entrenar (la cohorte más reciente de train). Se evalúa en test
comparando Brier y curva de fiabilidad antes/después.

Salida: models/calibrador.pkl, reports/calibracion.png
Uso:    python scripts/08_calibra.py
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

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
FEATURES_CAT = ["INDUSTRYGROUPTYPE", "STATEORCOUNTRY", "ENTITYTYPE",
                "REVENUERANGE", "ANTIGUEDAD"]
TARGET = "FOLLOWON"


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    X_cols = FEATURES_NUM + FEATURES_BIN + FEATURES_CAT
    for col in FEATURES_CAT:
        panel[col] = panel[col].fillna("DESCONOCIDO").astype(str)

    cohortes = sorted(panel["cohorte"].unique())
    cohortes_test = cohortes[-N_COHORTES_TEST:]
    cohorte_cal = cohortes[-N_COHORTES_TEST - 1]   # cohorte de calibración

    cal = panel[panel["cohorte"].eq(cohorte_cal)]
    test = panel[panel["cohorte"].isin(cohortes_test)]

    modelo = CatBoostClassifier()
    modelo.load_model(str(MODELS_DIR / "catboost.cbm"))
    idx_cat = [X_cols.index(c) for c in FEATURES_CAT]

    def prob_cruda(df):
        return modelo.predict_proba(Pool(df[X_cols], cat_features=idx_cat))[:, 1]

    p_cal = prob_cruda(cal)
    p_test = prob_cruda(test)

    # ---- Ajuste del calibrador isotónico sobre la cohorte de calibración ----
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, cal[TARGET].values)

    p_test_calibrado = iso.predict(p_test)

    # ---- Comparación antes/después ------------------------------------------
    brier_antes = brier_score_loss(test[TARGET], p_test)
    brier_despues = brier_score_loss(test[TARGET], p_test_calibrado)
    print(f"Cohorte de calibración: {cohorte_cal} | test: {list(cohortes_test)}")
    print(f"Tasa real de follow-on en test: {test[TARGET].mean():.1%}")
    print(f"Prob. media CRUDA (inflada):     {p_test.mean():.1%}")
    print(f"Prob. media CALIBRADA:           {p_test_calibrado.mean():.1%}")
    print(f"\nBrier antes:   {brier_antes:.4f}")
    print(f"Brier después: {brier_despues:.4f}  "
          f"({'mejora' if brier_despues < brier_antes else 'empeora'})")

    # ---- Curva de fiabilidad -------------------------------------------------
    plt.figure(figsize=(6, 6))
    for p, etiqueta in [(p_test, "Cruda"), (p_test_calibrado, "Calibrada")]:
        frac_pos, media_pred = calibration_curve(test[TARGET], p, n_bins=10)
        plt.plot(media_pred, frac_pos, "o-", label=etiqueta)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfecta")
    plt.xlabel("Probabilidad predicha")
    plt.ylabel("Frecuencia real observada")
    plt.title("Curva de fiabilidad (calibración)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "calibracion.png", dpi=140)
    plt.close()

    # ---- Persistencia --------------------------------------------------------
    with open(MODELS_DIR / "calibrador.pkl", "wb") as fh:
        pickle.dump(iso, fh)
    print(f"\nGuardado: models/calibrador.pkl, reports/calibracion.png")


if __name__ == "__main__":
    main()

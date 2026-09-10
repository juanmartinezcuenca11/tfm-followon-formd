"""
app.py — Aplicación Flask de scoring de follow-on funding.

Productiviza el modelo CatBoost siguiendo el enfoque del módulo de
productivización: expone el modelo tras un endpoint que recibe las
características de una ronda y devuelve:
  - probabilidad CALIBRADA de follow-on,
  - los factores (SHAP) que más empujan la predicción,
  - un "deal memo" en lenguaje natural generado por plantilla.

Ejecutar:  python app/app.py   (luego abrir http://127.0.0.1:5000)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# -----------------------------------------------------------------------------
# Definición de features (idéntica a la del entrenamiento)
# -----------------------------------------------------------------------------
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
X_COLS = FEATURES_NUM + FEATURES_BIN + FEATURES_CAT
IDX_CAT = [X_COLS.index(c) for c in FEATURES_CAT]

# Etiquetas legibles para el memo
NOMBRES = {
    "INDUSTRYGROUPTYPE": "sector", "STATEORCOUNTRY": "estado/país",
    "ENTITYTYPE": "forma jurídica", "REVENUERANGE": "tramo de ingresos",
    "ANTIGUEDAD": "antigüedad", "RATIO_COLOCADO": "ratio colocado/objetivo",
    "TOTALAMOUNTSOLD_NUM": "importe colocado", "N_DIRECTIVOS": "tamaño del equipo",
    "TOTALNUMBERALREADYINVESTED_NUM": "nº de inversores",
    "TOTALOFFERINGAMOUNT_NUM": "importe objetivo",
    "REVELA_INGRESOS": "declara ingresos", "PAGO_COMISIONES": "paga comisiones",
    "ISEQUITYTYPE_BIN": "equity", "ISDEBTTYPE_BIN": "deuda",
    "ISSECURITYTOBEACQUIREDTYPE_BIN": "convertible/SAFE",
    "ISOPTIONTOACQUIRETYPE_BIN": "opciones",
    "MINIMUMINVESTMENTACCEPTED_NUM": "inversión mínima",
    "OBJETIVO_INDEFINIDO": "objetivo indefinido",
}

# -----------------------------------------------------------------------------
# Carga de artefactos (una sola vez, al arrancar)
# -----------------------------------------------------------------------------
modelo = CatBoostClassifier()
modelo.load_model(str(MODELS_DIR / "catboost.cbm"))
with open(MODELS_DIR / "calibrador.pkl", "rb") as fh:
    calibrador = pickle.load(fh)

# Opciones de los desplegables: valores reales del panel (los que el modelo vio)
panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
OPCIONES = {
    c: sorted(panel[c].fillna("DESCONOCIDO").astype(str).value_counts().index.tolist())
    for c in FEATURES_CAT
}

app = Flask(__name__)


def construir_fila(form) -> pd.DataFrame:
    """Traduce el formulario a la fila de features que espera el modelo."""
    objetivo = float(form.get("objetivo") or 0)
    vendido = float(form.get("vendido") or 0)
    ratio = vendido / objetivo if objetivo > 0 else np.nan

    fila = {
        "INDUSTRYGROUPTYPE": form.get("sector", "DESCONOCIDO"),
        "STATEORCOUNTRY": form.get("estado", "DESCONOCIDO"),
        "ENTITYTYPE": form.get("entitytype", "DESCONOCIDO"),
        "REVENUERANGE": form.get("revenue", "DESCONOCIDO"),
        "ANTIGUEDAD": form.get("antiguedad", "DESCONOCIDO"),
        "TOTALOFFERINGAMOUNT_NUM": objetivo,
        "TOTALAMOUNTSOLD_NUM": vendido,
        "RATIO_COLOCADO": ratio,
        "MINIMUMINVESTMENTACCEPTED_NUM": float(form.get("min_inv") or 0),
        "TOTALNUMBERALREADYINVESTED_NUM": float(form.get("n_inversores") or 0),
        "N_DIRECTIVOS": float(form.get("n_directivos") or 0),
        "OBJETIVO_INDEFINIDO": 0,
        "PAGO_COMISIONES": int(form.get("comisiones") == "on"),
        "REVELA_INGRESOS": int(form.get("revenue", "") not in ("", "Decline to Disclose")),
        "ISEQUITYTYPE_BIN": int(form.get("equity") == "on"),
        "ISDEBTTYPE_BIN": int(form.get("deuda") == "on"),
        "ISSECURITYTOBEACQUIREDTYPE_BIN": int(form.get("convertible") == "on"),
        "ISOPTIONTOACQUIRETYPE_BIN": int(form.get("opciones") == "on"),
    }
    return pd.DataFrame([fila])[X_COLS]


def explicar(fila: pd.DataFrame):
    """Devuelve los factores SHAP ordenados por impacto para esta predicción."""
    pool = Pool(fila, cat_features=IDX_CAT)
    shap_raw = modelo.get_feature_importance(pool, type="ShapValues")[0]
    contrib = pd.Series(shap_raw[:-1], index=X_COLS)
    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    positivos = [(NOMBRES.get(k, k), v) for k, v in contrib.items() if v > 0.02][:3]
    negativos = [(NOMBRES.get(k, k), v) for k, v in contrib.items() if v < -0.02][:3]
    return positivos, negativos


def redactar_memo(prob, positivos, negativos, form) -> str:
    """Deal memo por plantilla: teje la predicción y sus factores en prosa."""
    if prob >= 0.30:
        nivel = "elevada"
    elif prob >= 0.18:
        nivel = "moderada"
    else:
        nivel = "baja"

    partes = [
        f"Ronda en el sector {form.get('sector','?')} ({form.get('estado','?')}), "
        f"constituida como {form.get('entitytype','?')}. "
        f"El modelo estima una probabilidad {nivel} de financiación posterior "
        f"(follow-on) del {prob:.0%} en un horizonte de 3 años."
    ]
    if positivos:
        favor = ", ".join(n for n, _ in positivos)
        partes.append(f"Factores que elevan la valoración: {favor}.")
    if negativos:
        contra = ", ".join(n for n, _ in negativos)
        partes.append(f"Factores que la moderan: {contra}.")
    partes.append(
        "Nota: estimación basada en patrones históricos de rondas Form D; "
        "no considera factores cualitativos del equipo ni del producto."
    )
    return " ".join(partes)


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    if request.method == "POST":
        fila = construir_fila(request.form)
        p_cruda = modelo.predict_proba(Pool(fila, cat_features=IDX_CAT))[0, 1]
        p_cal = float(calibrador.predict([p_cruda])[0])
        positivos, negativos = explicar(fila)
        resultado = {
            "prob": p_cal,
            "prob_pct": f"{p_cal:.0%}",
            "positivos": positivos,
            "negativos": negativos,
            "memo": redactar_memo(p_cal, positivos, negativos, request.form),
        }
    return render_template("index.html", opciones=OPCIONES, resultado=resultado)


if __name__ == "__main__":
    app.run(debug=True)

"""
09_graficos_eda.py
Genera las figuras del capítulo 4 (análisis exploratorio) a partir del panel
y de las tablas consolidadas.

Salida en reports/:
  eda_volumen_anual.png       volumen de rondas por año (2009-2026)
  eda_poblacion.png           empresas operativas vs vehículos de inversión
  eda_followon_cohorte.png    tasa de follow-on por cohorte (2011-2022)
  eda_estados.png             top estados por nº de primeras rondas
  eda_sectores.png            top sectores por nº de primeras rondas
  eda_importes.png            distribución (log) del importe colocado

Uso: python scripts/09_graficos_eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

AZUL = "#1F497D"
AZUL2 = "#1F78B4"
GRIS = "#888888"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def guardar(nombre):
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / nombre, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {nombre}")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")

    # ---- 1. Volumen de registros por año ------------------------------------
    sub = pd.read_parquet(INTERIM_DIR / "submission.parquet")
    fechas = pd.to_datetime(sub["FILING_DATE"], format="mixed",
                            dayfirst=True, errors="coerce")
    por_anio = fechas.dt.year.value_counts().sort_index()
    por_anio = por_anio[(por_anio.index >= 2009) & (por_anio.index <= 2026)]

    plt.figure(figsize=(9, 4.5))
    plt.bar(por_anio.index, por_anio.values, color=AZUL2)
    plt.title("Volumen de registros Form D por año (2009-2026)")
    plt.ylabel("Nº de registros")
    plt.xlabel("Año")
    plt.axvline(2021, color=GRIS, ls="--", lw=1)
    plt.text(2021, por_anio.max() * 0.95, " pico 2021", color=GRIS, fontsize=9)
    guardar("eda_volumen_anual.png")

    # ---- 2. Población: empresas operativas vs vehículos ---------------------
    off = pd.read_parquet(INTERIM_DIR / "offering.parquet")
    INDUSTRIAS_FONDO = ["Pooled Investment Fund", "Investing", "Investment Banking",
                        "REITS and Finance", "Other Banking and Financial Services",
                        "Commercial Banking", "Insurance"]
    fondo = (off["ISPOOLEDINVESTMENTFUNDTYPE"].fillna("").str.lower().eq("true")
             | off["INDUSTRYGROUPTYPE"].fillna("").isin(INDUSTRIAS_FONDO))
    n_fondo = int(fondo.sum())
    n_op = len(off) - n_fondo

    plt.figure(figsize=(6, 4))
    barras = plt.bar(["Vehículos de\ninversión", "Empresas\noperativas"],
                     [n_fondo, n_op], color=[GRIS, AZUL])
    plt.title("Composición de la población de registros")
    plt.ylabel("Nº de registros")
    for b, v in zip(barras, [n_fondo, n_op]):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v/1e3:.0f}k\n({v/len(off):.0%})",
                 ha="center", va="bottom", fontsize=10)
    plt.ylim(0, max(n_fondo, n_op) * 1.15)
    guardar("eda_poblacion.png")

    # ---- 3. Tasa de follow-on por cohorte -----------------------------------
    tasa = panel.groupby("cohorte")["FOLLOWON"].mean()
    plt.figure(figsize=(9, 4.5))
    plt.plot(tasa.index, tasa.values, "o-", color=AZUL, lw=2)
    plt.title("Tasa de follow-on por cohorte de entrada")
    plt.ylabel("Proporción con follow-on")
    plt.xlabel("Año de la primera ronda (cohorte)")
    plt.ylim(0, tasa.max() * 1.2)
    for x, y in zip(tasa.index, tasa.values):
        plt.text(x, y + 0.008, f"{y:.0%}", ha="center", fontsize=8, color=AZUL)
    guardar("eda_followon_cohorte.png")

    # ---- 4. Top estados ------------------------------------------------------
    estados = panel["STATEORCOUNTRY"].value_counts().head(12)
    plt.figure(figsize=(9, 4.5))
    plt.barh(estados.index[::-1], estados.values[::-1], color=AZUL2)
    plt.title("Primeras rondas por estado / país (top 12)")
    plt.xlabel("Nº de primeras rondas")
    guardar("eda_estados.png")

    # ---- 5. Top sectores -----------------------------------------------------
    sect = panel["INDUSTRYGROUPTYPE"].value_counts().head(12)
    plt.figure(figsize=(9, 4.5))
    plt.barh(sect.index[::-1], sect.values[::-1], color=AZUL2)
    plt.title("Primeras rondas por sector (top 12)")
    plt.xlabel("Nº de primeras rondas")
    guardar("eda_sectores.png")

    # ---- 6. Distribución de importes (escala log) ---------------------------
    amt = pd.to_numeric(panel["TOTALAMOUNTSOLD_NUM"], errors="coerce")
    amt = amt[amt > 0]
    plt.figure(figsize=(9, 4.5))
    plt.hist(np.log10(amt), bins=50, color=AZUL2, edgecolor="white")
    mediana = amt.median()
    plt.axvline(np.log10(mediana), color=AZUL, ls="--", lw=1.5)
    plt.text(np.log10(mediana), plt.ylim()[1] * 0.9,
             f" mediana ${mediana/1e3:.0f}k", color=AZUL, fontsize=9)
    plt.title("Distribución del importe colocado en la primera ronda")
    plt.xlabel("Importe colocado (escala log10, $)")
    plt.ylabel("Nº de rondas")
    guardar("eda_importes.png")

    print("\nHecho. 6 figuras en reports/")


if __name__ == "__main__":
    main()

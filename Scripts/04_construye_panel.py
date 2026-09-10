"""
04_construye_panel.py
Construye el panel de modelización a partir de los parquet consolidados:

  1. Población: filings de empresas operativas (excluye fondos/vehículos).
  2. Deduplicación: las enmiendas (D/A) se resuelven a su filing raíz y se
     usan solo para actualizar importes; cada ronda cuenta una vez.
  3. Primera ronda por CIK = primera oferta nueva (tipo D) observada.
  4. Variable objetivo FOLLOWON: ¿existe otra oferta nueva del mismo CIK
     entre GAP_MIN_DIAS y VENTANA_ANIOS tras la primera ronda?
  5. Censura: se excluyen del panel las primeras rondas sin ventana completa
     de observación (borde derecho) y las cohortes anteriores a COHORTE_MIN
     (borde izquierdo, periodo de adopción del filing electrónico).
  6. Features de la ronda inicial + tamaño del equipo directivo.

Salida: data/processed/panel.parquet  (+ resumen por consola)
Uso:    python scripts/04_construye_panel.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# -----------------------------------------------------------------------------
# Parámetros del panel (decisiones metodológicas; se justifican en la memoria)
# -----------------------------------------------------------------------------
VENTANA_ANIOS = 3      # ventana para observar el follow-on
GAP_MIN_DIAS = 90      # separación mínima para considerar "otra" ronda
COHORTE_MIN = 2011     # primera cohorte de entrada (evita el sesgo pre-2009/2010)
COHORTE_MAX = 2022   # última cohorte con ventana de follow-on de 3 años observable

INDUSTRIAS_EXCLUIDAS = [
    "Pooled Investment Fund", "Investing", "Investment Banking",
    "REITS and Finance", "Other Banking and Financial Services",
    "Commercial Banking", "Insurance",
]


# -----------------------------------------------------------------------------
# Carga y filtros de población
# -----------------------------------------------------------------------------
def cargar_tablas():
    sub = pd.read_parquet(INTERIM_DIR / "submission.parquet")
    off = pd.read_parquet(INTERIM_DIR / "offering.parquet")
    iss = pd.read_parquet(INTERIM_DIR / "issuers.parquet")
    rel = pd.read_parquet(INTERIM_DIR / "relatedpersons.parquet")
    iss = iss[iss["IS_PRIMARYISSUER_FLAG"].str.upper().eq("YES")].copy()
    return sub, off, iss, rel


def marcar_fondos(off: pd.DataFrame) -> pd.Series:
    flag_pooled = off["ISPOOLEDINVESTMENTFUNDTYPE"].fillna("").str.lower().eq("true")
    flag_40act = off.get("IS40ACT", pd.Series("", index=off.index)).fillna("").str.lower().eq("true")
    industria = off["INDUSTRYGROUPTYPE"].fillna("").isin(INDUSTRIAS_EXCLUIDAS)
    return flag_pooled | flag_40act | industria


# -----------------------------------------------------------------------------
# Resolución de enmiendas: cada D/A apunta (en cadena) a un filing raíz
# -----------------------------------------------------------------------------
def resolver_raiz(off: pd.DataFrame) -> pd.Series:
    """Devuelve, para cada ACCESSIONNUMBER, el accession del filing raíz."""
    mapa = (
        off.loc[off["PREVIOUSACCESSIONNUMBER"].notna() & off["PREVIOUSACCESSIONNUMBER"].ne(""),
                ["ACCESSIONNUMBER", "PREVIOUSACCESSIONNUMBER"]]
        .set_index("ACCESSIONNUMBER")["PREVIOUSACCESSIONNUMBER"]
        .to_dict()
    )
    raiz = {}
    for acc in off["ACCESSIONNUMBER"]:
        actual, vistos = acc, set()
        while actual in mapa and actual not in vistos:  # protege de ciclos
            vistos.add(actual)
            actual = mapa[actual]
        raiz[acc] = actual
    return off["ACCESSIONNUMBER"].map(raiz)


# -----------------------------------------------------------------------------
# Construcción del panel
# -----------------------------------------------------------------------------
def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sub, off, iss, rel = cargar_tablas()

    n0 = len(off)
    fondos = marcar_fondos(off)
    off = off.loc[~fondos].copy()
    print(f"Filtro de población: {n0:,} filings -> {len(off):,} "
          f"(excluidos {fondos.sum():,} de fondos/vehículos)")

    # Base filing-nivel: submission + offering + issuer principal
    base = (
        sub.merge(off, on="ACCESSIONNUMBER")
           .merge(iss[["ACCESSIONNUMBER", "CIK", "ENTITYNAME", "STATEORCOUNTRY",
                       "JURISDICTIONOFINC", "ENTITYTYPE",
                       "YEAROFINC_TIMESPAN_CHOICE", "YEAROFINC_VALUE_ENTERED"]],
                  on="ACCESSIONNUMBER")
    )
    base["fecha_filing"] = pd.to_datetime(base["FILING_DATE"], format="mixed", dayfirst=True, errors="coerce")
    base["fecha_venta"] = pd.to_datetime(base["SALE_DATE"], format="mixed", dayfirst=True, errors="coerce")
    # Fecha económica de la ronda: primera venta si existe; si no, filing
    base["fecha_ronda"] = base["fecha_venta"].fillna(base["fecha_filing"])

    # Rondas únicas: resolvemos cadenas de enmiendas a su raíz y nos quedamos
    # con UNA fila por ronda (la más reciente de la cadena: importes al día,
    # fecha de la ronda = la del filing raíz)
    base["raiz"] = resolver_raiz(base)
    base = base.sort_values("fecha_filing")
    fecha_raiz = base.groupby("raiz")["fecha_ronda"].first()
    rondas = base.drop_duplicates("raiz", keep="last").copy()
    rondas["fecha_ronda"] = rondas["raiz"].map(fecha_raiz)
    print(f"Rondas únicas tras resolver enmiendas: {len(rondas):,}")

    # ---- Primera ronda por empresa y etiqueta de follow-on ------------------
    rondas = rondas.sort_values("fecha_ronda")
    primeras = rondas.drop_duplicates("CIK", keep="first").copy()

    fin_datos = rondas["fecha_ronda"].max()
    ventana = pd.DateOffset(years=VENTANA_ANIOS)

    # Para cada CIK, fechas de todas sus rondas (para buscar la siguiente)
    fechas_por_cik = rondas.groupby("CIK")["fecha_ronda"].apply(sorted).to_dict()

    def etiqueta(row):
        t0 = row["fecha_ronda"]
        limite = t0 + ventana
        if limite > fin_datos:
            return np.nan  # censurada: sin ventana completa de observación
        for t in fechas_por_cik.get(row["CIK"], []):
            if t0 + pd.Timedelta(days=GAP_MIN_DIAS) <= t <= limite:
                return 1.0
        return 0.0

    primeras["FOLLOWON"] = primeras.apply(etiqueta, axis=1)

    # Cohortes de entrada válidas
    primeras["cohorte"] = primeras["fecha_ronda"].dt.year
    panel = primeras[(primeras["cohorte"] >= COHORTE_MIN) & (primeras["cohorte"] <= COHORTE_MAX)].copy()
    censuradas = panel["FOLLOWON"].isna()
    print(f"Primeras rondas desde {COHORTE_MIN}: {len(panel):,} "
          f"({censuradas.sum():,} censuradas por borde derecho)")
    panel = panel.loc[~censuradas].copy()

    # ---- Features ------------------------------------------------------------
    for col in ["TOTALOFFERINGAMOUNT", "TOTALAMOUNTSOLD", "MINIMUMINVESTMENTACCEPTED",
                "TOTALNUMBERALREADYINVESTED", "SALESCOMM_DOLLARAMOUNT",
                "FINDERSFEE_DOLLARAMOUNT"]:
        panel[col + "_NUM"] = pd.to_numeric(panel[col], errors="coerce")

    # 'Indefinite' en el importe objetivo es informativo por sí mismo
    panel["OBJETIVO_INDEFINIDO"] = panel["TOTALOFFERINGAMOUNT"].astype(str).str.contains(
        "Indefinite", case=False, na=False).astype(int)
    # Ratio de demanda: cuánto del objetivo se colocó (solo si objetivo definido)
    panel["RATIO_COLOCADO"] = np.where(
        panel["TOTALOFFERINGAMOUNT_NUM"] > 0,
        panel["TOTALAMOUNTSOLD_NUM"] / panel["TOTALOFFERINGAMOUNT_NUM"],
        np.nan,
    )
    panel["PAGO_COMISIONES"] = (
        (panel["SALESCOMM_DOLLARAMOUNT_NUM"].fillna(0) > 0)
        | (panel["FINDERSFEE_DOLLARAMOUNT_NUM"].fillna(0) > 0)
    ).astype(int)
    panel["REVELA_INGRESOS"] = (~panel["REVENUERANGE"].fillna("")
                                .str.contains("Decline", na=False)).astype(int)

    # Edad aproximada de la empresa al levantar la ronda
    panel["ANTIGUEDAD"] = panel["YEAROFINC_TIMESPAN_CHOICE"].fillna("DESCONOCIDO")

    # Tamaño del equipo directivo (personas en RELATEDPERSONS del filing)
    equipo = rel.groupby("ACCESSIONNUMBER").size().rename("N_DIRECTIVOS")
    panel = panel.merge(equipo, on="ACCESSIONNUMBER", how="left")
    panel["N_DIRECTIVOS"] = panel["N_DIRECTIVOS"].fillna(0)

    # Flags de instrumento ya vienen como true/false -> 0/1
    for col in ["ISEQUITYTYPE", "ISDEBTTYPE", "ISSECURITYTOBEACQUIREDTYPE",
                "ISOPTIONTOACQUIRETYPE"]:
        panel[col + "_BIN"] = panel[col].fillna("").str.lower().eq("true").astype(int)

    columnas_panel = [
        "CIK", "ENTITYNAME", "raiz", "fecha_ronda", "cohorte",
        "FOLLOWON",
        "INDUSTRYGROUPTYPE", "STATEORCOUNTRY", "JURISDICTIONOFINC", "ENTITYTYPE",
        "REVENUERANGE", "FEDERALEXEMPTIONS_ITEMS_LIST",
        "TOTALOFFERINGAMOUNT_NUM", "TOTALAMOUNTSOLD_NUM", "RATIO_COLOCADO",
        "OBJETIVO_INDEFINIDO", "MINIMUMINVESTMENTACCEPTED_NUM",
        "TOTALNUMBERALREADYINVESTED_NUM", "PAGO_COMISIONES", "REVELA_INGRESOS",
        "ANTIGUEDAD", "N_DIRECTIVOS",
        "ISEQUITYTYPE_BIN", "ISDEBTTYPE_BIN", "ISSECURITYTOBEACQUIREDTYPE_BIN",
        "ISOPTIONTOACQUIRETYPE_BIN",
    ]
    panel = panel[columnas_panel]

    destino = PROCESSED_DIR / "panel.parquet"
    panel.to_parquet(destino, index=False)

    print(f"\nPanel final: {len(panel):,} primeras rondas -> {destino}")
    print(f"Tasa de follow-on: {panel['FOLLOWON'].mean():.1%}")
    print("\nFollow-on por cohorte:")
    print(panel.groupby("cohorte")["FOLLOWON"].agg(["count", "mean"]).round(3))


if __name__ == "__main__":
    main()

"""
03_perfil_inicial.py
Validación y primer perfil del dataset consolidado. Responde a las preguntas
clave antes de construir el panel:
  - ¿Cuántos filings hay por año? ¿D vs D/A?
  - ¿Qué proporción son fondos de inversión (a excluir)?
  - ¿Cuántas empresas operativas con PRIMER filing hay? (tamaño real del dataset)

Uso: python scripts/03_perfil_inicial.py
"""

from pathlib import Path

import pandas as pd

INTERIM_DIR = Path(__file__).resolve().parents[1] / "data" / "interim"


def cargar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = pd.read_parquet(INTERIM_DIR / "submission.parquet")
    off = pd.read_parquet(INTERIM_DIR / "offering.parquet")
    iss = pd.read_parquet(INTERIM_DIR / "issuers.parquet")
    # Nos quedamos solo con el emisor principal de cada filing
    iss = iss[iss["IS_PRIMARYISSUER_FLAG"].str.upper().eq("YES")]
    return sub, off, iss


def es_fondo(off: pd.DataFrame) -> pd.Series:
    """Filtro de población: filings de vehículos de inversión, no de empresas."""
    flag_pooled = off["ISPOOLEDINVESTMENTFUNDTYPE"].fillna("").str.lower().eq("true")
    industrias_fin = off["INDUSTRYGROUPTYPE"].fillna("").isin(
        ["Pooled Investment Fund", "Investing", "Investment Banking", "REITS and Finance"]
    )
    return flag_pooled | industrias_fin


def main() -> None:
    sub, off, iss = cargar()

    fechas = pd.to_datetime(sub["FILING_DATE"], format="mixed", dayfirst=True, errors="coerce")
    sub["anio"] = fechas.dt.year
    print("Filings por año y tipo (D = nuevo, D/A = enmienda):")
    print(sub.pivot_table(index="anio", columns="SUBMISSIONTYPE",
                          values="ACCESSIONNUMBER", aggfunc="count", fill_value=0))

    # ---- Filtro de población ------------------------------------------------
    fondo = es_fondo(off)
    print(f"\nFilings de fondos/vehículos de inversión: {fondo.sum():,} "
          f"de {len(off):,} ({fondo.mean():.1%})")

    # ---- Panel mínimo: primer filing D por CIK de empresas operativas -------
    base = (
        sub[sub["SUBMISSIONTYPE"].eq("D")]
        .merge(off.loc[~fondo, ["ACCESSIONNUMBER", "INDUSTRYGROUPTYPE",
                                "TOTALAMOUNTSOLD"]], on="ACCESSIONNUMBER")
        .merge(iss[["ACCESSIONNUMBER", "CIK", "ENTITYNAME", "STATEORCOUNTRY"]],
               on="ACCESSIONNUMBER")
    )
    base["fecha"] = pd.to_datetime(base["FILING_DATE"], errors="coerce")
    primeras = base.sort_values("fecha").drop_duplicates("CIK", keep="first")

    print(f"\nEmpresas operativas con primera ronda observada: {primeras.CIK.nunique():,}")
    print("\nPrimeras rondas por año:")
    print(primeras["fecha"].dt.year.value_counts().sort_index())

    amt = pd.to_numeric(primeras["TOTALAMOUNTSOLD"], errors="coerce")
    print(f"\nImporte vendido en primera ronda: mediana {amt.median():,.0f} $ "
          f"| p25 {amt.quantile(.25):,.0f} | p75 {amt.quantile(.75):,.0f}")

    print("\nTop 10 industrias (primeras rondas):")
    print(primeras["INDUSTRYGROUPTYPE"].value_counts().head(10).to_string())

    print("\nTop 10 estados:")
    print(primeras["STATEORCOUNTRY"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()

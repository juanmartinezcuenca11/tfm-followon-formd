"""
02_consolida_formd.py
Lee todos los ZIPs trimestrales de data/raw, identifica cada tabla POR SUS
COLUMNAS (los nombres de fichero dentro de los ZIP no son fiables) y
consolida cada tabla en un único parquet con columna 'quarter' de origen.

Salida: data/interim/{submission,offering,issuers,relatedpersons,recipients,signatures}.parquet
Uso:    python scripts/02_consolida_formd.py
"""

import io
import zipfile
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"

# -----------------------------------------------------------------------------
# Identificación de tablas por columnas "firma" (columnas que solo existen en
# esa tabla). Vimos que los nombres de fichero vienen rotados en los ZIP,
# así que NUNCA confiamos en el nombre: solo en el contenido.
# -----------------------------------------------------------------------------
FIRMAS = {
    "submission": {"FILING_DATE", "SUBMISSIONTYPE", "SCHEMAVERSION"},
    "offering": {"TOTALOFFERINGAMOUNT", "TOTALAMOUNTSOLD", "INDUSTRYGROUPTYPE"},
    "issuers": {"CIK", "ENTITYNAME", "JURISDICTIONOFINC"},
    "relatedpersons": {"RELATIONSHIP_1", "FIRSTNAME", "LASTNAME"},
    "recipients": {"RECIPIENTNAME", "RECIPIENTCRDNUMBER"},
    "signatures": {"NAMEOFSIGNER", "SIGNATURETITLE", "SIGNATUREDATE"},
}


def identificar_tabla(columnas: set) -> str | None:
    """Devuelve el nombre real de la tabla si sus columnas firma están presentes."""
    for tabla, firma in FIRMAS.items():
        if firma.issubset(columnas):
            return tabla
    return None


def leer_zip(path_zip: Path) -> dict[str, pd.DataFrame]:
    """Extrae los TSV de un ZIP trimestral y los devuelve clasificados por tabla."""
    import re
    m = re.search(r'(\d{4}q[1-4])', path_zip.stem)
    quarter = m.group(1) if m else path_zip.stem
    tablas: dict[str, pd.DataFrame] = {}

    with zipfile.ZipFile(path_zip) as zf:
        for nombre in zf.namelist():
            if not nombre.lower().endswith(".tsv"):
                continue
            with zf.open(nombre) as fh:
                # Todo como string: los tipos se fijan después, con criterio,
                # en el script de construcción del panel.
                df = pd.read_csv(
                    io.TextIOWrapper(fh, encoding="latin-1"),
                    sep="\t",
                    dtype=str,
                    on_bad_lines="warn",
                )
            tabla = identificar_tabla(set(df.columns))
            if tabla is None:
                print(f"    AVISO: {nombre} en {quarter} no coincide con ninguna firma")
                continue
            df["quarter"] = quarter
            tablas[tabla] = df

    return tablas


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    zips = sorted(RAW_DIR.glob("*.zip"))
    if not zips:
        raise SystemExit(f"No hay ZIPs en {RAW_DIR}. Ejecuta antes 01_descarga_formd.py")

    acumulado: dict[str, list[pd.DataFrame]] = {t: [] for t in FIRMAS}

    print(f"Procesando {len(zips)} trimestres...\n")
    for path_zip in zips:
        print(f"  {path_zip.name}")
        for tabla, df in leer_zip(path_zip).items():
            acumulado[tabla].append(df)

    print("\nEscribiendo parquet consolidados:")
    for tabla, partes in acumulado.items():
        if not partes:
            print(f"  {tabla}: sin datos")
            continue
        df = pd.concat(partes, ignore_index=True)
        destino = INTERIM_DIR / f"{tabla}.parquet"
        df.to_parquet(destino, index=False)
        print(f"  {tabla}: {len(df):,} filas -> {destino.name}")


if __name__ == "__main__":
    main()

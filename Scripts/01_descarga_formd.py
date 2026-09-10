"""
01_descarga_formd.py
Descarga los Form D Data Sets trimestrales de la SEC (2008Q1 - actualidad).

Salida: data/raw/formd_YYYYqQ.zip
Uso:    python scripts/01_descarga_formd.py
"""

import time
from pathlib import Path

import requests

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
# La SEC exige identificarse en el User-Agent (sin esto devuelve 403).
HEADERS = {"User-Agent": "Juan Martinez juanmartinezcuencav@gmail.com"}

# La nomenclatura de los ficheros ha variado con los años; probamos varios
# patrones por trimestre y nos quedamos con el primero que exista.
URL_PATTERNS = [
    "https://www.sec.gov/files/dera/data/form-d-data-sets/{q}_d.zip",
    "https://www.sec.gov/files/dera/data/form-d-data-sets/{q}.zip",
    "https://www.sec.gov/files/structureddata/data/form-d-data-sets/{q}_d.zip",
    "https://www.sec.gov/files/structureddata/data/form-d-data-sets/{q}.zip",
]

ANIO_INICIO = 2008
ANIO_FIN = 2026          # los trimestres aún no publicados devolverán 404: es normal
PAUSA_SEGUNDOS = 1.0     # cortesía con el servidor de la SEC

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def descargar_trimestre(quarter: str) -> str:
    """Intenta descargar un trimestre probando los distintos patrones de URL.

    Devuelve un string descriptivo del resultado.
    """
    destino = RAW_DIR / f"formd_{quarter}.zip"
    if destino.exists():
        return f"ya existe ({destino.stat().st_size / 1e6:.1f} MB)"

    for pattern in URL_PATTERNS:
        url = pattern.format(q=quarter)
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
        except requests.RequestException as exc:
            return f"error de red: {exc}"

        if r.status_code == 200 and r.content[:2] == b"PK":  # cabecera ZIP válida
            destino.write_bytes(r.content)
            return f"OK ({len(r.content) / 1e6:.1f} MB)"

    return "no disponible (404 en todos los patrones)"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    quarters = [f"{y}q{q}" for y in range(ANIO_INICIO, ANIO_FIN + 1) for q in range(1, 5)]

    print(f"Descargando {len(quarters)} trimestres en {RAW_DIR}\n")
    for quarter in quarters:
        resultado = descargar_trimestre(quarter)
        print(f"  {quarter}: {resultado}")
        time.sleep(PAUSA_SEGUNDOS)

    n_zips = len(list(RAW_DIR.glob("formd_*.zip")))
    print(f"\nHecho. {n_zips} ZIPs en {RAW_DIR}")


if __name__ == "__main__":
    main()

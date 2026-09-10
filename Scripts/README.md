# TFM — Predicción de follow-on funding (SEC Form D)

## Estructura del proyecto

```
TFM/
├── data/
│   ├── raw/        # ZIPs trimestrales de la SEC (NO se suben a GitHub)
│   ├── interim/    # parquet consolidados por tabla (NO se suben a GitHub)
│   └── processed/  # panel de modelización final
├── scripts/
│   ├── 01_descarga_formd.py     # descarga los ZIPs de la SEC
│   ├── 02_consolida_formd.py    # consolida a un parquet por tabla
│   └── 03_perfil_inicial.py     # validación y primer perfil
├── notebooks/      # EDA exploratorio (opcional)
├── memoria/        # documento del TFM
└── README.md
```

## Puesta en marcha (Windows / VS Code)

```bash
# 1. Crear y activar entorno
python -m venv .venv
.venv\Scripts\activate

# 2. Instalar dependencias
pip install pandas pyarrow requests

# 3. Ejecutar en orden
python scripts/01_descarga_formd.py    # ~5-10 min (73 ZIPs, pausa de cortesía)
python scripts/02_consolida_formd.py   # ~5 min
python scripts/03_perfil_inicial.py    # instantáneo
```

## Notas importantes

- **User-Agent**: la SEC exige identificarse con nombre y email en las
  peticiones; ya está configurado en el script 01.
- **Nombres de fichero rotados**: los TSV dentro de los ZIP de la SEC traen
  nombres que no se corresponden con su contenido. El script 02 identifica
  cada tabla por sus columnas, nunca por el nombre.
- **2008 a 2009Q1 casi vacíos**: el filing electrónico fue voluntario hasta
  marzo de 2009. Es normal; esos trimestres se usan solo como histórico.
- **GitHub**: crear el repo con un `.gitignore` que excluya `data/` y `.venv/`
  (los datos se regeneran con el script 01; el repo lleva solo código).

```gitignore
data/
.venv/
__pycache__/
*.pkl
```

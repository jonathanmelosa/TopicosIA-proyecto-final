#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_cruce_geografico.py  —  PASO 1 del pipeline
===============================================
Construye el dataset de entrenamiento: una fila por imagen de Street View
con su UPZ, indicadores de pobreza y fecha de captura.

Pasos
-----
  1. Lee coordenadas desde los nombres de archivo  lat_lon.jpg.
  2. Une las fechas de captura desde fechas.csv (generado por fetch_fechas.py).
  3. Carga los polígonos UPZ (upz_bogota.geojson).
  4. Spatial join: asigna la UPZ a cada imagen según su coordenada.
  5. Une los indicadores de pobreza desde pobreza_upz.csv.
     Las UPZs compuestas de la encuesta (811, 812) se resuelven con un mapeo
     explícito de códigos del shapefile → código de la encuesta.
  6. Guarda el dataset final en 00_data/processed/imagenes_dataset.csv.
  7. Genera un mapa de cobertura coloreado por pobreza monetaria.

ENTRADAS
--------
  00_data/USME/               → imágenes .jpg  (lat_lon.jpg)
  00_data/USME/fechas.csv     → lat, lon, fecha_descargada, fecha_alternativa
  00_data/raw/upz/            → polígonos UPZ (GeoJSON o shapefile)
  00_data/raw/pobreza_upz.csv → upz_cod, inc_monetaria_pct, inc_ipm_pct,
                                inc_extrema_pct  (generado por 00_descargar_datos.py)

SALIDAS
-------
  00_data/processed/imagenes_dataset.csv
  02_outputs/figuras/cobertura_pobreza.png

USO
---
  python 01_code/01_cruce_geografico.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point

# =============================================================================
# RUTAS
# =============================================================================

_PROYECTO_ROOT = Path(__file__).resolve().parent.parent

IMG_DIR       = _PROYECTO_ROOT / "00_data" / "USME"
FECHAS_CSV    = IMG_DIR / "fechas.csv"
UPZ_DIR       = _PROYECTO_ROOT / "00_data" / "raw" / "upz"
POBREZA_CSV   = _PROYECTO_ROOT / "00_data" / "raw" / "pobreza_upz.csv"
PROCESSED_DIR = _PROYECTO_ROOT / "00_data" / "processed"
FIGURAS_DIR   = _PROYECTO_ROOT / "02_outputs" / "figuras"

SALIDA_CSV  = PROCESSED_DIR / "imagenes_dataset.csv"
SALIDA_MAPA = FIGURAS_DIR   / "cobertura_pobreza.png"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

# Mapeo shapefile → encuesta para UPZs que la EM 2021 agrupa en estratos
# compuestos (códigos 800+). Cada código individual del shapefile se traduce
# al código compuesto correspondiente en pobreza_upz.csv.
# Verificado contra NOMBRE_UPZ_GRUPO en em2021_merged.csv:
#   811 = "USME: Alfonso López + Ciudad Usme"    → shp 59 y 61
#   812 = "USME: Parque Entrenubes + Danubio"    → shp 60 y 56
MAPA_SHP_A_ENCUESTA: dict[str, str] = {
    "59": "811",
    "61": "811",
    "60": "812",
    "56": "812",
}

# =============================================================================
# PASO 1: Extraer coordenadas de los nombres de archivo
# =============================================================================

print("=" * 60)
print("  PASO 1 — Extracción de coordenadas de nombres de archivo")
print("=" * 60)

jpgs = sorted(IMG_DIR.glob("*.jpg"))
if not jpgs:
    print(f"\n[ERROR] No se encontraron imágenes .jpg en {IMG_DIR}")
    print("  → Corre 01_code/00_descarga/extract_imagenes.py primero.")
    sys.exit(1)

print(f"  Imágenes encontradas: {len(jpgs):,}")

registros = []
errores   = 0
for p in jpgs:
    stem = p.stem
    try:
        idx = stem.index("_-") if "_-" in stem else stem.rindex("_")
        lat = float(stem[:idx])
        lon = float(stem[idx + 1:])
        registros.append({"ruta_imagen": str(p), "lat": lat, "lon": lon})
    except (ValueError, IndexError):
        errores += 1

if errores:
    print(f"  [AVISO] {errores} archivos con nombre no parseable (ignorados).")

df_imgs = pd.DataFrame(registros)
print(f"  Registros válidos  : {len(df_imgs):,}")


# =============================================================================
# PASO 2: Unir fechas de captura desde fechas.csv
# =============================================================================

print()
print("=" * 60)
print("  PASO 2 — Unión con fechas de captura (fechas.csv)")
print("=" * 60)

if not FECHAS_CSV.exists():
    print(f"\n[ERROR] No se encontró {FECHAS_CSV}")
    print("  → Corre 01_code/00_descarga/fetch_fechas.py primero.")
    sys.exit(1)

df_fechas = pd.read_csv(FECHAS_CSV)
print(f"  Registros en fechas.csv: {len(df_fechas):,}")

# Redondear a 8 decimales para garantizar match exacto entre ambas fuentes
# (ambas derivan los floats del mismo nombre de archivo)
for df in (df_imgs, df_fechas):
    df["lat"] = df["lat"].round(8)
    df["lon"] = df["lon"].round(8)

df_imgs = df_imgs.merge(df_fechas[["lat", "lon", "fecha_descargada", "fecha_alternativa"]],
                        on=["lat", "lon"], how="left")

n_sin_fecha = df_imgs["fecha_descargada"].isna().sum()
if n_sin_fecha:
    print(f"  [AVISO] {n_sin_fecha:,} imágenes sin fecha (no estaban en fechas.csv).")

print(f"  Imágenes con fecha : {df_imgs['fecha_descargada'].notna().sum():,}")


# =============================================================================
# PASO 3: Cargar polígonos UPZ
# =============================================================================

print()
print("=" * 60)
print("  PASO 3 — Carga de polígonos UPZ")
print("=" * 60)

candidatos = list(UPZ_DIR.glob("**/*.geojson")) + list(UPZ_DIR.glob("**/*.shp"))
if not candidatos:
    print(f"\n[ERROR] No se encontró ningún archivo de UPZ en {UPZ_DIR}")
    print("  → Corre 01_code/00_descargar_datos.py primero.")
    sys.exit(1)

ruta_upz = candidatos[0]
print(f"  Cargando: {ruta_upz.name}")

upz_gdf = gpd.read_file(ruta_upz)
upz_gdf = upz_gdf.set_crs("EPSG:4326", allow_override=True)
if upz_gdf.crs.to_epsg() != 4326:
    upz_gdf = upz_gdf.to_crs("EPSG:4326")

# Normalizar columnas a nombres canónicos
upz_gdf = upz_gdf.rename(columns={"CODIGO_UPZ": "upz_cod_shp", "NOMBRE": "upz_nom"})
upz_gdf["upz_cod_shp"] = upz_gdf["upz_cod_shp"].astype(str).str.strip()

print(f"  UPZs cargadas: {len(upz_gdf)}")


# =============================================================================
# PASO 4: Spatial join — asignar UPZ a cada imagen
# =============================================================================

print()
print("=" * 60)
print("  PASO 4 — Spatial join: imagen → UPZ")
print("=" * 60)

gdf_imgs = gpd.GeoDataFrame(
    df_imgs,
    geometry=[Point(lon, lat) for lat, lon in zip(df_imgs["lat"], df_imgs["lon"])],
    crs="EPSG:4326",
)

cols_sjoin = ["ruta_imagen", "lat", "lon", "fecha_descargada", "fecha_alternativa", "geometry"]
joined = gpd.sjoin(
    gdf_imgs.loc[:, cols_sjoin],
    upz_gdf[["upz_cod_shp", "upz_nom", "geometry"]],
    how="left",
    predicate="within",
)
joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")

n_sin_upz = joined["upz_cod_shp"].isna().sum()
if n_sin_upz:
    print(f"  [AVISO] {n_sin_upz:,} imágenes fuera de cualquier polígono UPZ — se descartan.")
    joined = joined.dropna(subset=["upz_cod_shp"])

print(f"  Imágenes con UPZ asignada: {len(joined):,}")
print(f"  UPZs presentes           : {joined['upz_cod_shp'].nunique()}")
for upz, n in joined.groupby("upz_nom").size().sort_values(ascending=False).items():
    print(f"    {upz:<30}: {n:6,} imágenes")


# =============================================================================
# PASO 5: Unir indicadores de pobreza desde pobreza_upz.csv
# =============================================================================

print()
print("=" * 60)
print("  PASO 5 — Unión con indicadores de pobreza")
print("=" * 60)

if not POBREZA_CSV.exists():
    print(f"\n[ERROR] No se encontró {POBREZA_CSV}")
    print("  → Corre 01_code/00_descargar_datos.py primero.")
    sys.exit(1)

df_pobreza = pd.read_csv(POBREZA_CSV, dtype={"upz_cod": str})
df_pobreza["upz_cod"] = df_pobreza["upz_cod"].str.strip()
print(f"  UPZs en pobreza_upz.csv: {len(df_pobreza)}")

# Traducir códigos de shapefile a códigos de encuesta.
# Los que no están en MAPA_SHP_A_ENCUESTA se usan directamente (match 1:1).
joined["upz_cod"] = joined["upz_cod_shp"].apply(
    lambda c: MAPA_SHP_A_ENCUESTA.get(c, c)
)

cols_pobreza = ["upz_cod", "inc_monetaria_pct", "inc_ipm_pct", "inc_extrema_pct"]
dataset = joined.merge(df_pobreza[cols_pobreza], on="upz_cod", how="left")

n_sin_pobreza = dataset["inc_monetaria_pct"].isna().sum()
if n_sin_pobreza:
    upzs_sin = dataset.loc[dataset["inc_monetaria_pct"].isna(), "upz_cod_shp"].unique()
    print(f"  [AVISO] {n_sin_pobreza:,} imágenes sin datos de pobreza.")
    print(f"  Códigos shapefile sin match: {sorted(upzs_sin)}")
    print(f"  Se descartan.")
    dataset = dataset.dropna(subset=["inc_monetaria_pct"])

print(f"  Imágenes con pobreza asignada: {len(dataset):,}")


# =============================================================================
# PASO 6: Guardar dataset final
# =============================================================================

print()
print("=" * 60)
print("  PASO 6 — Guardar imagenes_dataset.csv")
print("=" * 60)

columnas_final = [
    "ruta_imagen", "lat", "lon",
    "upz_cod_shp", "upz_cod", "upz_nom",
    "inc_monetaria_pct", "inc_ipm_pct", "inc_extrema_pct",
    "fecha_descargada", "fecha_alternativa",
]
dataset[columnas_final].to_csv(SALIDA_CSV, index=False)
print(f"  Guardado : {SALIDA_CSV}")
print(f"  Filas    : {len(dataset):,}")
print(f"  Columnas : {columnas_final}")
print()
print("  Estadísticas de pobreza por UPZ:")
resumen = (
    dataset.groupby(["upz_cod", "upz_nom"])[
        ["inc_monetaria_pct", "inc_ipm_pct", "inc_extrema_pct"]
    ]
    .first()
    .reset_index()
)
print(f"  {'UPZ':<6} {'Nombre':<30} {'Monetaria':>10} {'IPM':>8} {'Extrema':>9} {'N fotos':>8}")
print(f"  {'-'*72}")
conteo = dataset.groupby("upz_cod").size().rename("n")
for _, row in resumen.sort_values("upz_cod").iterrows():
    n = conteo.get(row["upz_cod"], 0)
    print(
        f"  {row['upz_cod']:<6} {str(row['upz_nom']):<30}"
        f" {row['inc_monetaria_pct']:>9.2f}%"
        f" {row['inc_ipm_pct']:>7.2f}%"
        f" {row['inc_extrema_pct']:>8.2f}%"
        f" {n:>8,}"
    )

print()
print("  Distribución de fechas de captura:")
for fecha, n in dataset["fecha_descargada"].value_counts().sort_index().items():
    print(f"    {fecha}: {n:,} imágenes")


# =============================================================================
# PASO 7: Mapa de cobertura coloreado por pobreza monetaria
# =============================================================================

print()
print("=" * 60)
print("  PASO 7 — Mapa de cobertura coloreado por pobreza monetaria")
print("=" * 60)

vals  = dataset["inc_monetaria_pct"].values
vmin, vmax = vals.min(), vals.max()
norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(11, 11))

# Fondo: polígonos UPZ coloreados por pobreza monetaria
upz_gdf_plot = upz_gdf.copy()
upz_gdf_plot["upz_cod"] = upz_gdf_plot["upz_cod_shp"].apply(
    lambda c: MAPA_SHP_A_ENCUESTA.get(c, c)
)
upz_ipm = df_pobreza[["upz_cod", "inc_monetaria_pct"]].copy()
upz_gdf_plot = upz_gdf_plot.merge(upz_ipm, on="upz_cod", how="left")
upz_gdf_plot.plot(
    ax=ax,
    column="inc_monetaria_pct",
    cmap="RdYlGn_r",
    edgecolor="black",
    linewidth=0.5,
    alpha=0.3,
    legend=False,
    missing_kwds={"color": "lightgray"},
)

# Puntos: imágenes coloreadas por pobreza monetaria de su UPZ
sc = ax.scatter(
    dataset["lon"], dataset["lat"],
    c=vals, cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
    s=0.5, alpha=0.6, linewidths=0,
)

cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Pobreza monetaria (%)", fontsize=12)

# Etiquetas de UPZ
for _, row in upz_gdf_plot.drop_duplicates("upz_cod_shp").iterrows():
    if not row.geometry.is_empty:
        cx = row.geometry.centroid.x
        cy = row.geometry.centroid.y
        ax.text(cx, cy, str(row["upz_nom"]),
                fontsize=6, ha="center", va="center", color="black", alpha=0.7)

ax.set_title(
    f"Cobertura Street View — USME, Bogotá\n"
    f"{len(dataset):,} imágenes · Pobreza monetaria por UPZ (EM 2021)",
    fontsize=13,
)
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
plt.tight_layout()
plt.savefig(SALIDA_MAPA, dpi=150)
plt.close()

print(f"  Mapa guardado: {SALIDA_MAPA}")
print()
print("  Script completado. Siguiente paso:")
print("    python 01_code/02_split.py")

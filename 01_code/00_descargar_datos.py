#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_descargar_datos.py
=====================
Descarga los insumos geográficos y socioeconómicos necesarios para el
pipeline, luego genera la base consolidada de la Encuesta Multipropósito.

Pasos
-----
  1. Polígonos de UPZ (Unidades de Planeamiento Zonal) de Bogotá.
     Fuente: ArcGIS FeatureServer — proyecto Esri Colombia (114 UPZs).
     Campos relevantes: CODIGO_UPZ, NOMBRE, CODIGO_LOCALIDAD.
     URL Hub: https://hub.arcgis.com/datasets/project-esri-co::upz/about

     Nota: El MapServer/44 de Catastro Bogotá que aparece en ArcGIS Hub
     como "UPZ" en realidad contiene UPL (33 polígonos). El servicio
     correcto para UPZ es el FeatureServer indicado arriba.

  2. Encuesta Multipropósito 2021 — microdatos persona (em2021.csv).
     Fuente: Datos Abiertos Bogotá.
     Contiene variables socioeconómicas a nivel persona: DIRECTORIO_PER
     como identificador único, DIRECTORIO (hogar), COD_UPZ_GRUPO, etc.

  3. Variables adicionales EM 2021 (em2021_adicionales.csv).
     Fuente: Datos Abiertos Bogotá.
     Contiene indicadores derivados: déficit habitacional, IPM (N_pobre_ipm),
     pobreza monetaria, etc. Identificador único: directorio_per.

  4. Merge de los dos archivos de la encuesta.
     Llave: DIRECTORIO_PER (em2021) = directorio_per (adicionales).
     Se eliminan las columnas redundantes de adicionales (_id,
     directorio_hog, directorio_per, directorio) antes de unir.
     Resultado: em2021_merged.csv en 00_data/raw/.

Por qué automatizar la descarga
---------------------------------
Un proyecto reproducible no debe depender de archivos descargados
manualmente. Este script descarga todo lo necesario a 00_data/raw/ y
genera la base consolidada. Cualquier colaborador puede reproducir el
ejercicio desde cero corriendo los scripts en orden numérico.

REQUISITOS
----------
  pip install requests geopandas pandas

USO
---
  python 01_code/00_descargar_datos.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

# =============================================================================
# RUTAS
# =============================================================================

_PROYECTO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR        = _PROYECTO_ROOT / "00_data" / "raw"
UPZ_DIR        = RAW_DIR / "upz"

RAW_DIR.mkdir(parents=True, exist_ok=True)
UPZ_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONSTANTES
# =============================================================================

# Servicio ArcGIS REST con los 114 polígonos de UPZ de Bogotá.
UPZ_SERVICE_URL = (
    "https://services1.arcgis.com/7S16A7PAFcmSmqJA/arcgis/rest/services"
    "/UPZ/FeatureServer/0"
)

# Encuesta Multipropósito 2021 — microdatos a nivel persona.
EM2021_URL = (
    "https://datosabiertos.bogota.gov.co/dataset/8ac12a95-1415-4812-b343-f07f90608014"
    "/resource/b3fd892e-b9f9-4f34-ac22-6cc50612eac9/download/em2021.csv"
)

# Variables adicionales EM 2021 (indicadores derivados: IPM, déficit, etc.)
EM2021_ADICIONALES_URL = (
    "https://datosabiertos.bogota.gov.co/dataset/8ac12a95-1415-4812-b343-f07f90608014"
    "/resource/ba44798f-5257-4c5c-8a57-0966da1bb4fa/download"
    "/20240430_variables_adicionales_2021.csv"
)

# Columnas de adicionales que duplican identificadores ya presentes en em2021
COLS_REDUNDANTES_ADICIONALES = ["_id", "directorio_hog", "directorio_per", "directorio"]

TIMEOUT_S = 60


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def descargar_upz_geojson(service_url: str, destino: Path) -> None:
    """
    Descarga todos los polígonos UPZ desde un servicio ArcGIS REST
    y los guarda como GeoJSON en destino.

    Usa paginación automática con resultOffset si el MaxRecordCount
    del servicio es menor que el total de features.
    """
    query_url   = f"{service_url}/query"
    params_base = {
        "where":          "1=1",
        "outFields":      "*",
        "returnGeometry": "true",
        "f":              "geojson",
        "outSR":          "4326",
    }

    info        = requests.get(f"{service_url}?f=json", timeout=30).json()
    max_records = info.get("maxRecordCount", 1000)

    count_r = requests.get(
        query_url,
        params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
        timeout=30,
    ).json()
    total = count_r.get("count", 0)
    print(f"  Total de features en el servicio: {total}")
    print(f"  MaxRecordCount del servicio      : {max_records}")

    features = []
    crs      = None
    offset   = 0

    while offset < total or (total == 0 and offset == 0):
        params = {**params_base, "resultOffset": offset, "resultRecordCount": max_records}
        r      = requests.get(query_url, params=params, timeout=TIMEOUT_S)
        r.raise_for_status()
        data   = r.json()

        batch = data.get("features", [])
        if not batch:
            break

        features.extend(batch)
        if crs is None:
            crs = data.get("crs")

        offset += len(batch)
        sys.stdout.write(f"\r  Descargados: {len(features)}/{total} features")
        sys.stdout.flush()

        if len(batch) < max_records:
            break

    print()

    geojson = {"type": "FeatureCollection", "features": features}
    if crs:
        geojson["crs"] = crs

    with open(destino, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    print(f"  GeoJSON guardado: {destino.name}  ({destino.stat().st_size / 1e3:.0f} KB)")


def descargar_archivo(url: str, destino: Path, descripcion: str = "") -> None:
    """Descarga un archivo con barra de progreso simple."""
    if destino.exists():
        print(f"  [OMITIDO] Ya existe: {destino.name}")
        return

    print(f"  Descargando {descripcion or destino.name} ...")
    with requests.get(url, stream=True, timeout=TIMEOUT_S) as r:
        r.raise_for_status()
        total      = int(r.headers.get("Content-Length", 0))
        descargado = 0
        with open(destino, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                descargado += len(chunk)
                if total:
                    sys.stdout.write(
                        f"\r    {descargado / total * 100:.1f}%"
                        f"  ({descargado / 1e6:.1f} MB)"
                    )
                    sys.stdout.flush()
    print(f"\r    Completado: {destino.name}           ")


# =============================================================================
# PASO 1: Polígonos UPZ
# =============================================================================

print("=" * 60)
print("  PASO 1 — Polígonos UPZ (Catastro Bogotá / ArcGIS REST)")
print("=" * 60)

upz_geojson = UPZ_DIR / "upz_bogota.geojson"

if upz_geojson.exists():
    print(f"  [OK] GeoJSON ya descargado: {upz_geojson.name}")
else:
    try:
        print(f"  Servicio: {UPZ_SERVICE_URL}")
        descargar_upz_geojson(UPZ_SERVICE_URL, upz_geojson)

        import geopandas as gpd
        gdf = gpd.read_file(upz_geojson)
        print(f"  [OK] {len(gdf)} polígonos cargados")
        print(f"  Campos disponibles: {list(gdf.columns)}")

    except Exception as e:
        print(f"\n  [ERROR] No se pudo descargar los polígonos UPZ.")
        print(f"  Detalle: {e}")
        print("  ALTERNATIVA MANUAL:")
        print("  1. Ve a: https://hub.arcgis.com/datasets/project-esri-co::upz/about")
        print("  2. Descarga como GeoJSON y guárdalo en:")
        print(f"     {upz_geojson}")


# =============================================================================
# PASO 2: Encuesta Multipropósito 2021 — microdatos persona
# =============================================================================

print()
print("=" * 60)
print("  PASO 2 — EM 2021: microdatos persona (em2021.csv)")
print("=" * 60)

em2021_csv = RAW_DIR / "em2021.csv"

try:
    descargar_archivo(EM2021_URL, em2021_csv, "Encuesta Multipropósito 2021")
except Exception as e:
    print(f"\n  [ERROR] {e}")
    print("  DESCARGA MANUAL:")
    print("  1. Ve a: https://datosabiertos.bogota.gov.co/dataset/encuesta-multiproposito-2021-sdp")
    print("  2. Descarga em2021.csv y guárdalo en:")
    print(f"     {em2021_csv}")


# =============================================================================
# PASO 3: Variables adicionales EM 2021
# =============================================================================

print()
print("=" * 60)
print("  PASO 3 — EM 2021: variables adicionales (em2021_adicionales.csv)")
print("=" * 60)

em2021_adicionales_csv = RAW_DIR / "em2021_adicionales.csv"

try:
    descargar_archivo(
        EM2021_ADICIONALES_URL,
        em2021_adicionales_csv,
        "EM 2021 variables adicionales",
    )
except Exception as e:
    print(f"\n  [ERROR] {e}")
    print("  DESCARGA MANUAL:")
    print("  1. Ve a: https://datosabiertos.bogota.gov.co/dataset/encuesta-multiproposito-2021-sdp")
    print("  2. Descarga el archivo de variables adicionales y guárdalo en:")
    print(f"     {em2021_adicionales_csv}")


# =============================================================================
# PASO 4: Merge em2021 + adicionales → em2021_merged.csv
# =============================================================================

print()
print("=" * 60)
print("  PASO 4 — Merge EM 2021 + variables adicionales")
print("=" * 60)

em2021_merged_csv = RAW_DIR / "em2021_merged.csv"

if em2021_merged_csv.exists():
    print(f"  [OMITIDO] Ya existe: {em2021_merged_csv.name}")
else:
    if not em2021_csv.exists() or not em2021_adicionales_csv.exists():
        print("  [ERROR] Faltan archivos fuente; completa los pasos 2 y 3 primero.")
    else:
        print("  Leyendo em2021.csv ...")
        em = pd.read_csv(em2021_csv, encoding="latin1")

        print("  Leyendo em2021_adicionales.csv ...")
        adicionales = pd.read_csv(em2021_adicionales_csv, encoding="utf-8-sig")

        print(f"  Haciendo join por DIRECTORIO_PER ({len(em):,} registros) ...")
        merged = em.merge(
            adicionales,
            left_on="DIRECTORIO_PER",
            right_on="directorio_per",
            how="inner",
            validate="1:1",
        )

        # Eliminar columnas redundantes del resultado: _id, directorio_hog,
        # directorio_per y directorio de adicionales son exactamente iguales
        # a DIRECTORIO_PER, DIRECTORIO_HOG y DIRECTORIO de em2021.
        cols_a_borrar = [c for c in COLS_REDUNDANTES_ADICIONALES if c in merged.columns]
        merged = merged.drop(columns=cols_a_borrar)

        merged.to_csv(em2021_merged_csv, index=False, encoding="utf-8")
        print(f"  [OK] {len(merged):,} filas × {len(merged.columns)} columnas")
        print(f"  Guardado: {em2021_merged_csv.name}")


# =============================================================================
# PASO 5: Incidencia de pobreza por UPZ (con expansión de muestra)
# =============================================================================
#
# Indicadores calculados:
#   - inc_monetaria_pct  : % hogares en pobreza monetaria   (N_pobre_monetario)
#   - inc_ipm_pct        : % hogares en pobreza IPM          (N_pobre_ipm)
#   - inc_extrema_pct    : % hogares en pobreza extrema      (N_pobre_extremo)
#
# Metodología:
#   1. Colapsar a nivel hogar (un registro por DIRECTORIO_HOG): los tres
#      indicadores son constantes dentro del hogar.
#   2. Descartar hogares sin UPZ asignada (COD_UPZ_GRUPO == NaN).
#   3. Calcular incidencia ponderada por UPZ para cada indicador:
#        incidencia_pct = 100 × Σ(FEX_C × indicador) / Σ(FEX_C)
#      FEX_C es el factor de expansión del hogar en la Encuesta Multipropósito.
#
# Resultado: pobreza_upz.csv con una fila por UPZ.
# =============================================================================

print()
print("=" * 60)
print("  PASO 5 — Incidencia de pobreza por UPZ")
print("=" * 60)

pobreza_upz_csv = RAW_DIR / "pobreza_upz.csv"

VARS_POBREZA = {
    "N_pobre_monetario": "inc_monetaria_pct",
    "N_pobre_ipm"      : "inc_ipm_pct",
    "N_pobre_extremo"  : "inc_extrema_pct",
}

if pobreza_upz_csv.exists():
    print(f"  [OMITIDO] Ya existe: {pobreza_upz_csv.name}")
else:
    if not em2021_merged_csv.exists():
        print("  [ERROR] Falta em2021_merged.csv; completa el paso 4 primero.")
    else:
        print("  Leyendo em2021_merged.csv ...")
        cols_necesarias = (
            ["DIRECTORIO_HOG", "COD_UPZ_GRUPO", "NOMBRE_UPZ_GRUPO", "FEX_C"]
            + list(VARS_POBREZA.keys())
        )
        df_merged = pd.read_csv(em2021_merged_csv, usecols=cols_necesarias, low_memory=False)

        # Paso 1: colapsar a nivel hogar
        hogares = df_merged.drop_duplicates("DIRECTORIO_HOG").copy()
        print(f"  Hogares en la muestra      : {len(hogares):,}")

        # Paso 2: descartar hogares sin UPZ asignada
        sin_upz = hogares["COD_UPZ_GRUPO"].isna().sum()
        if sin_upz > 0:
            print(f"  [AVISO] {sin_upz:,} hogares sin COD_UPZ_GRUPO — se descartan.")
        hogares = hogares.dropna(subset=["COD_UPZ_GRUPO"])
        hogares["COD_UPZ_GRUPO"] = hogares["COD_UPZ_GRUPO"].astype(int).astype(str)

        # Paso 3: incidencia ponderada por UPZ para los tres indicadores
        def agregar_upz(g: pd.DataFrame) -> pd.Series:
            w   = g["FEX_C"]
            out = {
                "upz_nom"             : g["NOMBRE_UPZ_GRUPO"].iloc[0],
                "n_hogares_muestra"   : len(g),
                "n_hogares_expandidos": round(w.sum(), 1),
            }
            for var, col_out in VARS_POBREZA.items():
                out[col_out] = round(100 * (w * g[var]).sum() / w.sum(), 4)
            return pd.Series(out)

        pobreza_upz = (
            hogares.groupby("COD_UPZ_GRUPO")
            .apply(agregar_upz, include_groups=False)
            .reset_index()
            .rename(columns={"COD_UPZ_GRUPO": "upz_cod"})
        )

        pobreza_upz.to_csv(pobreza_upz_csv, index=False, encoding="utf-8")

        print(f"  UPZs con incidencia calculada: {len(pobreza_upz)}")
        print(f"\n  {'UPZ':<6} {'Nombre':<30} {'Monetaria':>10} {'IPM':>8} {'Extrema':>9}")
        print(f"  {'-'*65}")
        for _, row in pobreza_upz.sort_values("upz_cod").iterrows():
            print(
                f"  {row['upz_cod']:<6} {str(row['upz_nom']):<30}"
                f" {row['inc_monetaria_pct']:>9.2f}%"
                f" {row['inc_ipm_pct']:>7.2f}%"
                f" {row['inc_extrema_pct']:>8.2f}%"
            )
        print(f"\n  Guardado: {pobreza_upz_csv.name}")


# =============================================================================
# RESUMEN
# =============================================================================

print()
print("=" * 60)
print("  RESUMEN")
print("=" * 60)

archivos = {
    "Polígonos UPZ"          : upz_geojson,
    "EM 2021 microdatos"     : em2021_csv,
    "EM 2021 adicionales"    : em2021_adicionales_csv,
    "EM 2021 merged"         : em2021_merged_csv,
    "Pobreza por UPZ"        : pobreza_upz_csv,
}

todo_ok = True
for descripcion, ruta in archivos.items():
    estado = f"OK — {ruta.name}" if ruta.exists() else "PENDIENTE"
    print(f"  {descripcion:<25}: {estado}")
    if not ruta.exists():
        todo_ok = False

print()
if todo_ok:
    print("  Todos los insumos listos. Puedes continuar con:")
    print("    python 01_code/01_cruce_geografico.py")
else:
    print("  Completa las descargas pendientes antes de continuar.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Para cada imagen ya descargada en USME/:
  1. Consulta la Street View Metadata API para obtener la fecha de la foto
     que tenemos descargada (la más cercana a esas coordenadas).
  2. Vuelve a consultar la misma API con source=outdoor para ver si existe
     una panorámica adicional con fecha distinta.

Al final genera:
  - fechas.csv      : lat, lon, fecha_descargada, fecha_alternativa
  - resumen_fechas.csv : fecha, n_coordenadas   (cuántos puntos tienen foto en esa fecha)

El script es reanudable: si fechas.csv existe, saltea las filas ya procesadas.

Peticiones de metadata → gratuitas (no descuentan crédito de Street View).

Uso:
    pip install aiohttp
    python fetch_fechas.py
"""

from __future__ import annotations

import asyncio
import csv
import os
from collections import Counter
from pathlib import Path

import aiohttp

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
API_KEY        = os.environ.get("GSV_API_KEY", "")  # export GSV_API_KEY="..." (Street View Static API)
_ROOT          = Path(__file__).resolve().parents[2]
RUTA_IMGS      = _ROOT / "00_data" / "USME"
RUTA_CSV       = RUTA_IMGS / "fechas.csv"
RUTA_RESUMEN   = RUTA_IMGS / "resumen_fechas.csv"
MAX_CONCURRENT = 40
# ---------------------------------------------------------------------------


def parsear_nombre(nombre: str) -> tuple[float, float] | None:
    """Extrae (lat, lon) del nombre 'lat_lon.jpg'."""
    stem = Path(nombre).stem
    try:
        # El lon puede ser negativo: '4.3106_-74.1745' → 2 partes al dividir por '_'
        # pero '-74.1745' contiene solo un guion, así que split da exactamente 2 tokens.
        idx = stem.index("_-") if "_-" in stem else stem.rindex("_")
        lat = float(stem[:idx])
        lon = float(stem[idx + 1:])
        return lat, lon
    except (ValueError, IndexError):
        return None


async def consultar_metadata(
    session: aiohttp.ClientSession,
    semaforo: asyncio.Semaphore,
    lat: float,
    lon: float,
    source: str = "default",
) -> dict | None:
    """
    Llama al endpoint de metadata y retorna el dict JSON completo,
    o None si falla tras 3 intentos.

    source='default' → panorámica más cercana (la que descargamos).
    source='outdoor' → fuerza panorámica exterior; a veces es de fecha distinta.
    """
    url = (
        f"https://maps.googleapis.com/maps/api/streetview/metadata"
        f"?location={lat},{lon}&source={source}&key={API_KEY}"
    )
    async with semaforo:
        for intento in range(3):
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=20)
                ) as r:
                    return await r.json(content_type=None)
            except Exception:
                if intento == 2:
                    return None
                await asyncio.sleep(2)
    return None


async def procesar(
    session: aiohttp.ClientSession,
    semaforo: asyncio.Semaphore,
    lat: float,
    lon: float,
) -> tuple[float, float, str, str]:
    """
    Retorna (lat, lon, fecha_descargada, fecha_alternativa).
    fecha_alternativa es distinta de fecha_descargada solo si la fuente
    'outdoor' devuelve una panorámica diferente (distinto pano_id).
    """
    meta_default = await consultar_metadata(session, semaforo, lat, lon, "default")
    meta_outdoor = await consultar_metadata(session, semaforo, lat, lon, "outdoor")

    fecha_default = ""
    fecha_outdoor = ""
    pano_default  = ""
    pano_outdoor  = ""

    if meta_default and meta_default.get("status") == "OK":
        fecha_default = meta_default.get("date", "")
        pano_default  = meta_default.get("pano_id", "")

    if meta_outdoor and meta_outdoor.get("status") == "OK":
        fecha_outdoor = meta_outdoor.get("date", "")
        pano_outdoor  = meta_outdoor.get("pano_id", "")

    # Solo reportamos fecha alternativa si es una panorámica distinta
    fecha_alternativa = fecha_outdoor if pano_outdoor != pano_default else ""

    return lat, lon, fecha_default, fecha_alternativa


async def main() -> None:
    # --- Reanudación: cargar pares lat/lon ya procesados ---
    procesadas: set[str] = set()
    if RUTA_CSV.exists():
        with open(RUTA_CSV, newline="") as f:
            for row in csv.DictReader(f):
                procesadas.add(f"{row['lat']}_{row['lon']}")
        print(f"CSV existente: {len(procesadas)} entradas ya procesadas.")

    # --- Listar pendientes ---
    todas      = list(RUTA_IMGS.glob("*.jpg"))
    pendientes = []
    for p in todas:
        coords = parsear_nombre(p.name)
        if coords is None:
            continue
        key = f"{coords[0]}_{coords[1]}"
        if key not in procesadas:
            pendientes.append(coords)

    print(f"Total imágenes: {len(todas)} | Pendientes: {len(pendientes)}")
    if not pendientes:
        print("Nada que procesar. Generando resumen desde CSV existente...")
        generar_resumen()
        return

    # --- Consulta async ---
    semaforo  = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)

    modo = "a" if RUTA_CSV.exists() else "w"
    with open(RUTA_CSV, modo, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if modo == "w":
            writer.writerow(["lat", "lon", "fecha_descargada", "fecha_alternativa"])

        async with aiohttp.ClientSession(connector=connector) as session:
            tareas = [
                asyncio.ensure_future(procesar(session, semaforo, lat, lon))
                for lat, lon in pendientes
            ]

            completadas = 0
            total = len(tareas)
            for fut in asyncio.as_completed(tareas):
                resultado = await fut
                writer.writerow(resultado)
                completadas += 1
                if completadas % 200 == 0 or completadas == total:
                    csvfile.flush()
                    print(f"\r  Consultadas: {completadas}/{total}", end="", flush=True)

    print(f"\n\nfechas.csv guardado en: {RUTA_CSV}")
    generar_resumen()


def generar_resumen() -> None:
    """Lee fechas.csv y escribe resumen_fechas.csv con conteos por fecha."""
    if not RUTA_CSV.exists():
        print("No hay fechas.csv para resumir.")
        return

    conteo_descargadas:   Counter = Counter()
    conteo_alternativas:  Counter = Counter()

    with open(RUTA_CSV, newline="") as f:
        for row in csv.DictReader(f):
            fd = row.get("fecha_descargada", "").strip()
            fa = row.get("fecha_alternativa", "").strip()
            if fd:
                conteo_descargadas[fd] += 1
            if fa:
                conteo_alternativas[fa] += 1

    # Unir todas las fechas conocidas
    todas_fechas = sorted(set(conteo_descargadas) | set(conteo_alternativas))

    with open(RUTA_RESUMEN, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fecha", "n_fotos_descargadas", "n_fotos_alternativas"])
        for fecha in todas_fechas:
            writer.writerow([
                fecha,
                conteo_descargadas.get(fecha, 0),
                conteo_alternativas.get(fecha, 0),
            ])

    print(f"\nResumen de fechas disponibles ({len(todas_fechas)} fechas únicas):")
    print(f"{'Fecha':<12} {'Fotos descargadas':>20} {'Alternativas disponibles':>25}")
    print("-" * 60)
    for fecha in todas_fechas:
        print(
            f"{fecha:<12}"
            f"{conteo_descargadas.get(fecha, 0):>20}"
            f"{conteo_alternativas.get(fecha, 0):>25}"
        )
    print(f"\nresumen_fechas.csv guardado en: {RUTA_RESUMEN}")


asyncio.run(main())

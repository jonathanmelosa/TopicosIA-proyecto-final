#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_cobertura.py
======================
Corrobora si se agotaron todas las imágenes de Street View disponibles
en la localidad procesada por extract_imagenes.py.

El problema que resuelve
------------------------
Cuando extract_imagenes.py termina, la grilla tiene puntos marcados como
imagen=False. Pero "sin imagen" puede significar dos cosas distintas:

  A) Realmente no hay cobertura de Street View en ese punto ni cerca.
     → Agotamos la cobertura disponible. No hay más que hacer.

  B) Hay una imagen cerca pero Google no la asoció porque el punto de
     consulta estaba a más de ~50 m (radio por defecto de la API).
     → Quedan imágenes sin descargar. Hay que ajustar el muestreo.

Este script distingue A de B usando el parámetro `radius` de la API de
metadatos de Street View. La petición estándar busca en un radio de 50 m.
Si re-consultamos los puntos sin imagen con radio=150 m y Google sigue
retornando ZERO_RESULTS, podemos concluir que realmente no hay cobertura.

Estrategia en tres pasos
------------------------
  1. DIAGNÓSTICO ESTÁTICO: estadísticas de la grilla actual (sin API).
     Muestra cobertura total, distribución geográfica y si el script
     terminó o fue interrumpido.

  2. RE-CONSULTA CON RADIO AMPLIADO: para los puntos sin imagen, se vuelve
     a consultar la API con radius=150 m (3× el radio por defecto).
     - Si un punto retorna OK → había una imagen cercana que se perdió
       por el espaciado de la grilla (imagen "escapada").
     - Si retorna ZERO_RESULTS → no hay nada en 150 m a la redonda.

  3. VEREDICTO: con base en el porcentaje de imágenes escapadas se emite
     un juicio sobre si la cobertura está agotada o si conviene afinar
     el muestreo.

Umbral de decisión
------------------
Se considera que la cobertura está agotada si menos del 5 % de los puntos
sin imagen retornan OK con radio ampliado. Por encima del 5 % se recomienda
correr refinar_muestreo.py con un DELTA_M_FINO menor.

El 5 % es un umbral conservador: en una grilla de 10 000 puntos con 4 000
sin imagen, un 5 % equivale a 200 imágenes potencialmente perdidas. Si ese
número es irrelevante para el análisis, el umbral se puede subir a 10 %.

Limitación importante: el radio de verificación debe escalar con el espaciado
------------------------------------------------------------------
El radio de búsqueda (RADIO_AMPLIADO) debe ser proporcional al espaciado
actual de la grilla. Si el radio es mucho mayor que el espaciado, la
verificación produce falsos positivos: un punto sin imagen siempre encontrará
una imagen dentro del radio ampliado porque hay puntos vecinos del mismo
dataset a pocos metros que SÍ tienen imagen. Esas imágenes ya están en la
colección — no son "escapadas".

Regla práctica: usar RADIO_AMPLIADO = 3 × DELTA_M_actual.
  - DELTA_M=50 m → RADIO=150 m  (configuración inicial)
  - DELTA_M=25 m → RADIO=75 m
  - DELTA_M=10 m → RADIO=30 m   (configuración actual, límite práctico)

Cuando se corre refinar_muestreo.py con un DELTA_M_FINO nuevo, actualizar
RADIO_AMPLIADO en este script antes de volver a verificar.

REQUISITOS
----------
  pip install aiohttp pandas geopandas matplotlib
"""

# ---------------------------------------------------------------------------
# Biblioteca estándar
# ---------------------------------------------------------------------------
import asyncio
import json
import os
import signal
import sys

# ---------------------------------------------------------------------------
# Terceros
# ---------------------------------------------------------------------------
import aiohttp
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from requests.exceptions import Timeout
import requests


# =============================================================================
# PARÁMETROS — deben coincidir con los de extract_imagenes.py
# =============================================================================

API_KEY        = "AIzaSyApmNVPLaiGL330cLNdTFc6xvgqjShRTKw"
RUTA_BASE      = "/Users/macbook/Documents/Documentos/MeCA/202601/Seminario de investigación/Literatura/lit_imagenes"
RUTA_SHAPEFILE = "/Users/macbook/Documents/Documentos/MeCA/202601/Seminario de investigación/Literatura/lit_imagenes/loca/Loca.shp"
LOCALIDAD      = "USME"

# Radio de búsqueda para la re-consulta (en metros).
# El radio debe ser proporcional al espaciado actual de la grilla para que
# la verificación sea significativa:
#
#   DELTA_M=50 → RADIO=150  (3×): detecta fotos entre puntos de 50 m
#   DELTA_M=25 → RADIO=75   (3×): detecta fotos entre puntos de 25 m
#   DELTA_M=10 → RADIO=30   (3×): detecta fotos entre puntos de 10 m
#
# Con un radio demasiado grande respecto al espaciado (ej. 150 m con grilla
# de 10 m), la verificación da falsos positivos: encuentra imágenes que ya
# están capturadas por puntos vecinos del mismo dataset y las reporta como
# "escapadas" cuando en realidad ya están en la colección.
#
# Regla práctica: RADIO = 3 × DELTA_M_actual. Actualizar este valor cada
# vez que se corre refinar_muestreo.py con un DELTA_M_FINO más pequeño.
RADIO_AMPLIADO  = 30   # 3 × 10 m (DELTA_M_FINO actual en refinar_muestreo.py)

# Porcentaje de puntos "escapados" por encima del cual se recomienda afinar
# el muestreo. Ver sección "Umbral de decisión" en el docstring.
UMBRAL_ESCAPADOS_PCT = 5.0

# Peticiones simultáneas para la re-consulta (solo metadatos, sin costo)
MAX_CONCURRENT  = 60

# =============================================================================

os.chdir(RUTA_BASE)
ruta_grilla = os.path.join(LOCALIDAD, "grilla_vial.pkl")


# =============================================================================
# PASO 1: DIAGNÓSTICO ESTÁTICO
# Lee grilla_vial.pkl y muestra estadísticas sin llamar a la API.
# =============================================================================

print("=" * 60)
print("  VERIFICACIÓN DE COBERTURA STREET VIEW")
print(f"  Localidad: {LOCALIDAD}")
print("=" * 60)

if not os.path.exists(ruta_grilla):
    print(f"\n[ERROR] No se encontró {ruta_grilla}")
    print("  → Corre extract_imagenes.py primero.")
    sys.exit(1)

grilla = pd.read_pickle(ruta_grilla, compression="gzip")

total       = len(grilla)
con_imagen  = int(grilla["imagen"].sum())
sin_imagen  = total - con_imagen
cobertura   = con_imagen / total * 100

print(f"\n--- Estadísticas de la grilla actual ---")
print(f"  Puntos totales en red vial : {total:>7,}")
print(f"  Con imagen (imagen=True)   : {con_imagen:>7,}  ({cobertura:.1f}%)")
print(f"  Sin imagen (imagen=False)  : {sin_imagen:>7,}  ({100-cobertura:.1f}%)")

# Detectar si el script terminó o fue interrumpido.
# Si hay puntos sin imagen se puede deber a:
#   a) el script no terminó (fue interrumpido), o
#   b) el script terminó pero esas calles no tienen cobertura.
# El paso 2 distingue estos casos para los puntos sin imagen.
if sin_imagen == 0:
    print("\n  Todos los puntos fueron procesados y tienen imagen.")
    print("  La cobertura de la red vial está completamente agotada.")
    print("  No es necesario continuar con la re-consulta.\n")
    # Saltar directo al mapa
    pendientes_idx = []
else:
    print(f"\n  Hay {sin_imagen} puntos sin imagen.")
    print("  Procediendo a re-consultar con radio ampliado para distinguir")
    print(f"  entre 'sin cobertura real' e 'imagen escapada por espaciado'.")
    pendientes_idx = grilla[grilla["imagen"] == False].index.tolist()


# =============================================================================
# PASO 2: RE-CONSULTA CON RADIO AMPLIADO (solo si hay puntos pendientes)
# Para cada punto sin imagen, consulta la API con radius=RADIO_AMPLIADO.
# Solo se consultan metadatos (petición gratuita), no se descargan imágenes.
# =============================================================================

# Resultados de la re-consulta: índice → estado ("OK", "ZERO_RESULTS", etc.)
resultados_reconsulta = {}


async def consultar_radio_ampliado(
    session:  aiohttp.ClientSession,
    semaforo: asyncio.Semaphore,
    parar:    asyncio.Event,
    idx:      int,
    lat:      float,
    lon:      float,
) -> tuple[int, str]:
    """
    Re-consulta un punto sin imagen usando un radio de búsqueda mayor y
    determina si la imagen encontrada ya está en disco o es genuinamente nueva.

    Posibles retornos:
      "ZERO_RESULTS"  → no hay imagen en el radio. Sin cobertura real.
      "YA_CAPTURADA"  → hay imagen dentro del radio pero el archivo .jpg
                        ya existe en disco (capturada por un punto vecino).
                        Es un falso positivo — no es una imagen escapada.
      "OK"            → hay imagen dentro del radio Y el archivo .jpg NO
                        existe en disco. Es una imagen genuinamente escapada.
      "DENEGADO"      → error de API key.
      "ERROR"         → fallo de red tras 3 intentos.

    Por qué comprobar si el archivo existe
    ---------------------------------------
    Con grillas densas (10 m), un punto con imagen=False casi siempre tiene
    un vecino con imagen=True a pocos metros que descargó exactamente la
    misma foto física. Si no comprobamos el archivo, la verificación cuenta
    esa foto como "escapada" cuando ya la tenemos en disco.
    """
    if parar.is_set():
        return idx, "CANCELADO"

    url = (
        f"https://maps.googleapis.com/maps/api/streetview/metadata"
        f"?location={lat},{lon}&radius={RADIO_AMPLIADO}&key={API_KEY}"
    )

    async with semaforo:
        for intento in range(3):
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=15)
                ) as r:
                    data   = await r.json(content_type=None)
                    estado = data.get("status", "UNKNOWN")

                if estado == "REQUEST_DENIED":
                    print(f"\n[ERROR] API denegada: {data.get('error_message', '')}")
                    parar.set()
                    return idx, "DENEGADO"

                if estado == "OK":
                    # Extraer las coordenadas reales de la imagen encontrada
                    # y comprobar si el archivo ya está descargado en disco.
                    lat_real = data.get("location", {}).get("lat")
                    lon_real = data.get("location", {}).get("lng")
                    if lat_real is not None and lon_real is not None:
                        img_nombre = f"{lat_real}_{lon_real}.jpg"
                        if os.path.exists(os.path.join(LOCALIDAD, img_nombre)):
                            return idx, "YA_CAPTURADA"
                    return idx, "OK"

                return idx, estado

            except asyncio.CancelledError:
                return idx, "CANCELADO"
            except Exception:
                if intento == 2:
                    return idx, "ERROR"
                await asyncio.sleep(2)

    return idx, "ERROR"


async def reconsultar_todos() -> None:
    """
    Lanza la re-consulta async de todos los puntos sin imagen y
    almacena los resultados en el diccionario `resultados_reconsulta`.
    """
    if not pendientes_idx:
        return

    parar    = asyncio.Event()
    semaforo = asyncio.Semaphore(MAX_CONCURRENT)

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(
        signal.SIGINT,
        lambda: (
            sys.stdout.write("\n[AVISO] Interrumpido por el usuario.\n"),
            sys.stdout.flush(),
            parar.set(),
        ),
    )

    procesados = 0
    n          = len(pendientes_idx)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    async with aiohttp.ClientSession(connector=connector) as session:
        tareas = [
            consultar_radio_ampliado(
                session, semaforo, parar,
                idx, grilla.lat[idx], grilla.lon[idx],
            )
            for idx in pendientes_idx
        ]

        for coro in asyncio.as_completed(tareas):
            if parar.is_set():
                break

            idx, estado = await coro
            resultados_reconsulta[idx] = estado

            procesados += 1
            sys.stdout.write(f"\r  Re-consultando: {procesados}/{n}")
            sys.stdout.flush()

    print()  # salto de línea tras la barra de progreso


if pendientes_idx:
    print(f"\n--- Re-consulta con radio={RADIO_AMPLIADO} m ({len(pendientes_idx)} puntos) ---")
    asyncio.run(reconsultar_todos())


# =============================================================================
# PASO 3: VEREDICTO
# Analiza los resultados de la re-consulta y emite una conclusión.
# =============================================================================

if pendientes_idx and resultados_reconsulta:

    estados         = pd.Series(resultados_reconsulta)
    n_ok            = (estados == "OK").sum()            # escapadas reales (no están en disco)
    n_ya_capturada  = (estados == "YA_CAPTURADA").sum()  # falsos positivos (ya en disco)
    n_zero          = (estados == "ZERO_RESULTS").sum()  # sin cobertura real
    n_otros         = len(estados) - n_ok - n_ya_capturada - n_zero
    pct_escapados   = n_ok / len(pendientes_idx) * 100

    print(f"\n--- Resultados de la re-consulta ---")
    print(f"  Imagen nueva (no está en disco)  : {n_ok:>6,}  ({pct_escapados:.1f}%)")
    print(f"  Imagen ya capturada (en disco)   : {n_ya_capturada:>6,}  ({n_ya_capturada/len(pendientes_idx)*100:.1f}%)")
    print(f"  ZERO_RESULTS (sin cobertura)     : {n_zero:>6,}  ({n_zero/len(pendientes_idx)*100:.1f}%)")
    print(f"  Errores / cancelados             : {n_otros:>6,}")

    print(f"\n--- Veredicto ---")
    if pct_escapados < UMBRAL_ESCAPADOS_PCT:
        print(f"  COBERTURA AGOTADA.")
        print(f"  Solo el {pct_escapados:.1f}% de los puntos sin imagen tienen una foto")
        print(f"  nueva (no descargada aún) dentro de {RADIO_AMPLIADO} m.")
        print(f"  Los {n_ya_capturada:,} puntos con 'imagen ya capturada' son falsos positivos:")
        print(f"  la foto existe en disco, descargada por un punto vecino.")
        print(f"  Los {n_zero:,} puntos restantes no tienen cobertura real de Street View.")
        print(f"  No es necesario afinar el muestreo.")
    else:
        print(f"  QUEDAN IMÁGENES POR RECOGER.")
        print(f"  El {pct_escapados:.1f}% de los puntos sin imagen tienen una foto")
        print(f"  genuinamente nueva dentro de {RADIO_AMPLIADO} m (umbral: {UMBRAL_ESCAPADOS_PCT}%).")
        print(f"  Esto equivale a ~{n_ok:,} imágenes que no están en disco.")
        print(f"  Recomendación: corre refinar_muestreo.py con DELTA_M_FINO=10 m.")
        print(f"  10 m es el límite práctico: coincide con el intervalo mínimo real")
        print(f"  de captura de los vehículos de Google Street View.")
        print(f"  Bajar de 10 m no añade imágenes nuevas — solo peticiones redundantes.")

    # Guardar solo las escapadas genuinas (no las ya capturadas)
    idx_escapados = estados[estados == "OK"].index.tolist()
    if idx_escapados:
        escapados_df = grilla.loc[idx_escapados, ["lat", "lon"]].copy()
        ruta_escapados = os.path.join(LOCALIDAD, "puntos_escapados.csv")
        escapados_df.to_csv(ruta_escapados, index=True)
        print(f"\n  Puntos con imagen escapada guardados en: {ruta_escapados}")
    else:
        # Limpiar CSV anterior si ya no hay escapados reales
        ruta_escapados = os.path.join(LOCALIDAD, "puntos_escapados.csv")
        if os.path.exists(ruta_escapados):
            os.remove(ruta_escapados)

elif not pendientes_idx:
    print(f"\n--- Veredicto ---")
    print(f"  COBERTURA AGOTADA.")
    print(f"  Todos los puntos de la red vial tienen imagen=True.")


# =============================================================================
# PASO 4: MAPA DE COBERTURA DETALLADO
# Tres categorías: con imagen, sin cobertura real, imagen escapada.
# =============================================================================

shp_bog = gpd.read_file(RUTA_SHAPEFILE, encoding="utf-8")
shp_bog = shp_bog.set_index("LocNombre", drop=True)
shp_bog = shp_bog.set_crs("EPSG:4326", allow_override=True)

fig, ax = plt.subplots(figsize=(11, 11))
gpd.GeoDataFrame(shp_bog.loc[[LOCALIDAD]]).plot(
    ax=ax, color="whitesmoke", edgecolor="black", linewidth=1
)

# Puntos con imagen (azul)
mask_ok = grilla["imagen"] == True
ax.scatter(
    grilla.loc[mask_ok, "lon"],
    grilla.loc[mask_ok, "lat"],
    s=2, color="steelblue", label=f"Con imagen ({con_imagen:,})", zorder=3,
)

if resultados_reconsulta:
    estados = pd.Series(resultados_reconsulta)

    # Puntos sin cobertura real (gris claro)
    idx_zero = estados[estados == "ZERO_RESULTS"].index
    ax.scatter(
        grilla.loc[idx_zero, "lon"],
        grilla.loc[idx_zero, "lat"],
        s=1, color="lightgray", alpha=0.5,
        label=f"Sin cobertura real ({len(idx_zero):,})", zorder=2,
    )

    # Falsos positivos: imagen encontrada pero ya está en disco (naranja)
    idx_ya = estados[estados == "YA_CAPTURADA"].index
    if len(idx_ya) > 0:
        ax.scatter(
            grilla.loc[idx_ya, "lon"],
            grilla.loc[idx_ya, "lat"],
            s=1, color="orange", alpha=0.4,
            label=f"Ya capturada por punto vecino ({len(idx_ya):,})", zorder=3,
        )

    # Imágenes genuinamente escapadas: hay foto nueva, no está en disco (rojo)
    idx_esc = estados[estados == "OK"].index
    if len(idx_esc) > 0:
        ax.scatter(
            grilla.loc[idx_esc, "lon"],
            grilla.loc[idx_esc, "lat"],
            s=4, color="crimson", alpha=0.8,
            label=f"Imagen escapada — no descargada ({len(idx_esc):,})", zorder=4,
        )
else:
    # Sin re-consulta: puntos sin imagen en rojo claro genérico
    ax.scatter(
        grilla.loc[~mask_ok, "lon"],
        grilla.loc[~mask_ok, "lat"],
        s=1, color="lightcoral", alpha=0.4,
        label=f"Sin imagen ({sin_imagen:,})", zorder=2,
    )

ax.set_title(
    f"Verificación de cobertura Street View — {LOCALIDAD}\n"
    f"Cobertura: {cobertura:.1f}%  |  Radio re-consulta: {RADIO_AMPLIADO} m",
    fontsize=13,
)
ax.legend(markerscale=4, fontsize=10)
plt.tight_layout()

ruta_mapa = os.path.join(LOCALIDAD, "verificacion_cobertura.png")
plt.savefig(ruta_mapa, dpi=150)
plt.show()
print(f"\nMapa guardado en: {ruta_mapa}")

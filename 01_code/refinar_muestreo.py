#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refinar_muestreo.py
===================
Descarga imágenes de Street View que quedaron sin capturar en la primera
pasada de extract_imagenes.py por culpa del espaciado entre puntos.

Cuándo correr este script
--------------------------
SOLO si verificar_cobertura.py emitió el veredicto:
  "QUEDAN IMÁGENES POR RECOGER"
  (más del 5 % de puntos sin imagen retornan OK con radio de 150 m)

Si el veredicto fue "COBERTURA AGOTADA", este script no aportará nada y
solo gastará peticiones de API gratuitas innecesariamente.

Por qué no simplemente cambiar DELTA_M en extract_imagenes.py
--------------------------------------------------------------
Reducir DELTA_M de 50 m a 25 m y volver a correr extract_imagenes.py
generaría ~4× más puntos y reprocesaría calles ya cubiertas, desperdiciando
tiempo y peticiones. Este script hace lo correcto:

  1. Carga la grilla existente (grilla_vial.pkl, generada con DELTA_M=50 m).
  2. Genera una grilla fina nueva con DELTA_M_FINO (por defecto 25 m).
  3. Para cada punto de la grilla fina, verifica si ya existe un punto
     exitoso (imagen=True) en la grilla existente a menos de DELTA_M_FINO
     metros. Si existe, descarta el punto nuevo — esa zona ya está cubierta.
  4. Solo procesa los puntos genuinamente nuevos (zonas entre los puntos
     de la grilla original donde podrían existir imágenes sin capturar).
  5. Fusiona los nuevos resultados en grilla_vial.pkl para que el estado
     de la descarga quede centralizado en un solo archivo.

Qué esperar
-----------
Con DELTA_M_FINO=25 m el número de puntos nuevos a procesar debería ser
mucho menor que la grilla completa de 25 m, porque la mayoría de las calles
ya están cubiertas. El beneficio se concentra en:
  - Calles cortas que quedaron entre dos puntos de 50 m sin ser muestreadas.
  - Intersecciones donde el punto de 50 m quedó desplazado respecto a la foto.
  - Zonas con alta densidad de imágenes (centro de la localidad) donde
    el espaciado de 50 m es demasiado grueso.

Limitación conocida
-------------------
10 m es el límite práctico de refinamiento. Google captura sus fotos cada
~10 m en calles principales, así que muestrear a menor distancia no añade
imágenes nuevas: la API retorna la misma foto y os.path.exists() la descarta.
Bajar de 10 m solo incrementa el tiempo y las peticiones sin beneficio real.

REQUISITOS
----------
  pip install aiohttp osmnx geopandas shapely scikit-learn matplotlib
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
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from requests.exceptions import Timeout
from scipy.spatial import cKDTree   # búsqueda de vecino más cercano en O(log N)
from shapely.geometry import Point


# =============================================================================
# PARÁMETROS — deben coincidir con los de extract_imagenes.py
# =============================================================================

API_KEY        = "AIzaSyApmNVPLaiGL330cLNdTFc6xvgqjShRTKw"
RUTA_BASE      = "/Users/macbook/Documents/Documentos/MeCA/202601/Seminario de investigación/Literatura/lit_imagenes"
RUTA_SHAPEFILE = "/Users/macbook/Documents/Documentos/MeCA/202601/Seminario de investigación/Literatura/lit_imagenes/loca/Loca.shp"
LOCALIDAD      = "USME"

# Espaciado de la grilla fina. Debe ser menor que DELTA_M del script original.
#
# ¿Por qué 10 m y no menos?
# Google captura fotos cada ~10 m en calles principales y cada ~20–30 m en
# calles secundarias. Bajar de 10 m no añade imágenes nuevas porque entre dos
# fotos consecutivas de Google no hay nada que capturar: la API retorna la
# misma foto más cercana, el nombre de archivo coincide y os.path.exists()
# la descarta sin costo. Solo se pierde tiempo en peticiones de metadatos.
#
# ¿Por qué no 15 m?
# Con 15 m se pueden escapar fotos en calles donde Google capturó cada 10–12 m,
# que es el intervalo mínimo real de sus vehículos de Street View.
# 10 m garantiza que no queda ninguna foto sin capturar independientemente
# del intervalo de captura de Google en esa calle.
DELTA_M_FINO   = 10

# Radio de exclusión: puntos de la grilla fina a menos de esta distancia
# de un punto ya exitoso (imagen=True) se descartan.
# Se usa DELTA_M_FINO (10 m) para no dejar huecos ni solapar: cualquier
# punto nuevo que caiga dentro de 10 m de una imagen ya descargada no
# aportará una foto distinta.
RADIO_EXCLUSION_M = DELTA_M_FINO

IMG_SIZE       = "640x640"
MAX_CONCURRENT = 40
GUARDAR_CADA_S = 30

# =============================================================================

os.chdir(RUTA_BASE)
ruta_grilla = os.path.join(LOCALIDAD, "grilla_vial.pkl")


# =============================================================================
# PASO 0: Guardia — verificar que tiene sentido correr este script
# =============================================================================

print("=" * 60)
print("  REFINAMIENTO DE MUESTREO")
print(f"  Localidad: {LOCALIDAD}  |  DELTA_M_FINO: {DELTA_M_FINO} m")
print("=" * 60)

if not os.path.exists(ruta_grilla):
    print(f"\n[ERROR] No se encontró {ruta_grilla}")
    print("  → Corre primero extract_imagenes.py y luego verificar_cobertura.py")
    sys.exit(1)

# Si existe puntos_escapados.csv (generado por verificar_cobertura.py),
# lo usamos para confirmar que hay imágenes escapadas. Si no existe, avisamos.
ruta_escapados = os.path.join(LOCALIDAD, "puntos_escapados.csv")
if not os.path.exists(ruta_escapados):
    print("\n[AVISO] No se encontró puntos_escapados.csv")
    print("  → Se recomienda correr verificar_cobertura.py antes de este script.")
    print("  → Si ya lo corriste y el veredicto fue 'COBERTURA AGOTADA',")
    print("     no es necesario continuar.")
    respuesta = input("  → ¿Deseas continuar de todas formas? (s/n): ").strip().lower()
    if respuesta != "s":
        sys.exit(0)
else:
    escapados = pd.read_csv(ruta_escapados)
    print(f"\n  verificar_cobertura.py detectó {len(escapados)} imágenes escapadas.")
    print(f"  Este script intentará capturarlas con DELTA_M={DELTA_M_FINO} m.\n")


# =============================================================================
# PASO 1: Cargar grilla existente
# =============================================================================

grilla_existente = pd.read_pickle(ruta_grilla, compression="gzip")
n_existentes     = len(grilla_existente)
n_con_imagen     = int(grilla_existente["imagen"].sum())

print(f"--- Grilla existente (DELTA_M=50 m) ---")
print(f"  Puntos totales  : {n_existentes:,}")
print(f"  Con imagen      : {n_con_imagen:,}  ({n_con_imagen/n_existentes*100:.1f}%)")
print(f"  Sin imagen      : {n_existentes - n_con_imagen:,}")


# =============================================================================
# PASO 2: Generar grilla fina sobre la red vial
# =============================================================================

def samplear_red_vial(poligono, delta_m: int) -> pd.DataFrame:
    """
    Genera puntos cada delta_m metros sobre la red vial vehicular.
    Igual que en extract_imagenes.py — ver ese script para documentación
    completa de las decisiones metodológicas.
    """
    G = ox.graph_from_polygon(poligono, network_type="drive")
    _, edges = ox.graph_to_gdfs(G)

    utm_crs   = edges.estimate_utm_crs()
    edges_utm = edges.to_crs(utm_crs)

    puntos_geom = []
    for geom in edges_utm.geometry:
        longitud = geom.length
        n        = max(2, int(longitud / delta_m))
        for d in np.linspace(0, longitud, n):
            puntos_geom.append(geom.interpolate(d))

    gdf = gpd.GeoDataFrame(geometry=puntos_geom, crs=utm_crs).to_crs("EPSG:4326")
    df  = pd.DataFrame({
        "lon": gdf.geometry.x.round(5),
        "lat": gdf.geometry.y.round(5),
    }).drop_duplicates().reset_index(drop=True)
    df["imagen"] = False
    return df


print(f"\n--- Generando grilla fina (DELTA_M={DELTA_M_FINO} m) ---")
shp_bog  = gpd.read_file(RUTA_SHAPEFILE, encoding="utf-8").set_index("LocNombre", drop=True)
shp_bog  = shp_bog.set_crs("EPSG:4326", allow_override=True)
poligono = shp_bog.loc[LOCALIDAD, "geometry"]

grilla_fina = samplear_red_vial(poligono, delta_m=DELTA_M_FINO)
print(f"  Puntos en grilla fina: {len(grilla_fina):,}")


# =============================================================================
# PASO 3: Descartar puntos de la grilla fina ya cubiertos por la existente
#
# Usamos cKDTree (árbol k-d) para encontrar eficientemente el punto más
# cercano en la grilla existente a cada punto de la grilla fina.
# cKDTree trabaja en coordenadas cartesianas planas, por lo que convertimos
# grados a metros aproximados multiplicando por el factor de escala en Bogotá
# (1° lat ≈ 111 000 m, 1° lon ≈ 99 000 m a latitud 4.5°).
# La conversión no es exacta pero el error es <1 % a esta escala, suficiente
# para decidir si dos puntos están a menos de 25 m.
# =============================================================================

print(f"\n--- Filtrando puntos ya cubiertos (radio={RADIO_EXCLUSION_M} m) ---")

# Factor de conversión grados → metros en Bogotá (latitud ~4.5°)
LAT_M = 111_000          # metros por grado de latitud
LON_M = 99_000           # metros por grado de longitud (cos(4.5°) × 111 000)

# Coordenadas en metros de los puntos con imagen=True en la grilla existente
exitosos = grilla_existente[grilla_existente["imagen"] == True]
if len(exitosos) == 0:
    print("  [AVISO] No hay puntos con imagen=True en la grilla existente.")
    print("  → No se puede filtrar. Se procesarán todos los puntos de la grilla fina.")
    puntos_nuevos = grilla_fina.copy()
else:
    coords_exitosos = np.column_stack([
        exitosos["lat"].values * LAT_M,
        exitosos["lon"].values * LON_M,
    ])
    coords_finos = np.column_stack([
        grilla_fina["lat"].values * LAT_M,
        grilla_fina["lon"].values * LON_M,
    ])

    # Para cada punto de la grilla fina, buscamos el punto exitoso más cercano.
    # Si está a menos de RADIO_EXCLUSION_M metros, el punto ya está cubierto.
    arbol      = cKDTree(coords_exitosos)
    distancias, _ = arbol.query(coords_finos, k=1)

    mask_nuevo = distancias > RADIO_EXCLUSION_M
    puntos_nuevos = grilla_fina[mask_nuevo].reset_index(drop=True)

print(f"  Puntos de grilla fina descartados (ya cubiertos) : {(~mask_nuevo).sum():,}")
print(f"  Puntos genuinamente nuevos a procesar            : {len(puntos_nuevos):,}")

if len(puntos_nuevos) == 0:
    print("\n  No hay puntos nuevos. La cobertura con DELTA_M=50 m ya es suficiente.")
    print("  El veredicto de verificar_cobertura.py pudo ser un falso positivo.")
    sys.exit(0)


# =============================================================================
# PASO 4: Prueba síncrona de la API
# =============================================================================

N_PRUEBA       = 5
muestra_prueba = puntos_nuevos.sample(min(N_PRUEBA, len(puntos_nuevos)), random_state=0)

print(f"\n--- Prueba de API ({N_PRUEBA} puntos) ---")
for _, fila in muestra_prueba.iterrows():
    url_test = (
        f"https://maps.googleapis.com/maps/api/streetview/metadata"
        f"?location={fila['lat']},{fila['lon']}&key={API_KEY}"
    )
    try:
        r      = requests.get(url_test, timeout=15)
        data   = json.loads(r.text)
        estado = data.get("status", "UNKNOWN")
        if estado == "REQUEST_DENIED":
            print(f"\n[ERROR] API denegada: {data.get('error_message', '')}")
            sys.exit(1)
    except (ConnectionError, Timeout) as e:
        print(f"\n[ERROR] Fallo de red: {e}")
        sys.exit(1)

print("  Prueba superada.")
print(f"\nIniciando descarga de {len(puntos_nuevos):,} puntos nuevos...")
print("Presiona Ctrl+C para detener y guardar el progreso.\n")


# =============================================================================
# PASO 5: Descarga asíncrona de puntos nuevos
# Igual que en extract_imagenes.py — ver ese script para documentación completa.
# =============================================================================

async def auto_guardar(parar: asyncio.Event, grilla_combinada: pd.DataFrame) -> None:
    while not parar.is_set():
        await asyncio.sleep(GUARDAR_CADA_S)
        if not parar.is_set():
            grilla_combinada.to_pickle(ruta_grilla, compression="gzip")


async def procesar_punto(
    session:  aiohttp.ClientSession,
    semaforo: asyncio.Semaphore,
    parar:    asyncio.Event,
    idx:      int,
    lat:      float,
    lon:      float,
) -> tuple[int, bool]:
    if parar.is_set():
        return idx, False

    url_meta = (
        f"https://maps.googleapis.com/maps/api/streetview/metadata"
        f"?location={lat},{lon}&key={API_KEY}"
    )
    url_img = (
        f"https://maps.googleapis.com/maps/api/streetview"
        f"?size={IMG_SIZE}&location={lat},{lon}&fov=90&pitch=0&key={API_KEY}"
    )

    async with semaforo:
        for intento in range(3):
            try:
                async with session.get(
                    url_meta, timeout=aiohttp.ClientTimeout(total=20)
                ) as r:
                    data   = await r.json(content_type=None)
                    estado = data.get("status", "UNKNOWN")

                if estado == "REQUEST_DENIED":
                    print(f"\n[ERROR] API denegada.")
                    parar.set()
                    return idx, False

                imagen_disponible = (
                    estado == "OK"
                    and "oogle" in data.get("copyright", "")
                )
                if not imagen_disponible:
                    return idx, False

                lat_real   = data["location"]["lat"]
                lon_real   = data["location"]["lng"]
                img_nombre = f"{lat_real}_{lon_real}.jpg"
                ruta_img   = os.path.join(LOCALIDAD, img_nombre)

                if not os.path.exists(ruta_img):
                    async with session.get(
                        url_img, timeout=aiohttp.ClientTimeout(total=30)
                    ) as r_img:
                        r_img.raise_for_status()
                        contenido = await r_img.read()
                    with open(ruta_img, "wb") as f:
                        f.write(contenido)

                return idx, True

            except asyncio.CancelledError:
                return idx, False
            except Exception:
                if intento == 2:
                    return idx, False
                await asyncio.sleep(2)

    return idx, False


async def main() -> None:
    parar    = asyncio.Event()
    semaforo = asyncio.Semaphore(MAX_CONCURRENT)

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(
        signal.SIGINT,
        lambda: (
            sys.stdout.write("\n\n[AVISO] Deteniendo... guardando progreso.\n"),
            sys.stdout.flush(),
            parar.set(),
        ),
    )

    procesados     = 0
    imagenes_nuevas = 0
    n_nuevos        = len(puntos_nuevos)

    # La grilla combinada se construye antes de empezar para poder guardarla
    # periódicamente desde el task de auto-guardado.
    # Se construye concatenando la existente con los nuevos puntos, asignando
    # índices que continúan desde donde termina la grilla existente.
    grilla_combinada = pd.concat(
        [grilla_existente, puntos_nuevos],
        ignore_index=True,
    )
    # Los índices de puntos_nuevos dentro de grilla_combinada
    offset      = len(grilla_existente)
    idx_nuevos  = list(range(offset, offset + n_nuevos))

    tarea_guardado = asyncio.create_task(auto_guardar(parar, grilla_combinada))

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    async with aiohttp.ClientSession(connector=connector) as session:
        tareas = [
            procesar_punto(
                session, semaforo, parar,
                idx_nuevos[i],
                puntos_nuevos.lat.iloc[i],
                puntos_nuevos.lon.iloc[i],
            )
            for i in range(n_nuevos)
        ]

        for coro in asyncio.as_completed(tareas):
            if parar.is_set():
                break

            idx, tiene_imagen = await coro

            if tiene_imagen:
                grilla_combinada.loc[idx, "imagen"] = True
                imagenes_nuevas += 1

            procesados += 1
            sys.stdout.write(
                f"\r  Procesados: {procesados}/{n_nuevos}"
                f"  |  Imágenes nuevas: {imagenes_nuevas}"
            )
            sys.stdout.flush()

    tarea_guardado.cancel()

    # Guardar la grilla fusionada en el mismo archivo que usa el resto del flujo
    grilla_combinada.to_pickle(ruta_grilla, compression="gzip")

    total_con_imagen = int(grilla_combinada["imagen"].sum())
    print(f"\n\nGrilla actualizada guardada en: {ruta_grilla}")
    print(f"Imágenes nuevas en esta pasada : {imagenes_nuevas}")
    print(f"Total imágenes acumuladas      : {total_con_imagen} / {len(grilla_combinada)}")


asyncio.run(main())

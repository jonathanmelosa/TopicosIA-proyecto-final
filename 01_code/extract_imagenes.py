#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga imágenes de Google Street View sobre la red vial de una localidad
de Bogotá. Combina muestreo sobre OpenStreetMap con peticiones asíncronas.

Decisiones metodológicas principales:
  1. Red vial (OSMnx) en lugar de grilla uniforme: Street View solo existe
     en calles, por lo que ~80 % de los puntos de una grilla uniforme
     retornan ZERO_RESULTS. Muestrear sobre la red vial reduce los puntos
     de ~80 k a ~10 k y sube la tasa de éxito de ~15 % a ~70 %.
  2. Async (aiohttp) en lugar de threads: un solo hilo maneja 40+ peticiones
     concurrentes sin bloqueo de GIL ni overhead de cambio de contexto.
  3. Reanudabilidad automática: si grilla.pkl existe, se recarga y solo se
     procesan los puntos todavía pendientes (imagen == False).

REQUISITOS:
  pip install geopandas osmnx aiohttp requests shapely scikit-learn matplotlib
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
import aiohttp                              # peticiones HTTP asíncronas
import geopandas as gpd                    # shapefiles
import matplotlib.pyplot as plt            # mapa de cobertura final
import numpy as np
import osmnx as ox                         # red vial de OpenStreetMap
import pandas as pd
import requests                            # solo para la prueba síncrona inicial
from requests.exceptions import Timeout
from sklearn.metrics import DistanceMetric # distancia Haversine entre puntos


# =============================================================================
# PARÁMETROS — edita solo esta sección
# =============================================================================

API_KEY        = "AIzaSyApmNVPLaiGL330cLNdTFc6xvgqjShRTKw"  # ← clave Google Maps Platform
RUTA_BASE      = "/Users/macbook/Documents/Documentos/MeCA/202601/Seminario de investigación/Literatura/lit_imagenes"
RUTA_SHAPEFILE = "/Users/macbook/Documents/Documentos/MeCA/202601/Seminario de investigación/Literatura/lit_imagenes/loca/Loca.shp"
LOCALIDAD      = "USME"    # nombre exacto en la columna LocNombre del shapefile

DELTA_M        = 50        # metros entre puntos consecutivos sobre cada calle
MAX_CONCURRENT = 40        # peticiones simultáneas máximas (bajar si ves OVER_QUERY_LIMIT)
GUARDAR_CADA_S = 30        # segundos entre auto-guardados de grilla.pkl
IMG_SIZE       = "640x640" # resolución de imagen (máx gratuito: 640x640)

# =============================================================================

os.chdir(RUTA_BASE)


# =============================================================================
# FUNCIÓN: samplear_red_vial()
# Reemplaza crear_grilla(). Genera puntos a lo largo de las calles reales
# en vez de sobre todo el polígono (incluidos edificios, parques, montañas).
# =============================================================================

def samplear_red_vial(poligono, delta_m: int = 50) -> pd.DataFrame:
    """
    Descarga la red vial dentro del polígono desde OpenStreetMap y genera
    puntos uniformemente espaciados cada delta_m metros sobre cada calle.

    Pasos internos:
      1. ox.graph_from_polygon → grafo dirigido de calles vehiculares.
      2. Proyección a UTM (coordenadas métricas) para medir distancias reales.
      3. Interpolación lineal sobre cada arista cada delta_m metros.
      4. Reconversión a WGS84 y eliminación de duplicados (las aristas
         bidireccionales del grafo generan el mismo punto dos veces).

    Parámetros:
      poligono : Shapely Polygon en EPSG:4326 (WGS84)
      delta_m  : separación en metros entre puntos consecutivos

    Retorna:
      DataFrame con columnas 'lon', 'lat', 'imagen' (False por defecto)
    """
    print("Descargando red vial de OpenStreetMap...")

    # network_type='drive' incluye solo vías vehiculares.
    # Street View existe principalmente donde circulan vehículos, no en
    # senderos peatonales ni ciclovías sin tráfico motorizado.
    G = ox.graph_from_polygon(poligono, network_type="drive")
    _, edges = ox.graph_to_gdfs(G)

    # Proyectamos a UTM para que las distancias estén en metros.
    # estimate_utm_crs() detecta automáticamente la zona UTM de Bogotá (18N).
    utm_crs   = edges.estimate_utm_crs()
    edges_utm = edges.to_crs(utm_crs)

    puntos_geom = []
    for geom in edges_utm.geometry:
        longitud = geom.length                     # metros
        n        = max(2, int(longitud / delta_m)) # puntos a interpolar
        for d in np.linspace(0, longitud, n):
            puntos_geom.append(geom.interpolate(d))

    # Volvemos a WGS84, que es el sistema que usa la API de Google
    gdf = gpd.GeoDataFrame(geometry=puntos_geom, crs=utm_crs).to_crs("EPSG:4326")

    # Redondeo a 5 decimales ≈ 1.1 m de precisión, suficiente para evitar
    # duplicados sin perder resolución útil
    df = pd.DataFrame({
        "lon": gdf.geometry.x.round(5),
        "lat": gdf.geometry.y.round(5),
    }).drop_duplicates().reset_index(drop=True)

    df["imagen"] = False
    return df


# =============================================================================
# FUNCIÓN: distancia_promedio()
# Diagnóstico: estima la separación real entre puntos vecinos.
# =============================================================================

def distancia_promedio(df: pd.DataFrame) -> float:
    """
    Calcula la distancia promedio en metros entre cada punto y su vecino
    más cercano usando Haversine sobre una muestra de hasta 500 puntos.
    """
    muestra = df.sample(min(500, len(df)), random_state=42).copy()
    muestra["lon_rad"] = np.radians(muestra["lon"])
    muestra["lat_rad"] = np.radians(muestra["lat"])

    dist = 6371 * DistanceMetric.get_metric("haversine").pairwise(
        muestra[["lat_rad", "lon_rad"]],
        muestra[["lat_rad", "lon_rad"]],
    )
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=0).mean() * 1000  # km → m


# =============================================================================
# PASO 1: Cargar shapefile de localidades
# =============================================================================

shp_bog = gpd.read_file(RUTA_SHAPEFILE, encoding="utf-8")
shp_bog = shp_bog.set_index("LocNombre", drop=True)
# allow_override=True evita error si el .prj ya declara el CRS
shp_bog = shp_bog.set_crs("EPSG:4326", allow_override=True)

poligono    = shp_bog.loc[LOCALIDAD, "geometry"]
ruta_grilla = os.path.join(LOCALIDAD, "grilla_vial.pkl")

if not os.path.exists(LOCALIDAD):
    os.mkdir(LOCALIDAD)
    print(f"Carpeta '{LOCALIDAD}' creada.")
else:
    print(f"Carpeta '{LOCALIDAD}' encontrada. Continuando ahí.")


# =============================================================================
# PASO 2: Cargar grilla anterior o generar desde red vial
# =============================================================================

if os.path.exists(ruta_grilla):
    # Reanudación: se recarga el estado previo para no reprocesar puntos ya
    # visitados. Solo se procesarán los que tienen imagen == False.
    grilla = pd.read_pickle(ruta_grilla, compression="gzip")
    ya     = int(grilla["imagen"].sum())
    total  = len(grilla)
    print(f"Grilla anterior cargada: {ya}/{total} puntos ya procesados.")
else:
    grilla = samplear_red_vial(poligono, delta_m=DELTA_M)
    total  = len(grilla)
    dist_m = distancia_promedio(grilla)
    print(f"Red vial muestreada: {total} puntos | separación promedio: {dist_m:.0f} m")

pendientes = grilla[grilla["imagen"] == False].index.tolist()
print(f"Puntos pendientes: {len(pendientes)} / {total}\n")


# =============================================================================
# PASO 3: Prueba síncrona de la API (10 puntos antes de lanzar async)
# =============================================================================

N_PRUEBA       = 10
muestra_prueba = grilla.sample(min(N_PRUEBA, len(grilla)), random_state=0)
img_en_muestra = 0

print(f"Probando API con {N_PRUEBA} puntos de muestra...")
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
            print("  → Verifica que 'Street View Static API' esté habilitada")
            print("    en Google Cloud Console y que la clave tenga facturación activa.")
            sys.exit(1)
        elif estado == "OK":
            img_en_muestra += 1

    except (ConnectionError, Timeout) as e:
        print(f"\n[ERROR] Fallo de red durante la prueba: {e}")
        sys.exit(1)

print(f"Prueba superada — {img_en_muestra}/{N_PRUEBA} puntos con imagen.")
if img_en_muestra == 0:
    print("[AVISO] La API funciona pero no hay imágenes en la muestra.")
    print("  → Verifica la ubicación geográfica del shapefile.")
print(f"\nIniciando descarga async ({MAX_CONCURRENT} peticiones simultáneas).")
print("Presiona Ctrl+C para detener y guardar el progreso.\n")


# =============================================================================
# PASO 4: Descarga asíncrona
# =============================================================================

async def auto_guardar(parar: asyncio.Event, intervalo_s: int) -> None:
    """
    Tarea de fondo: guarda grilla.pkl cada intervalo_s segundos mientras
    la descarga está activa. Permite revisar el avance sin esperar al final.
    """
    while not parar.is_set():
        await asyncio.sleep(intervalo_s)
        if not parar.is_set():
            grilla.to_pickle(ruta_grilla, compression="gzip")


async def procesar_punto(
    session:  aiohttp.ClientSession,
    semaforo: asyncio.Semaphore,
    parar:    asyncio.Event,
    idx:      int,
    lat:      float,
    lon:      float,
) -> tuple[int, bool]:
    """
    Para un punto de la red vial:
      1. Consulta metadatos (petición gratuita): ¿hay imagen aquí?
      2. Si la hay y es de Google, descarga la imagen (petición con costo).

    El semáforo garantiza que no se supere MAX_CONCURRENT peticiones
    simultáneas, evitando errores OVER_QUERY_LIMIT de la API.

    Solo se aceptan imágenes cuyo campo 'copyright' contiene "oogle"
    (Google). Esto excluye imágenes de terceros (Mapillary, etc.) que
    aparecerían en la misma API pero con calidad o ángulo inconsistentes.

    Retorna (idx, tiene_imagen: bool).
    """
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
                    url_meta,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as r:
                    data   = await r.json(content_type=None)
                    estado = data.get("status", "UNKNOWN")

                if estado == "REQUEST_DENIED":
                    print(f"\n[ERROR] API denegada: {data.get('error_message', '')}")
                    parar.set()
                    return idx, False

                imagen_disponible = (
                    estado == "OK"
                    and "oogle" in data.get("copyright", "")
                )
                if not imagen_disponible:
                    return idx, False

                # Google ajusta las coordenadas al punto real de la foto;
                # las usamos como nombre de archivo para evitar duplicados
                # cuando dos puntos de la grilla caen en la misma imagen.
                lat_real   = data["location"]["lat"]
                lon_real   = data["location"]["lng"]
                img_nombre = f"{lat_real}_{lon_real}.jpg"
                ruta_img   = os.path.join(LOCALIDAD, img_nombre)

                if not os.path.exists(ruta_img):
                    async with session.get(
                        url_img,
                        timeout=aiohttp.ClientTimeout(total=30),
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

    # add_signal_handler integra Ctrl+C con el event loop de asyncio,
    # más limpio que signal.signal() que interrumpe el hilo principal.
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
    imagenes_total = int(grilla["imagen"].sum())  # incluye sesiones anteriores

    tarea_guardado = asyncio.create_task(auto_guardar(parar, GUARDAR_CADA_S))

    # TCPConnector limita las conexiones abiertas simultáneas al mismo valor
    # que el semáforo para no desperdiciar sockets
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    async with aiohttp.ClientSession(connector=connector) as session:
        tareas = [
            procesar_punto(session, semaforo, parar, idx, grilla.lat[idx], grilla.lon[idx])
            for idx in pendientes
        ]

        # as_completed devuelve cada resultado en cuanto termina,
        # permitiendo actualizar el progreso en tiempo real sin esperar
        # a que terminen todas las tareas.
        for coro in asyncio.as_completed(tareas):
            if parar.is_set():
                break

            idx, tiene_imagen = await coro

            if tiene_imagen:
                grilla.loc[idx, "imagen"] = True
                imagenes_total += 1

            procesados += 1
            sys.stdout.write(
                f"\r  Procesados: {procesados}/{len(pendientes)}"
                f"  |  Imágenes: {imagenes_total}"
            )
            sys.stdout.flush()

    tarea_guardado.cancel()
    grilla.to_pickle(ruta_grilla, compression="gzip")
    print(f"\n\nGrilla guardada en: {ruta_grilla}")
    print(f"Total imágenes: {grilla['imagen'].sum()} / {total}")


asyncio.run(main())


# =============================================================================
# PASO 5: Mapa de cobertura Street View
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 10))
gpd.GeoDataFrame(shp_bog.loc[[LOCALIDAD]]).plot(
    ax=ax, color="lightgray", edgecolor="black"
)
ax.scatter(
    grilla.loc[grilla["imagen"], "lon"],
    grilla.loc[grilla["imagen"], "lat"],
    s=2, color="steelblue", label="Con imagen",
)
ax.scatter(
    grilla.loc[~grilla["imagen"], "lon"],
    grilla.loc[~grilla["imagen"], "lat"],
    s=1, color="lightcoral", alpha=0.4, label="Sin imagen",
)
ax.set_title(f"Cobertura Street View — {LOCALIDAD}", fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(LOCALIDAD, "cobertura_mapa.png"), dpi=150)
plt.show()
print("Mapa guardado como 'cobertura_mapa.png'")

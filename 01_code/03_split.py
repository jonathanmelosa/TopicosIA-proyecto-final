#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_split.py  —  PASO 3 del pipeline
=====================================
Divide las imágenes en conjuntos de entrenamiento (70%), validación (15%)
y prueba (15%) usando un *split espacial por UPZ*.

Solo se incluyen fotos capturadas antes de 2021 (fecha_descargada < "2021"),
para que sean temporalmente consistentes con la Encuesta Multipropósito 2021.

¿Por qué split espacial y no aleatorio?
----------------------------------------
Con un split aleatorio, dos imágenes de la misma calle tomadas a 50 metros
de distancia podrían caer una en train y otra en test. Esas imágenes son
prácticamente idénticas: el modelo "recuerda" la escena y el R² se infla
artificialmente, sin que eso refleje capacidad de generalización real.

El split espacial resuelve esto asignando *todas* las imágenes de una UPZ
al mismo subconjunto. Como las UPZs son zonas geográficamente contiguas y
distintas, el modelo de prueba evalúa verdadera generalización a territorios
no vistos durante el entrenamiento.

Esto es análogo al uso de variación entre municipios para identificación
causal en econometría: se evita el problema de usar variación "dentro" de la
misma unidad (donde los observables son demasiado similares) y se usa variación
"entre" unidades espacialmente separadas.

Algoritmo de asignación de UPZs a splits
-----------------------------------------
Con solo ~7 UPZs en USME, la asignación por fracción exacta no es posible
(no podemos tomar el 70% de 7 UPZs y obtener 4.9 UPZs). El algoritmo:

  1. Ordena las UPZs por su tamaño (número de imágenes), de mayor a menor.
  2. Las asigna al split con menos imágenes acumuladas hasta completar las
     proporciones objetivo (greedy bin-packing).
  3. Garantiza que ningún split quede vacío.

Este enfoque minimiza el desequilibrio entre splits dado el número reducido
de UPZs.

REQUISITOS
----------
  pip install pandas scikit-learn matplotlib

ENTRADAS
--------
  00_data/processed/imagenes_dataset.csv   (generado por 01_cruce_geografico.py)

SALIDAS
-------
  00_data/processed/splits/train.csv
  00_data/processed/splits/val.csv
  00_data/processed/splits/test.csv
  02_outputs/figuras/distribucion_splits.png

USO
---
  python 01_code/03_split.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# RUTAS
# =============================================================================

_PROYECTO_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV   = _PROYECTO_ROOT / "00_data" / "processed" / "imagenes_dataset.csv"
SPLITS_DIR  = _PROYECTO_ROOT / "00_data" / "processed" / "splits"
FIGURAS_DIR = _PROYECTO_ROOT / "02_outputs" / "figuras"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PARÁMETROS
# =============================================================================

PROP_TRAIN = 0.70   # 70% entrenamiento
PROP_VAL   = 0.15   # 15% validación
PROP_TEST  = 0.15   # 15% prueba
SEED       = 42     # semilla para reproducibilidad en el shuffle de UPZs


# =============================================================================
# PASO 1: Cargar datos
# =============================================================================

print("=" * 60)
print("  PASO 3 — Split espacial por UPZ  (70 / 15 / 15)")
print("=" * 60)

if not INPUT_CSV.exists():
    print(f"\n[ERROR] No se encontró {INPUT_CSV}")
    print("  → Corre 01_code/01_cruce_geografico.py primero.")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV, dtype={"upz_cod": str, "upz_cod_shp": str})
print(f"\n  Total imágenes cargadas: {len(df):,}")

# Filtrar fotos previas a 2021 para consistencia temporal con la EM 2021.
# fecha_descargada tiene formato "YYYY-MM"; comparamos los 4 primeros caracteres.
n_sin_fecha = df["fecha_descargada"].isna().sum()
if n_sin_fecha:
    print(f"  [AVISO] {n_sin_fecha:,} imágenes sin fecha — se descartan.")
    df = df.dropna(subset=["fecha_descargada"])

df = df[df["fecha_descargada"].str[:4].astype(int) < 2021].copy()
print(f"  Imágenes con fecha < 2021: {len(df):,}")

# Excluir UPZs con cobertura insuficiente de imágenes.
UPZ_EXCLUIDAS = ["Parque Entrenubes", "Los Libertadores"]
mask_excluir = df["upz_nom"].str.contains("|".join(UPZ_EXCLUIDAS), case=False, na=False)
n_excluidas = mask_excluir.sum()
if n_excluidas:
    upzs_removidas = df.loc[mask_excluir, "upz_nom"].unique().tolist()
    print(f"  UPZs excluidas por baja cobertura: {upzs_removidas} ({n_excluidas:,} imágenes)")
    df = df[~mask_excluir].copy()

print(f"  Imágenes tras exclusiones: {len(df):,}")
print(f"  UPZs presentes           : {df['upz_cod'].nunique()}")

# Estadísticas por UPZ
upz_stats = df.groupby(["upz_cod", "upz_nom"]).agg(
    n_imagenes=("ruta_imagen", "count"),
    ipm_medio=("inc_ipm_pct", "mean"),
).reset_index().sort_values("n_imagenes", ascending=False)

print("\n  Distribución por UPZ:")
print(f"  {'UPZ':5s}  {'Nombre':30s}  {'N imgs':>8s}  {'IPM medio':>10s}")
print(f"  {'-'*5}  {'-'*30}  {'-'*8}  {'-'*10}")
for _, row in upz_stats.iterrows():
    print(f"  {str(row['upz_cod']):5s}  {str(row['upz_nom'])[:30]:30s}  "
          f"{row['n_imagenes']:8,d}  {row['ipm_medio']:10.2f}")


# =============================================================================
# PASO 2: Asignación greedy de UPZs a splits
# =============================================================================

print()
print("  Asignando UPZs a splits (greedy bin-packing)...")

# Barajar con semilla fija para reproducibilidad.
# Se baraja ANTES de aplicar greedy para que el resultado no dependa del
# orden original de las UPZs (que podría estar sesgado geográficamente).
rng       = np.random.default_rng(SEED)
upz_orden = upz_stats.sample(frac=1, random_state=SEED)  # shuffle reproducible

n_total = len(df)
objetivo = {
    "train": PROP_TRAIN * n_total,
    "val":   PROP_VAL   * n_total,
    "test":  PROP_TEST  * n_total,
}
acumulado  = {"train": 0, "val": 0, "test": 0}
asignacion = {}  # upz_cod → split

# Greedy: asignar cada UPZ al split con mayor déficit respecto a su objetivo
for _, row in upz_orden.iterrows():
    cod = row["upz_cod"]
    n   = row["n_imagenes"]

    deficit = {
        split: objetivo[split] - acumulado[split]
        for split in objetivo
    }
    split_elegido = max(deficit, key=deficit.get)
    asignacion[cod] = split_elegido
    acumulado[split_elegido] += n

print()
print("  Resultado de la asignación:")
for split in ("train", "val", "test"):
    upzs = [cod for cod, s in asignacion.items() if s == split]
    n    = acumulado[split]
    pct  = n / n_total * 100
    print(f"    {split:5s}: {n:7,d} imágenes ({pct:.1f}%)  |  UPZs: {upzs}")

# Verificar que ningún split quedó vacío
for split in ("train", "val", "test"):
    if acumulado[split] == 0:
        print(f"\n[ERROR] El split '{split}' quedó vacío. "
              f"No hay suficientes UPZs para el split 70/15/15.")
        print("  Considera relajar las proporciones o agregar más localidades.")
        sys.exit(1)


# =============================================================================
# PASO 3: Crear DataFrames de cada split
# =============================================================================

df["split"] = df["upz_cod"].astype(str).map(
    {str(k): v for k, v in asignacion.items()}
)

df_train = df[df["split"] == "train"].drop(columns=["split"])
df_val   = df[df["split"] == "val"  ].drop(columns=["split"])
df_test  = df[df["split"] == "test" ].drop(columns=["split"])

# Mezclar train para que los batches del DataLoader sean variados desde el
# primer epoch incluso si num_workers=0 o shuffle=False en el DataLoader.
df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
df_val   = df_val.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)


# =============================================================================
# PASO 4: Validar que la distribución de IPM sea comparable entre splits
# =============================================================================

print()
print("  Estadísticas de IPM por split:")
print(f"  {'Split':6s}  {'N':>7s}  {'Media':>7s}  {'Std':>6s}  {'Mín':>5s}  {'Máx':>5s}")
print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*5}")
for nombre, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
    print(
        f"  {nombre:6s}  {len(d):7,d}  "
        f"{d['inc_ipm_pct'].mean():7.2f}  {d['inc_ipm_pct'].std():6.2f}  "
        f"{d['inc_ipm_pct'].min():5.2f}  {d['inc_ipm_pct'].max():5.2f}"
    )

# AVISO si la distribución de IPM difiere mucho entre splits (señal de que
# las UPZs más pobres quedaron concentradas en un solo split).
medias  = [df_train["inc_ipm_pct"].mean(), df_val["inc_ipm_pct"].mean(), df_test["inc_ipm_pct"].mean()]
rango   = max(medias) - min(medias)
if rango > 10:
    print(f"\n  [AVISO] La media de IPM varía {rango:.1f} puntos entre splits.")
    print(f"  Con pocas UPZs es difícil garantizar distribuciones idénticas.")
    print(f"  Esto es esperado y no es un error. Tener en cuenta al interpretar")
    print(f"  las métricas de validación y prueba.")


# =============================================================================
# PASO 5: Guardar splits
# =============================================================================

df_train.to_csv(SPLITS_DIR / "train.csv", index=False)
df_val.to_csv(  SPLITS_DIR / "val.csv",   index=False)
df_test.to_csv( SPLITS_DIR / "test.csv",  index=False)

print()
print(f"  Splits guardados en {SPLITS_DIR}:")
print(f"    train.csv : {len(df_train):,d} filas")
print(f"    val.csv   : {len(df_val):,d}   filas")
print(f"    test.csv  : {len(df_test):,d}  filas")


# =============================================================================
# PASO 6: Figura — distribución de splits
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

colores = {"train": "steelblue", "val": "darkorange", "test": "seagreen"}
datos   = {"train": df_train, "val": df_val, "test": df_test}

for ax, (nombre, d) in zip(axes, datos.items()):
    ax.hist(d["inc_ipm_pct"], bins=20, color=colores[nombre], edgecolor="white", alpha=0.9)
    ax.axvline(d["inc_ipm_pct"].mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Media={d['inc_ipm_pct'].mean():.1f}")
    ax.set_title(f"{nombre.capitalize()} — {len(d):,d} imágenes", fontsize=12)
    ax.set_xlabel("IPM (%)")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)

# Mostrar qué UPZs fueron asignadas a cada split
for ax, (nombre, _) in zip(axes, datos.items()):
    upzs = [str(c) for c, s in asignacion.items() if s == nombre]
    ax.text(
        0.98, 0.97,
        f"UPZs: {', '.join(upzs)}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=7,
        color="gray",
    )

fig.suptitle(
    "Distribución del IPM por split (split espacial por UPZ)\n"
    "Encuesta Multipropósito 2021 — Bogotá, USME",
    fontsize=13,
)
plt.tight_layout()
ruta_fig = FIGURAS_DIR / "distribucion_splits.png"
plt.savefig(ruta_fig, dpi=150)
plt.close()

print()
print(f"  Figura guardada: {ruta_fig}")
print()
print("  Script completado. Siguiente paso:")
print("    python 01_code/05_entrenamiento.py")

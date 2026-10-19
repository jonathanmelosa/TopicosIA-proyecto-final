#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_entrenamiento.py  —  PASO 5 del pipeline
=============================================
Entrena PovertyNet (ResNet18 fine-tuning) para predecir el IPM a partir
de imágenes de Google Street View usando MSE como función de pérdida.

Técnicas de regularización implementadas
-----------------------------------------
  1. Weight Decay (L2): implementado directamente en el optimizador Adam
     (parámetro weight_decay). Equivalente a la regularización Ridge en
     regresión lineal: penaliza pesos grandes, distribuye la información
     entre muchas neuronas y reduce la dependencia de features individuales.
     J_R = J + (ω/2m)·Σ‖W[l]‖²_F

  2. Dropout: definido en la arquitectura (04_modelo.py). Desactiva
     aleatoriamente el 50% de las neuronas del penúltimo layer durante
     el entrenamiento, obligando al modelo a aprender representaciones
     redundantes y robustas.

  3. Early Stopping: monitorea la pérdida de validación cada época y
     detiene el entrenamiento cuando no mejora por PATIENCE épocas
     consecutivas. Guarda el modelo con la mejor pérdida de validación
     observada, no el del final del entrenamiento.

Optimizador Adam
-----------------
Adam (Adaptive Moment Estimation) es una versión mejorada del gradiente
descendiente con dos mejoras clave:
  - Momento de primer orden (m): promedio móvil del gradiente (suaviza
    la dirección de actualización).
  - Momento de segundo orden (v): promedio móvil del gradiente al cuadrado
    (adapta el learning rate por parámetro).
Actualización: W ← W - α · m̂ / (√v̂ + ε)

Esto lo hace más robusto que SGD para fine-tuning donde los gradientes
pueden ser muy dispares entre las capas congeladas y las libres.

Diferential Learning Rate
--------------------------
layer4 recibe un LR 10× menor que la capa de regresión. La razón:
layer4 ya tiene una buena inicialización (ImageNet) y necesita ajustarse
suavemente. La capa de regresión es nueva (inicialización aleatoria) y
necesita un LR mayor para converger rápidamente.

REQUISITOS
----------
  pip install torch torchvision pandas matplotlib tqdm

ENTRADAS
--------
  00_data/processed/splits/{train,val,test}.csv

SALIDAS
-------
  02_outputs/modelos/mejor_modelo.pt     → pesos del mejor modelo (val loss)
  02_outputs/modelos/ultimo_modelo.pt    → pesos del último epoch
  02_outputs/figuras/curvas_entrenamiento.png
  02_outputs/figuras/predicciones_test.png

USO
---
  python 01_code/05_entrenamiento.py
  python 01_code/05_entrenamiento.py --epochs 30 --batch 16 --lr 1e-4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import importlib.util

# MPS (Apple Silicon) tiene bugs numéricos con algunas operaciones de ResNet18.
# Este flag hace que las ops sin implementación nativa en MPS caigan a CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Carga de módulos con prefijos numéricos (Python no permite importarlos
# directamente con 'import 02_dataset', así que usamos importlib).
# ---------------------------------------------------------------------------

def _cargar_modulo(nombre_archivo: str):
    """Carga un módulo Python desde un archivo con prefijo numérico."""
    ruta = Path(__file__).resolve().parent / nombre_archivo
    spec = importlib.util.spec_from_file_location(nombre_archivo, ruta)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ds  = _cargar_modulo("02_dataset.py")
_mdl = _cargar_modulo("04_modelo.py")

StreetViewDataset = _ds.StreetViewDataset
crear_dataloaders = _ds.crear_dataloaders
PovertyNet        = _mdl.PovertyNet

# =============================================================================
# RUTAS
# =============================================================================

_PROYECTO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR     = _PROYECTO_ROOT / "00_data"     / "processed" / "splits"
MODELOS_DIR    = _PROYECTO_ROOT / "02_outputs"  / "modelos"
FIGURAS_DIR    = _PROYECTO_ROOT / "02_outputs"  / "figuras"

MODELOS_DIR.mkdir(parents=True, exist_ok=True)
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HIPERPARÁMETROS POR DEFECTO
# =============================================================================

DEFAULTS = {
    "epochs":       50,     # máximo de épocas (early stopping puede terminar antes)
    "batch_size":   32,     # imágenes por batch
    "lr":           1e-4,   # learning rate para la capa de regresión (reducido para evitar explosión)
    "lr_backbone":  1e-5,   # learning rate para layer4 (10× menor, ya preentrenada)
    "weight_decay": 1e-4,   # coeficiente de regularización L2 (ω en el diseño)
    "dropout_p":    0.5,    # probabilidad de dropout en la capa penúltima
    "patience":     7,      # épocas sin mejora para early stopping
    "num_workers":  0,      # 0 en macOS: num_workers>0 falla al serializar módulos cargados con importlib
    "seed":         42,
}


# =============================================================================
# FUNCIONES
# =============================================================================

def fijar_semilla(seed: int) -> None:
    """
    Fija la semilla en Python, NumPy y PyTorch para reproducibilidad.
    No garantiza reproducibilidad perfecta en GPU (operaciones no deterministas),
    pero sí entre distintas ejecuciones en CPU.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detectar_device(override: "str | None" = None) -> torch.device:
    """
    Detecta el dispositivo de cómputo disponible en orden de preferencia:
      1. CUDA (GPU NVIDIA) — óptimo para entrenamiento
      2. MPS  (Apple Silicon) — aceleración en Mac M1/M2/M3
      3. CPU  — fallback universal

    override: "cpu" | "mps" | "cuda" para forzar un dispositivo concreto.
    """
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def epoch_train(
    modelo:    PovertyNet,
    loader:    DataLoader,
    criterio:  nn.MSELoss,
    optimizer: torch.optim.Adam,
    device:    torch.device,
) -> float:
    """
    Ejecuta un epoch de entrenamiento completo y retorna el RMSE medio.

    Ciclo por batch:
      1. forward pass  : calcular predicciones
      2. calcular loss : MSE entre predicción e IPM real
      3. backward pass : calcular gradientes (backpropagation)
      4. actualizar    : optimizer.step() ajusta los pesos
      5. limpiar       : optimizer.zero_grad() resetea gradientes

    El RMSE (raíz del MSE) se reporta en lugar del MSE porque está en las
    mismas unidades que el IPM (puntos porcentuales), facilitando la
    interpretación económica del error.
    """
    modelo.train()
    # Todos los BatchNorm en eval: evita inestabilidad numérica en MPS cuando
    # las estadísticas del batch de Street View difieren de ImageNet.
    # Los pesos γ/β de layer4 siguen siendo entrenables; solo se fija running_mean/var.
    for m in modelo.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    total_loss = 0.0
    n_muestras = 0
    nan_batches = 0

    for imgs, ipms in loader:
        imgs = imgs.to(device, non_blocking=True)
        ipms = ipms.to(device, non_blocking=True)

        optimizer.zero_grad()

        preds = modelo(imgs)
        loss  = criterio(preds, ipms)

        if not torch.isfinite(loss):
            nan_batches += 1
            optimizer.zero_grad()
            continue

        loss.backward()

        # Cero a gradientes NaN/inf antes de clipear (workaround MPS)
        for p in modelo.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad.zero_()

        nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(imgs)
        n_muestras += len(imgs)

    if nan_batches:
        print(f"\n  [AVISO] {nan_batches} batches con loss no finita (omitidos)")

    if n_muestras == 0:
        return float("inf")
    return (total_loss / n_muestras) ** 0.5


@torch.no_grad()
def epoch_eval(
    modelo:   PovertyNet,
    loader:   DataLoader,
    criterio: nn.MSELoss,
    device:   torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evalúa el modelo en un loader sin actualizar pesos.

    Retorna (RMSE, predicciones, valores_reales) para poder calcular
    métricas adicionales (R², MAE) fuera de esta función.

    @torch.no_grad() desactiva el cálculo de gradientes, reduciendo el uso
    de memoria y acelerando la evaluación (~2× respecto a mode train).
    """
    modelo.eval()     # desactiva dropout, batchnorm en eval mode
    total_loss = 0.0
    n_muestras = 0
    preds_all  = []
    reals_all  = []

    for imgs, ipms in loader:
        imgs = imgs.to(device, non_blocking=True)
        ipms = ipms.to(device, non_blocking=True)

        preds = modelo(imgs)
        loss  = criterio(preds, ipms)

        if torch.isfinite(loss):
            total_loss += loss.item() * len(imgs)
            n_muestras += len(imgs)
        preds_all.append(preds.cpu().float().numpy())
        reals_all.append(ipms.cpu().float().numpy())

    preds_arr = np.concatenate(preds_all)
    reals_arr = np.concatenate(reals_all)

    # Filtrar predicciones no-finitas para que RMSE y R² sean coherentes
    mask = np.isfinite(preds_arr)
    preds_arr = preds_arr[mask]
    reals_arr = reals_arr[mask]

    rmse = (total_loss / n_muestras) ** 0.5 if n_muestras > 0 else float("inf")

    return rmse, preds_arr, reals_arr


def r2_score(preds: np.ndarray, reals: np.ndarray) -> float:
    """
    R² = 1 - SS_res / SS_tot.
    R² = 1 → predicción perfecta.
    R² = 0 → el modelo no es mejor que predecir la media.
    R² < 0 → el modelo es peor que predecir la media.
    """
    p = preds.astype(np.float64)
    r = reals.astype(np.float64)
    ss_res = ((r - p) ** 2).sum()
    ss_tot = ((r - r.mean()) ** 2).sum()
    if not np.isfinite(ss_res) or not np.isfinite(ss_tot) or ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# =============================================================================
# ENTRENAMIENTO PRINCIPAL
# =============================================================================

def entrenar(cfg: dict) -> None:
    """
    Ejecuta el loop completo de entrenamiento con early stopping.

    Parámetros
    ----------
    cfg : dict con las claves de DEFAULTS (hiperparámetros).
    """
    fijar_semilla(cfg["seed"])
    device = detectar_device(cfg.get("device"))
    print(f"  Dispositivo: {device}")

    # -------------------------------------------------------------------------
    # Datos
    # -------------------------------------------------------------------------
    train_loader, val_loader, test_loader = crear_dataloaders(
        splits_dir=SPLITS_DIR,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    print(f"  Train: {len(train_loader.dataset):,d} | "
          f"Val: {len(val_loader.dataset):,d} | "
          f"Test: {len(test_loader.dataset):,d}")

    # -------------------------------------------------------------------------
    # Sanity check: verificar NaN en datos y forward pass antes de entrenar
    # -------------------------------------------------------------------------
    print("\n  Sanity check del primer batch...")
    imgs_s, ipms_s = next(iter(train_loader))
    nan_imgs = torch.isnan(imgs_s).sum().item()
    nan_ipms = torch.isnan(ipms_s).sum().item()
    print(f"    Imágenes : shape={list(imgs_s.shape)}  NaN={nan_imgs}  "
          f"rango=[{imgs_s.min():.3f}, {imgs_s.max():.3f}]")
    print(f"    Targets  : NaN={nan_ipms}  valores={ipms_s.tolist()}")
    if nan_imgs > 0 or nan_ipms > 0:
        print("  [ERROR] NaN encontrado en los datos. Revisa las imágenes o los splits.")
        return

    # -------------------------------------------------------------------------
    # Modelo
    # -------------------------------------------------------------------------
    modelo = PovertyNet(dropout_p=cfg["dropout_p"], pretrained=True).to(device)
    print(f"  Parámetros entrenables: {modelo.n_params(True):,d} / {modelo.n_params():,d}")

    with torch.no_grad():
        out_s = modelo(imgs_s.to(device))
    nan_out = torch.isnan(out_s).sum().item()
    print(f"    Forward pass: NaN={nan_out}  rango=[{out_s.min():.3f}, {out_s.max():.3f}]")
    if nan_out > 0:
        print("  [ERROR] El modelo produce NaN en el forward pass — posible bug de MPS.")
        print("  Prueba: python3 01_code/05_entrenamiento.py  (con PYTORCH_ENABLE_MPS_FALLBACK=1)")
        return
    print("  Sanity check OK.\n")

    # -------------------------------------------------------------------------
    # Optimizador con differential learning rate
    # layer4 recibe lr 10× menor porque ya tiene buena inicialización de ImageNet.
    # La capa de regresión es nueva (inicialización aleatoria) → lr mayor.
    # -------------------------------------------------------------------------
    params = [
        {"params": modelo.features[-1].parameters(), "lr": cfg["lr_backbone"]},
        {"params": modelo.regressor.parameters(),    "lr": cfg["lr"]},
    ]
    optimizer = torch.optim.Adam(params, weight_decay=cfg["weight_decay"])

    # Learning rate scheduler: reduce el LR en un factor 0.5 si la val loss
    # no mejora en patience/2 épocas. Complementa el early stopping.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["patience"] // 2
    )

    criterio = nn.MSELoss()

    # -------------------------------------------------------------------------
    # Early stopping + registro de métricas
    # -------------------------------------------------------------------------
    mejor_val_rmse   = float("inf")
    epocas_sin_mejora = 0
    historial        = {"train_rmse": [], "val_rmse": [], "val_r2": []}

    ruta_mejor  = MODELOS_DIR / "mejor_modelo.pt"
    ruta_ultimo = MODELOS_DIR / "ultimo_modelo.pt"

    print(f"\n  Iniciando entrenamiento (max {cfg['epochs']} épocas, "
          f"patience={cfg['patience']})...")
    print(f"  {'Época':>5s}  {'Train RMSE':>11s}  {'Val RMSE':>10s}  "
          f"{'Val R²':>8s}  {'LR':>10s}  {'Estado':}")
    print(f"  {'-'*5}  {'-'*11}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*15}")

    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        train_rmse             = epoch_train(modelo, train_loader, criterio, optimizer, device)
        val_rmse, preds_v, rv  = epoch_eval(modelo,   val_loader,  criterio, device)
        val_r2                 = r2_score(preds_v, rv)

        historial["train_rmse"].append(train_rmse)
        historial["val_rmse"].append(val_rmse)
        historial["val_r2"].append(val_r2)

        scheduler.step(val_rmse)

        lr_actual = optimizer.param_groups[-1]["lr"]
        estado    = ""

        if val_rmse < mejor_val_rmse:
            mejor_val_rmse    = val_rmse
            epocas_sin_mejora = 0
            estado            = "✓ mejor"
            torch.save({
                "epoch":            epoch,
                "model_state_dict": modelo.state_dict(),
                "val_rmse":         val_rmse,
                "val_r2":           val_r2,
                "cfg":              cfg,
            }, ruta_mejor)
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= cfg["patience"]:
                estado = "STOP"

        print(
            f"  {epoch:5d}  {train_rmse:11.4f}  {val_rmse:10.4f}  "
            f"{val_r2:8.4f}  {lr_actual:10.2e}  {estado}"
        )

        # Guardar checkpoint del último epoch
        torch.save({
            "epoch":            epoch,
            "model_state_dict": modelo.state_dict(),
            "val_rmse":         val_rmse,
            "cfg":              cfg,
        }, ruta_ultimo)

        if epocas_sin_mejora >= cfg["patience"]:
            print(f"\n  [Early Stopping] Sin mejora por {cfg['patience']} épocas.")
            break

    t_total = time.time() - t0
    print(f"\n  Entrenamiento completado en {t_total/60:.1f} min")
    print(f"  Mejor val RMSE: {mejor_val_rmse:.4f}")

    # -------------------------------------------------------------------------
    # Evaluación final en test (con el mejor modelo)
    # -------------------------------------------------------------------------
    if not ruta_mejor.exists():
        print("\n[ERROR] No se guardó ningún checkpoint (val RMSE nunca mejoró).")
        print("  El entrenamiento divergió. Revisa los hiperparámetros.")
        return

    print("\n  Evaluando en test con el mejor modelo...")
    ckpt = torch.load(ruta_mejor, map_location=device)
    modelo.load_state_dict(ckpt["model_state_dict"])

    test_rmse, preds_t, reals_t = epoch_eval(modelo, test_loader, criterio, device)
    test_r2  = r2_score(preds_t, reals_t)
    test_mae = float(np.abs(preds_t - reals_t).mean())

    print(f"\n  {'Métrica':20s}  {'Valor':>10s}")
    print(f"  {'-'*20}  {'-'*10}")
    print(f"  {'RMSE (test)':20s}  {test_rmse:10.4f}")
    print(f"  {'MAE (test)':20s}  {test_mae:10.4f}")
    print(f"  {'R² (test)':20s}  {test_r2:10.4f}")

    # Guardar métricas como JSON para uso en 06_interpretabilidad.py
    metricas = {
        "mejor_epoch":    int(ckpt["epoch"]),
        "val_rmse":       float(mejor_val_rmse),
        "test_rmse":      float(test_rmse),
        "test_mae":       float(test_mae),
        "test_r2":        float(test_r2),
        "cfg":            cfg,
    }
    with open(MODELOS_DIR / "metricas.json", "w") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # Figura 1: Curvas de pérdida (train vs. val)
    # -------------------------------------------------------------------------
    epocas = range(1, len(historial["train_rmse"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(epocas, historial["train_rmse"], label="Train RMSE", color="steelblue")
    ax.plot(epocas, historial["val_rmse"],   label="Val RMSE",   color="darkorange")
    ax.axvline(ckpt["epoch"], color="green", linestyle="--", alpha=0.7,
               label=f"Mejor epoch ({ckpt['epoch']})")
    ax.set_xlabel("Época")
    ax.set_ylabel("RMSE (puntos de IPM)")
    ax.set_title("Curvas de pérdida")
    ax.legend()

    ax = axes[1]
    ax.plot(epocas, historial["val_r2"], color="seagreen")
    ax.axvline(ckpt["epoch"], color="green", linestyle="--", alpha=0.7)
    ax.set_xlabel("Época")
    ax.set_ylabel("R²")
    ax.set_title("R² en validación")

    fig.suptitle("Entrenamiento PovertyNet — USME, Bogotá", fontsize=13)
    plt.tight_layout()
    ruta_fig1 = FIGURAS_DIR / "curvas_entrenamiento.png"
    plt.savefig(ruta_fig1, dpi=150)
    plt.close()
    print(f"\n  Figura guardada: {ruta_fig1}")

    # -------------------------------------------------------------------------
    # Figura 2: Predicciones vs. valores reales en test
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(reals_t, preds_t, s=4, alpha=0.4, color="steelblue")

    lim = [min(reals_t.min(), preds_t.min()) - 1,
           max(reals_t.max(), preds_t.max()) + 1]
    ax.plot(lim, lim, color="red", linestyle="--", linewidth=1.5, label="Predicción perfecta")
    ax.set_xlabel("IPM real (%)")
    ax.set_ylabel("IPM predicho (%)")
    ax.set_title(
        f"Predicciones vs. reales — test\n"
        f"RMSE={test_rmse:.2f}  MAE={test_mae:.2f}  R²={test_r2:.3f}"
    )
    ax.legend()
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    plt.tight_layout()
    ruta_fig2 = FIGURAS_DIR / "predicciones_test.png"
    plt.savefig(ruta_fig2, dpi=150)
    plt.close()
    print(f"  Figura guardada: {ruta_fig2}")
    print()
    print("  Script completado. Siguiente paso:")
    print("    python 01_code/06_interpretabilidad.py")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar PovertyNet")
    parser.add_argument("--epochs",       type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch",        type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--lr",           type=float, default=DEFAULTS["lr"])
    parser.add_argument("--lr_backbone",  type=float, default=DEFAULTS["lr_backbone"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--dropout",      type=float, default=DEFAULTS["dropout_p"])
    parser.add_argument("--patience",     type=int,   default=DEFAULTS["patience"])
    parser.add_argument("--workers",      type=int,   default=DEFAULTS["num_workers"])
    parser.add_argument("--seed",         type=int,   default=DEFAULTS["seed"])
    parser.add_argument("--device",       type=str,   default=None,
                        help="Forzar dispositivo: cpu | mps | cuda")
    args = parser.parse_args()

    cfg = {
        "epochs":       args.epochs,
        "batch_size":   args.batch,
        "lr":           args.lr,
        "lr_backbone":  args.lr_backbone,
        "weight_decay": args.weight_decay,
        "dropout_p":    args.dropout,
        "patience":     args.patience,
        "num_workers":  args.workers,
        "seed":         args.seed,
        "device":       args.device,
    }

    print("=" * 60)
    print("  PASO 5 — Entrenamiento de PovertyNet")
    print("=" * 60)
    print("\n  Hiperparámetros:")
    for k, v in cfg.items():
        print(f"    {k:15s}: {v}")
    print()

    entrenar(cfg)

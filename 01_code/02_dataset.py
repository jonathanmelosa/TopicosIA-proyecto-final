#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_dataset.py  —  PASO 2 del pipeline
======================================
Define la clase StreetViewDataset (torch.utils.data.Dataset) y las
transformaciones de preprocesamiento y data augmentation usadas en todo
el pipeline.

Este módulo NO corre directamente; es importado por 05_entrenamiento.py
y 06_interpretabilidad.py. Se puede ejecutar como script para verificar
que las imágenes se cargan y transforman correctamente.

Decisiones de diseño
---------------------
  1. Normalización ImageNet: ResNet18 fue preentrenada con la distribución
     de ImageNet (µ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]). Usar la
     misma normalización es obligatorio para que el transfer learning funcione:
     si los valores de entrada no están en la escala esperada, los pesos
     preentrenados producen activaciones fuera de rango y el fine-tuning
     diverge o converge muy lentamente.

  2. Data augmentation solo en train: las transformaciones aleatorias aumentan
     la variedad de ejemplos durante el entrenamiento para reducir el
     sobreajuste. En validación y prueba se aplican solo el resize y la
     normalización para que las métricas sean reproducibles y comparables.

  3. Augmentations elegidas:
     - RandomHorizontalFlip: la pobreza de una calle no depende de si la
       escena está "al derecho" o "al revés". Añade ejemplos sin costo.
     - ColorJitter (brillo, contraste, saturación, matiz): las fotos de Street
       View varían mucho según la hora del día y condiciones climáticas.
       El modelo no debe aprender a asociar "tarde nublado" con "pobreza alta".
     - RandomRotation (±5°): pequeñas inclinaciones de la cámara no deberían
       afectar la predicción. Se limita a ±5° para no distorsionar la
       perspectiva arquitectónica que sí es informativa.
     - RandomResizedCrop: simula diferentes distancias al sujeto. Permite que
       el modelo aprenda de detalles locales (estado de fachadas) y contexto
       global (tipo de calle) por igual.

  4. Tamaño de imagen 224×224: es el tamaño de entrada estándar de ResNet18.
     Las imágenes de Street View son 640×640; el resize las reduce al tamaño
     correcto antes de la normalización.

REQUISITOS
----------
  pip install torch torchvision Pillow pandas

USO COMO SCRIPT (verificación)
-------------------------------
  python 01_code/02_dataset.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# =============================================================================
# CONSTANTES DE NORMALIZACIÓN ImageNet
# Estas estadísticas fueron calculadas sobre los 1.2 M de imágenes de ImageNet.
# Usarlas garantiza que las activaciones de ResNet18 preentrenada estén en
# el rango correcto desde el primer paso del fine-tuning.
# =============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224  # píxeles — tamaño de entrada estándar de ResNet18


# =============================================================================
# TRANSFORMACIONES
# =============================================================================

def get_transforms(modo: str) -> transforms.Compose:
    """
    Retorna el pipeline de transformaciones apropiado para cada modo.

    Parámetros
    ----------
    modo : "train" | "val" | "test"
        En "train" se aplican augmentations aleatorias.
        En "val" y "test" solo resize + normalización para reproducibilidad.

    Retorna
    -------
    torchvision.transforms.Compose con la secuencia de transformaciones.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if modo == "train":
        return transforms.Compose([
            # Recorte aleatorio entre el 80%-100% del área y resize a 224×224.
            # Simula variaciones en la distancia de la cámara al sujeto.
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),

            # Reflejo horizontal aleatorio con probabilidad 0.5.
            # La pobreza de una calle es invariante a la dirección de la escena.
            transforms.RandomHorizontalFlip(p=0.5),

            # Variaciones de color: simula diferentes condiciones de iluminación
            # (mañana, tarde, nublado, soleado) que cambian la apariencia de la
            # misma calle sin cambiar su nivel socioeconómico.
            transforms.ColorJitter(
                brightness=0.3,   # ±30% de brillo
                contrast=0.3,     # ±30% de contraste
                saturation=0.2,   # ±20% de saturación
                hue=0.05,         # ±5% de matiz (cambio pequeño para no invertir colores)
            ),

            # Rotación máxima ±5°: compensa inclinaciones leves de la cámara.
            # Más de ±10° distorsionaría la perspectiva arquitectónica,
            # que sí es informativa del nivel socioeconómico.
            transforms.RandomRotation(degrees=5),

            # Conversión a tensor float32 [0,1] y transposición a (C, H, W)
            transforms.ToTensor(),

            # Normalización con estadísticas ImageNet: transforma [0,1] a
            # valores centrados en 0 con desviación ~1, en el rango
            # aproximado [-2.5, 2.5] (igual que los datos de preentrenamiento).
            normalize,
        ])

    else:  # "val" o "test"
        return transforms.Compose([
            # Resize a 256 primero y luego crop central a 224: práctica estándar
            # en evaluación de modelos ImageNet. Conserva más área que crop directo.
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ])


# =============================================================================
# DATASET
# =============================================================================

class StreetViewDataset(Dataset):
    """
    Dataset de imágenes de Google Street View con IPM como variable objetivo.

    Cada elemento del dataset es una tupla (imagen_tensor, ipm_float):
      - imagen_tensor : torch.FloatTensor de forma (3, 224, 224), normalizado
                        con media y desviación estándar de ImageNet.
      - ipm_float     : torch.FloatTensor escalar con el IPM de la UPZ
                        donde fue tomada la fotografía.

    Parámetros
    ----------
    df    : pd.DataFrame con columnas 'ruta_imagen' e 'inc_ipm_pct'.
    modo  : "train" | "val" | "test" — determina las transformaciones.
    """

    def __init__(self, df: pd.DataFrame, modo: str = "val") -> None:
        self.df         = df.reset_index(drop=True)
        self.transform  = get_transforms(modo)
        self.modo       = modo

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fila  = self.df.iloc[idx]
        ruta  = fila["ruta_imagen"]
        ipm   = float(fila["inc_ipm_pct"])

        # Abrir imagen en RGB (Street View guarda en JPEG, modo RGB)
        img = Image.open(ruta).convert("RGB")

        # Aplicar transformaciones (augmentation en train, resize+norm en eval)
        img_tensor = self.transform(img)

        # El IPM se convierte a tensor escalar para que sea compatible con
        # la función de pérdida MSELoss que espera tensores de la misma forma.
        ipm_tensor = torch.tensor(ipm, dtype=torch.float32)

        return img_tensor, ipm_tensor

    def __repr__(self) -> str:
        return (
            f"StreetViewDataset(modo='{self.modo}', "
            f"n={len(self)}, "
            f"ipm_mean={self.df['inc_ipm_pct'].mean():.2f})"
        )


# =============================================================================
# FUNCIÓN AUXILIAR: crear DataLoaders para train/val/test
# =============================================================================

def crear_dataloaders(
    splits_dir:  Path,
    batch_size:  int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Lee los CSVs de split (train.csv, val.csv, test.csv) y retorna tres
    DataLoaders listos para usar en el entrenamiento.

    Parámetros
    ----------
    splits_dir  : directorio con train.csv, val.csv, test.csv
    batch_size  : imágenes por batch (32 es un buen punto de partida)
    num_workers : procesos paralelos para carga de imágenes
    pin_memory  : acelera la transferencia CPU → GPU cuando se usa CUDA

    Retorna
    -------
    (train_loader, val_loader, test_loader)

    Notas sobre batch_size
    ----------------------
    32 es el valor estándar para fine-tuning de ResNet18 en GPUs de 8–16 GB.
    Con GPUs más pequeñas (4 GB), reducir a 16.
    Con varias GPUs, aumentar proporcionalmente.
    El gradiente se promedía sobre el batch, así que batches más grandes
    producen gradientes más suaves pero requieren más memoria.
    """
    df_train = pd.read_csv(splits_dir / "train.csv")
    df_val   = pd.read_csv(splits_dir / "val.csv")
    df_test  = pd.read_csv(splits_dir / "test.csv")

    ds_train = StreetViewDataset(df_train, modo="train")
    ds_val   = StreetViewDataset(df_val,   modo="val")
    ds_test  = StreetViewDataset(df_test,  modo="test")

    # pin_memory solo tiene utilidad con CUDA; en MPS no está soportado.
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# VERIFICACIÓN — ejecutar este script directamente para validar el pipeline
# =============================================================================

if __name__ == "__main__":
    _PROYECTO_ROOT = Path(__file__).resolve().parent.parent
    SPLITS_DIR     = _PROYECTO_ROOT / "00_data" / "processed" / "splits"
    FIGURAS_DIR    = _PROYECTO_ROOT / "02_outputs" / "figuras"
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  VERIFICACIÓN DEL DATASET")
    print("=" * 60)

    for split in ("train", "val", "test"):
        csv = SPLITS_DIR / f"{split}.csv"
        if not csv.exists():
            print(f"  [AVISO] {csv.name} no encontrado. Corre 03_split.py primero.")
            continue

        df   = pd.read_csv(csv)
        ds   = StreetViewDataset(df, modo=split)
        print(f"\n  {split:5s}: {len(ds):>6,} imágenes | {ds}")

        # Cargar un batch de muestra
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        imgs, ipms = next(iter(loader))

        print(f"         Tensor shape : {imgs.shape}  dtype={imgs.dtype}")
        print(f"         IPM muestra  : {ipms.tolist()}")
        print(f"         Pixel range  : [{imgs.min():.3f}, {imgs.max():.3f}]")

    # Visualizar un batch de train con augmentation
    csv_train = SPLITS_DIR / "train.csv"
    if csv_train.exists():
        df_train = pd.read_csv(csv_train)
        ds_train = StreetViewDataset(df_train, modo="train")
        loader   = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=0)
        imgs, ipms = next(iter(loader))

        # Desnormalizar para visualización
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        imgs_vis = (imgs * std + mean).clamp(0, 1)

        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for i, ax in enumerate(axes.flat):
            if i >= len(imgs_vis):
                ax.axis("off")
                continue
            img_np = imgs_vis[i].permute(1, 2, 0).numpy()
            ax.imshow(img_np)
            ax.set_title(f"IPM = {ipms[i].item():.1f}", fontsize=9)
            ax.axis("off")

        fig.suptitle("Muestra de imágenes con augmentation (train)", fontsize=13)
        plt.tight_layout()
        ruta_fig = FIGURAS_DIR / "muestra_augmentation.png"
        plt.savefig(ruta_fig, dpi=120)
        plt.close()
        print(f"\n  Muestra guardada en: {ruta_fig}")

    print("\n  Verificación completada.")

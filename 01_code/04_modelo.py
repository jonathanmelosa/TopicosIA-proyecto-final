#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_modelo.py  —  PASO 4 del pipeline
======================================
Define la arquitectura del modelo: ResNet18 preentrenada con ImageNet,
adaptada para regresión del IPM mediante fine-tuning.

Este módulo es importado por 05_entrenamiento.py y 06_interpretabilidad.py.
Se puede ejecutar directamente para inspeccionar la arquitectura.

¿Qué es ResNet18?
-----------------
ResNet18 es una red neuronal convolucional con 18 capas de peso (weight layers).
Fue diseñada por He et al. (2015) para superar el problema del vanishing gradient
en redes muy profundas mediante conexiones residuales (shortcut connections).

Estructura de alto nivel:
  conv1 → bn1 → relu → maxpool         (stem: extrae features de bajo nivel)
  layer1 (2 bloques residuales)         (features: bordes, texturas)
  layer2 (2 bloques residuales)         (features: formas simples)
  layer3 (2 bloques residuales)         (features: partes de objetos)
  layer4 (2 bloques residuales)         (features: objetos, escenas)
  avgpool → fc                          (clasificación/regresión)

Cada bloque residual tiene la forma:
  salida = ReLU(z[l] + x)   donde z[l] = W2·ReLU(W1·x + b1) + b2
La conexión residual (+x) permite que el gradiente fluya directamente
hacia capas anteriores, resolviendo el vanishing gradient.

Estrategia de fine-tuning
--------------------------
Congelamos layers 1–3 (features de bajo nivel: bordes, texturas, colores).
Esas representaciones son universales y ya son correctas gracias a ImageNet.

Dejamos libres layer4 y el nuevo fc (features de alto nivel: objetos, escenas).
Esos son los que necesitan adaptarse a las particularidades visuales de las
calles de Bogotá para predecir pobreza.

Por qué no congelar todo o descongelar todo
--------------------------------------------
- Congelar todo: la red no puede aprender nada específico de USME. Solo la
  capa final se actualiza, lo que suele ser insuficiente para regresión.
- Descongelar todo: con ~25k imágenes el modelo tiene riesgo de overfitting
  en las capas profundas. Además, el entrenamiento es más lento y costoso.
- Congelar capas 1–3, descongelar layer4+fc: equilibrio óptimo entre
  aprovechar el preentrenamiento y adaptar las representaciones de alto nivel.

Capa de salida
--------------
La fc original de ResNet18 produce 1,000 probabilidades (ImageNet tiene 1,000
clases). La reemplazamos con una nueva fc que produce UN ÚNICO número real:
el IPM predicho. La arquitectura es:

  avgpool_features (512 dims) → Dropout(0.5) → Linear(512 → 1)

El Dropout antes de la capa de salida reduce el sobreajuste distribuyendo
la información entre todos los 512 features en lugar de depender de unos pocos.

REQUISITOS
----------
  pip install torch torchvision

USO COMO SCRIPT (inspección de arquitectura)
---------------------------------------------
  python 01_code/04_modelo.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# =============================================================================
# ARQUITECTURA
# =============================================================================

class PovertyNet(nn.Module):
    """
    ResNet18 con transfer learning para regresión del IPM.

    Capas congeladas  : conv1, bn1, layer1, layer2, layer3
    Capas entrenables : layer4, avgpool, nuevo fc

    La capa de salida reemplaza la original de 1000 clases por una de 1
    neurona (regresión escalar → IPM predicho).

    Parámetros
    ----------
    dropout_p    : probabilidad de dropout antes de la capa final [0, 1)
    pretrained   : si True, carga los pesos preentrenados en ImageNet
    """

    def __init__(self, dropout_p: float = 0.5, pretrained: bool = True) -> None:
        super().__init__()

        # Cargar ResNet18 con pesos ImageNet (DEFAULT = los mejores disponibles)
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base    = models.resnet18(weights=weights)

        # -----------------------------------------------------------------
        # Extraer el backbone (todo excepto la capa de clasificación final)
        # avgpool produce un vector de 512 dimensiones por imagen.
        # -----------------------------------------------------------------
        self.features = nn.Sequential(
            base.conv1,    # (B, 3,   224, 224) → (B, 64,  112, 112)
            base.bn1,
            base.relu,
            base.maxpool,  # (B, 64,  112, 112) → (B, 64,   56,  56)
            base.layer1,   # (B, 64,   56,  56) → (B, 64,   56,  56)
            base.layer2,   # (B, 64,   56,  56) → (B, 128,  28,  28)
            base.layer3,   # (B, 128,  28,  28) → (B, 256,  14,  14)
            base.layer4,   # (B, 256,  14,  14) → (B, 512,   7,   7)
        )
        self.avgpool = base.avgpool  # (B, 512, 7, 7) → (B, 512, 1, 1)

        # Nueva capa de regresión: Dropout + Linear(512 → 1)
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(base.fc.in_features, 1),  # in_features = 512 en ResNet18
        )

        # -----------------------------------------------------------------
        # Congelar capas de bajo nivel (features universales de ImageNet).
        # Estas capas detectan bordes, texturas y formas simples que son
        # igualmente útiles para cualquier tarea de visión.
        # -----------------------------------------------------------------
        capas_a_congelar = [
            base.conv1, base.bn1,
            base.layer1, base.layer2, base.layer3,
        ]
        for capa in capas_a_congelar:
            for param in capa.parameters():
                param.requires_grad = False

        # layer4 y regressor quedan con requires_grad=True (valor por defecto)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante.

        Parámetros
        ----------
        x : torch.Tensor de forma (B, 3, 224, 224) normalizado con ImageNet stats.

        Retorna
        -------
        torch.Tensor de forma (B,) con los IPM predichos (valores continuos).

        El squeeze(-1) convierte (B, 1) → (B,) para que sea compatible con
        MSELoss que espera tensores de la misma forma que los targets.
        """
        feats = self.features(x)                     # (B, 512, 7, 7)
        feats = self.avgpool(feats)                   # (B, 512, 1, 1)
        feats = torch.flatten(feats, start_dim=1)     # (B, 512)
        pred  = self.regressor(feats)                 # (B, 1)
        return pred.squeeze(-1)                       # (B,)

    def capas_entrenables(self) -> list[str]:
        """Lista los nombres de los parámetros con requires_grad=True."""
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def n_params(self, solo_entrenables: bool = False) -> int:
        """
        Cuenta el total de parámetros del modelo.

        Parámetros
        ----------
        solo_entrenables : si True, solo cuenta los que tienen requires_grad=True.
        """
        return sum(
            p.numel() for p in self.parameters()
            if (not solo_entrenables or p.requires_grad)
        )


# =============================================================================
# FUNCIÓN AUXILIAR: cargar modelo desde checkpoint
# =============================================================================

def cargar_modelo(
    ruta_checkpoint: Path,
    device:          torch.device,
    dropout_p:       float = 0.5,
) -> PovertyNet:
    """
    Instancia PovertyNet y carga los pesos guardados en ruta_checkpoint.

    Parámetros
    ----------
    ruta_checkpoint : ruta al archivo .pt guardado por 05_entrenamiento.py
    device          : dispositivo donde cargar el modelo (cpu / cuda / mps)
    dropout_p       : debe coincidir con el valor usado durante el entrenamiento

    Retorna
    -------
    PovertyNet en modo eval() cargado en device.
    """
    modelo = PovertyNet(dropout_p=dropout_p, pretrained=False)
    state  = torch.load(ruta_checkpoint, map_location=device)
    modelo.load_state_dict(state["model_state_dict"])
    modelo.to(device)
    modelo.eval()
    return modelo


# =============================================================================
# INSPECCIÓN — ejecutar este script directamente para ver la arquitectura
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  ARQUITECTURA: PovertyNet (ResNet18 fine-tuning)")
    print("=" * 60)

    modelo = PovertyNet(dropout_p=0.5, pretrained=True)

    print(f"\n  Parámetros totales      : {modelo.n_params():>12,d}")
    print(f"  Parámetros entrenables  : {modelo.n_params(True):>12,d}")
    print(f"  Parámetros congelados   : {modelo.n_params() - modelo.n_params(True):>12,d}")

    print(f"\n  Porcentaje entrenable   : "
          f"{modelo.n_params(True) / modelo.n_params() * 100:.1f}%")

    print(f"\n  Capas entrenables ({len(modelo.capas_entrenables())} parámetros):")
    for name in modelo.capas_entrenables():
        p = dict(modelo.named_parameters())[name]
        print(f"    {name:55s}  shape={list(p.shape)}")

    # Prueba de forward pass con un batch aleatorio
    print("\n  Prueba de forward pass...")
    x      = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        pred = modelo(x)
    print(f"  Input shape  : {list(x.shape)}")
    print(f"  Output shape : {list(pred.shape)}")
    print(f"  Output values: {pred.tolist()}")
    print("\n  [OK] El modelo funciona correctamente.")

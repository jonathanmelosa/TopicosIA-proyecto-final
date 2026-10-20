#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_interpretabilidad.py  —  PASO 6 del pipeline
================================================
Explica *qué* aprendió PovertyNet y *por qué* predice lo que predice,
usando dos técnicas complementarias:

  1. Grad-CAM  — ¿En qué parte de la imagen se fijó el modelo?
  2. SHAP      — ¿Cuánto contribuyó cada feature a la predicción?

Grad-CAM (Gradient-weighted Class Activation Mapping)
-------------------------------------------------------
Calcula el gradiente de la predicción (IPM) respecto a los mapas de
activación de la última capa convolucional (layer4 de ResNet18).

  αₖ = (1/Z) · ΣᵢΣⱼ (∂ŷ / ∂Aᵏᵢⱼ)    ← importancia del mapa k
  Mapa = ReLU(Σₖ αₖ · Aᵏ)            ← combinación ponderada

αₖ responde: "¿cuánto movería la predicción de IPM si la activación del
mapa k aumentara una unidad?". Un αₖ grande significa que ese mapa k
detecta algo importante para predecir pobreza.

El mapa resultante es un heatmap (224×224) que se superpone a la imagen
original. Las zonas calientes (rojo) son las que el modelo consideró más
relevantes para su predicción.

Validación económica: si el modelo aprendió algo sensato, el heatmap
debería activarse sobre fachadas, pavimento e infraestructura — no sobre
el cielo o marcas de agua.

SHAP (SHapley Additive Explanations) con GradientExplainer
-----------------------------------------------------------
Los valores SHAP provienen de la teoría de juegos cooperativos y son el
único concepto de solución que satisface eficiencia, simetría, nulidad y
aditividad (Riascos, PDF de Interpretabilidad).

  φᵢ(f,x) = Σ_{S ⊆ N\{i}} [|S|!(|N|-|S|-1)!/|N|!] · [f(S∪{i}) - f(S)]

Interpretan: φᵢ es la contribución marginal promedio del feature i a la
predicción, considerando todos los órdenes posibles de entrada.

Para redes neuronales usamos GradientExplainer (captum), que aproxima
los valores SHAP usando los gradientes del modelo: más eficiente que el
cálculo exacto O(2^N) y suficientemente preciso para redes profundas.

Los valores SHAP se calculan a nivel de superpíxel (regiones conectadas
de la imagen), lo que facilita la interpretación visual.

REQUISITOS
----------
  pip install torch torchvision matplotlib numpy pandas Pillow captum tqdm

  Si captum no está disponible, el análisis SHAP se omite graciosamente.

ENTRADAS
--------
  02_outputs/modelos/mejor_modelo.pt     → pesos del modelo entrenado
  00_data/processed/splits/test.csv      → imágenes de test

SALIDAS
-------
  02_outputs/figuras/gradcam_muestra.png → grid de Grad-CAM (8 imágenes)
  02_outputs/figuras/shap_muestra.png    → SHAP overlay (4 imágenes)

USO
---
  python 01_code/06_interpretabilidad.py
  python 01_code/06_interpretabilidad.py --n_gradcam 16 --n_shap 4
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Carga de módulos con prefijos numéricos
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
get_transforms    = _ds.get_transforms
IMAGENET_MEAN     = _ds.IMAGENET_MEAN
IMAGENET_STD      = _ds.IMAGENET_STD
PovertyNet        = _mdl.PovertyNet
cargar_modelo     = _mdl.cargar_modelo

# =============================================================================
# RUTAS
# =============================================================================

_PROYECTO_ROOT  = Path(__file__).resolve().parent.parent
MODELOS_DIR     = _PROYECTO_ROOT / "02_outputs" / "modelos"
FIGURAS_DIR     = _PROYECTO_ROOT / "02_outputs" / "figuras"
SPLITS_DIR      = _PROYECTO_ROOT / "00_data"    / "processed" / "splits"

FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# GRAD-CAM
# =============================================================================

class GradCAM:
    """
    Implementación de Grad-CAM para la última capa convolucional de ResNet18.

    La implementación usa hooks de PyTorch para:
      1. Capturar las activaciones del último bloque convolucional (forward hook).
      2. Capturar los gradientes respecto a esas activaciones (backward hook).

    Con esos dos tensores se calcula el mapa de calor según la fórmula:
      αₖ = (1/Z) · ΣᵢΣⱼ (∂ŷ / ∂Aᵏᵢⱼ)   ← promedio global de los gradientes
      Mapa = ReLU(Σₖ αₖ · Aᵏ)             ← ponderación + ReLU

    La ReLU en el último paso descarta activaciones negativas (que indicarían
    que la región REDUCE el IPM predicho) porque nos interesa identificar las
    regiones que AUMENTAN la predicción, no las que la reducen.

    Parámetros
    ----------
    modelo : PovertyNet — debe estar en modo eval() para que el Dropout esté
             desactivado durante el análisis (resultados deterministas).
    """

    def __init__(self, modelo: PovertyNet) -> None:
        self.modelo     = modelo
        self._activs    = None   # activaciones del forward pass
        self._grads     = None   # gradientes del backward pass

        # La última capa convolucional de ResNet18 es el cuarto bloque de layer4.
        # features[-1] es el módulo Sequential de ResNet18, su último elemento
        # es layer4, y el último BasicBlock de layer4 es el objetivo.
        target_layer = modelo.features[-1][-1]

        # Forward hook: captura la salida de la capa objetivo durante el forward pass.
        self._fwd_hook = target_layer.register_forward_hook(
            lambda m, inp, out: setattr(self, "_activs", out)
        )
        # Backward hook: captura los gradientes de la capa objetivo.
        self._bwd_hook = target_layer.register_full_backward_hook(
            lambda m, grad_in, grad_out: setattr(self, "_grads", grad_out[0])
        )

    def __call__(
        self,
        img_tensor: torch.Tensor,
        device:     torch.device,
    ) -> tuple[np.ndarray, float]:
        """
        Calcula el mapa Grad-CAM para una imagen.

        Parámetros
        ----------
        img_tensor : tensor (1, 3, 224, 224) normalizado con ImageNet stats.
        device     : dispositivo de cómputo.

        Retorna
        -------
        (heatmap_norm, pred_ipm)
          heatmap_norm : np.ndarray (224, 224) con valores en [0, 1]
          pred_ipm     : float con la predicción del IPM
        """
        self.modelo.eval()
        img_tensor = img_tensor.unsqueeze(0).to(device).requires_grad_(True)

        # Forward pass con gradientes habilitados
        pred = self.modelo(img_tensor)  # escalar

        # Backward pass respecto a la predicción
        self.modelo.zero_grad()
        pred.backward()

        # αₖ = promedio global espacial de los gradientes (mean sobre H×W)
        # Shape de _grads: (1, 512, 7, 7) → alpha: (512,)
        alpha = self._grads.squeeze(0).mean(dim=(1, 2))  # (512,)

        # Combinar: Σₖ αₖ · Aᵏ
        # Shape de _activs: (1, 512, 7, 7)
        activs   = self._activs.squeeze(0)   # (512, 7, 7)
        heatmap  = (alpha[:, None, None] * activs).sum(dim=0)   # (7, 7)
        heatmap  = F.relu(heatmap)            # descartar contribuciones negativas

        # Normalizar al rango [0, 1] e interpolar al tamaño original (224×224)
        heatmap  = heatmap.detach().cpu()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_np = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        return heatmap_np, float(pred.item())

    def limpiar(self) -> None:
        """Eliminar los hooks para liberar memoria."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def desnormalizar(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Invierte la normalización ImageNet para visualización.
    Retorna un np.ndarray (H, W, 3) en [0, 1].
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (img_tensor * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def superponer_heatmap(
    img_np:      np.ndarray,
    heatmap:     np.ndarray,
    alpha:       float = 0.45,
) -> np.ndarray:
    """
    Superpone el heatmap de Grad-CAM sobre la imagen original.

    Parámetros
    ----------
    img_np  : np.ndarray (H, W, 3) en [0, 1]
    heatmap : np.ndarray (H, W) en [0, 1]
    alpha   : opacidad del heatmap (0 = invisible, 1 = solo heatmap)

    Retorna
    -------
    np.ndarray (H, W, 3) en [0, 1] con la superposición
    """
    colormap   = cm.jet(heatmap)[:, :, :3]    # mapa de color JET: azul→rojo
    overlay    = (1 - alpha) * img_np + alpha * colormap
    return np.clip(overlay, 0, 1)


# =============================================================================
# SHAP con GradientExplainer (captum)
# =============================================================================

def calcular_shap(
    modelo:         PovertyNet,
    imgs_tensor:    torch.Tensor,
    baseline:       torch.Tensor,
    device:         torch.device,
    n_steps:        int = 50,
) -> np.ndarray:
    """
    Calcula atribuciones con Integrated Gradients (aproximación a SHAP).

    Integrated Gradients atribuye la diferencia entre la predicción con la
    imagen real y la predicción con una imagen de referencia (baseline, p.ej.
    imagen negra o ruido) al efecto de cada píxel:

      IG_i(x) = (xᵢ - x'ᵢ) · ∫₀¹ (∂f(x' + α·(x-x')) / ∂xᵢ) dα

    La integral se aproxima con una interpolación lineal de n_steps pasos.
    El resultado es un tensor de la misma forma que la imagen: un valor
    de atribución por píxel por canal de color.

    Parámetros
    ----------
    modelo       : PovertyNet en eval() cargada en device
    imgs_tensor  : (N, 3, 224, 224) imágenes a explicar
    baseline     : (1, 3, 224, 224) imagen de referencia (imagen negra por defecto)
    device       : dispositivo de cómputo
    n_steps      : puntos de interpolación (mayor = más preciso, más lento)

    Retorna
    -------
    np.ndarray (N, 224, 224) con las atribuciones absolutas por píxel,
    promediadas sobre los 3 canales de color.
    """
    modelo.eval()
    imgs_tensor = imgs_tensor.to(device)
    baseline    = baseline.to(device)

    # Construir la secuencia de interpolaciones entre baseline e imagen real:
    # x(α) = baseline + α · (imagen - baseline),  α ∈ [0, 1]
    alphas       = torch.linspace(0, 1, n_steps, device=device)  # (n_steps,)
    interpolados = []
    for img in imgs_tensor:
        interp = baseline + alphas[:, None, None, None] * (img.unsqueeze(0) - baseline)
        interpolados.append(interp)

    # Calcular gradientes para cada imagen y cada paso de interpolación
    atribuciones_batch = []
    for img_orig, interp in zip(imgs_tensor, interpolados):
        interp = interp.requires_grad_(True)
        preds  = modelo(interp)        # (n_steps,)
        grad   = torch.autograd.grad(
            outputs=preds.sum(),
            inputs=interp,
            create_graph=False,
        )[0]                           # (n_steps, 3, 224, 224)

        # Promedio de gradientes × diferencia imagen-baseline
        grad_medio   = grad.mean(dim=0)                         # (3, 224, 224)
        delta        = img_orig - baseline.squeeze(0)           # (3, 224, 224)
        atrib        = (grad_medio * delta).abs().mean(dim=0)   # (224, 224)
        atribuciones_batch.append(atrib.detach().cpu().numpy())

    return np.stack(atribuciones_batch)   # (N, 224, 224)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def analizar(n_gradcam: int = 8, n_shap: int = 4) -> None:
    """
    Ejecuta el análisis completo de interpretabilidad.

    Parámetros
    ----------
    n_gradcam : número de imágenes para el grid de Grad-CAM
    n_shap    : número de imágenes para el análisis SHAP
    """
    # -------------------------------------------------------------------------
    # Cargar modelo
    # -------------------------------------------------------------------------
    ruta_ckpt = MODELOS_DIR / "mejor_modelo.pt"
    if not ruta_ckpt.exists():
        print(f"[ERROR] No se encontró el modelo en {ruta_ckpt}")
        print("  → Corre 05_entrenamiento.py primero.")
        sys.exit(1)

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    modelo = cargar_modelo(ruta_ckpt, device, dropout_p=0.5)
    print(f"  Modelo cargado desde: {ruta_ckpt.name}")
    print(f"  Dispositivo: {device}")

    # -------------------------------------------------------------------------
    # Cargar imágenes de test
    # -------------------------------------------------------------------------
    test_csv = SPLITS_DIR / "test.csv"
    if not test_csv.exists():
        print(f"[ERROR] No se encontró {test_csv}")
        print("  → Corre 03_split.py primero.")
        sys.exit(1)

    df_test = pd.read_csv(test_csv)
    transform_eval = get_transforms("test")

    # Seleccionar muestra estratificada: mitad con IPM alto, mitad con IPM bajo
    mitad    = max(n_gradcam // 2, 1)
    df_alto  = df_test.nlargest(mitad,  "inc_ipm_pct")
    df_bajo  = df_test.nsmallest(mitad, "inc_ipm_pct")
    df_muestra = pd.concat([df_alto, df_bajo]).sample(
        min(n_gradcam, len(df_test)), random_state=42
    ).reset_index(drop=True)

    print(f"\n  Imágenes para Grad-CAM: {len(df_muestra)}")
    print(f"  Rango de IPM: [{df_muestra['inc_ipm_pct'].min():.1f}, {df_muestra['inc_ipm_pct'].max():.1f}]")

    # -------------------------------------------------------------------------
    # GRAD-CAM
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  GRAD-CAM")
    print("=" * 60)

    grad_cam = GradCAM(modelo)

    ncols   = 4
    nrows   = len(df_muestra) // ncols + (1 if len(df_muestra) % ncols else 0)
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * 4, nrows * 2 * 3.5))

    # Aplanar axes para indexación simple
    if nrows * 2 == 2 and ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    axes = axes.reshape(nrows * 2, ncols)

    for i, (_, fila) in enumerate(df_muestra.iterrows()):
        row_img  = (i // ncols) * 2
        col      = i % ncols

        # Cargar imagen y transformar
        img_pil    = Image.open(fila["ruta_imagen"]).convert("RGB")
        img_tensor = transform_eval(img_pil)

        # Calcular Grad-CAM y predicción
        heatmap, ipm_pred = grad_cam(img_tensor, device)

        # Imagen original (desnormalizada)
        img_np    = desnormalizar(img_tensor)
        overlay   = superponer_heatmap(img_np, heatmap)

        # Fila superior: imagen original
        ax_orig = axes[row_img, col]
        ax_orig.imshow(img_np)
        ax_orig.set_title(
            f"IPM real={fila['inc_ipm_pct']:.1f}  pred={ipm_pred:.1f}\n"
            f"Error={abs(ipm_pred - fila['inc_ipm_pct']):.1f}",
            fontsize=8,
        )
        ax_orig.axis("off")

        # Fila inferior: overlay Grad-CAM
        ax_cam = axes[row_img + 1, col]
        ax_cam.imshow(overlay)
        ax_cam.set_title("Grad-CAM", fontsize=8)
        ax_cam.axis("off")

        print(f"  [{i+1}/{len(df_muestra)}] IPM real={fila['inc_ipm_pct']:.1f}  pred={ipm_pred:.1f}")

    grad_cam.limpiar()

    # Apagar ejes sobrantes
    for i in range(len(df_muestra), nrows * ncols):
        axes[(i // ncols) * 2, i % ncols].axis("off")
        axes[(i // ncols) * 2 + 1, i % ncols].axis("off")

    fig.suptitle(
        "Grad-CAM: zonas de la imagen usadas para predecir IPM\n"
        "Rojo = alta importancia  |  Azul = baja importancia",
        fontsize=13,
    )
    plt.tight_layout()
    ruta_gradcam = FIGURAS_DIR / "gradcam_muestra.png"
    plt.savefig(ruta_gradcam, dpi=150)
    plt.close()
    print(f"\n  Figura Grad-CAM guardada: {ruta_gradcam}")

    # -------------------------------------------------------------------------
    # SHAP (Integrated Gradients)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SHAP — Integrated Gradients")
    print("=" * 60)

    df_shap   = df_muestra.head(n_shap)
    imgs_list = []
    for _, fila in df_shap.iterrows():
        img_pil    = Image.open(fila["ruta_imagen"]).convert("RGB")
        img_tensor = transform_eval(img_pil)
        imgs_list.append(img_tensor)

    imgs_tensor = torch.stack(imgs_list)

    # Baseline: imagen negra normalizada
    # La imagen negra (todos los píxeles = 0 antes de la normalización) equivale
    # a un fondo neutro sin información visual. Es la opción estándar para
    # Integrated Gradients en imágenes.
    # Después de la normalización ImageNet, el "negro" es el vector -µ/σ.
    black_pixel = torch.zeros(1, 3, 224, 224)
    mean_t = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std_t  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    baseline = (black_pixel - mean_t) / std_t

    print(f"  Calculando IG para {len(df_shap)} imágenes (n_steps=50)...")
    atribuciones = calcular_shap(modelo, imgs_tensor, baseline, device, n_steps=50)

    # Figura SHAP
    fig, axes = plt.subplots(n_shap, 3, figsize=(12, n_shap * 3.5))
    if n_shap == 1:
        axes = axes.reshape(1, 3)

    for i in range(n_shap):
        fila       = df_shap.iloc[i]
        img_np     = desnormalizar(imgs_list[i])
        atrib      = atribuciones[i]          # (224, 224) valores absolutos

        # Normalizar al rango [0, 1]
        atrib_norm = (atrib - atrib.min()) / (atrib.max() - atrib.min() + 1e-8)

        ipm_pred_t = modelo(imgs_list[i].unsqueeze(0).to(device))
        ipm_pred   = float(ipm_pred_t.item())

        # Columna 1: imagen original
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(
            f"Original\nIPM real={fila['inc_ipm_pct']:.1f}  pred={ipm_pred:.1f}",
            fontsize=9,
        )
        axes[i, 0].axis("off")

        # Columna 2: mapa de atribuciones
        im = axes[i, 1].imshow(atrib_norm, cmap="hot")
        axes[i, 1].set_title("Atribuciones SHAP\n(más brillante = más importante)", fontsize=9)
        axes[i, 1].axis("off")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Columna 3: overlay
        overlay_shap = superponer_heatmap(img_np, atrib_norm, alpha=0.5)
        axes[i, 2].imshow(overlay_shap)
        axes[i, 2].set_title("Imagen + SHAP overlay", fontsize=9)
        axes[i, 2].axis("off")

        print(f"  [{i+1}/{n_shap}] IPM real={fila['inc_ipm_pct']:.1f}  pred={ipm_pred:.1f}  "
              f"atrib max={atrib.max():.4f}")

    fig.suptitle(
        "SHAP (Integrated Gradients): contribución de cada zona a la predicción de IPM\n"
        "Zonas brillantes = mayor contribución a la predicción",
        fontsize=12,
    )
    plt.tight_layout()
    ruta_shap = FIGURAS_DIR / "shap_muestra.png"
    plt.savefig(ruta_shap, dpi=150)
    plt.close()
    print(f"\n  Figura SHAP guardada: {ruta_shap}")

    print()
    print("=" * 60)
    print("  INTERPRETACIÓN ESPERADA")
    print("=" * 60)
    print("""
  Si el modelo aprendió características económicamente sensatas:

  Grad-CAM:
    ✓ Las zonas calientes deberían coincidir con fachadas, pavimento,
      infraestructura visible y materiales de construcción.
    ✗ No deberían activarse sobre el cielo, las marcas de agua de Google
      o zonas fuera de la calle.

  SHAP:
    ✓ Píxeles asociados a deterioro urbano (grietas, pintura descascarada,
      ausencia de zonas verdes) deberían tener atribuciones altas en
      imágenes con IPM elevado.
    ✓ Píxeles asociados a infraestructura formal (andenes limpios, locales
      comerciales, señalización) deberían tener atribuciones altas en
      imágenes con IPM bajo.

  Un resultado negativo (atribuciones sobre zonas irrelevantes) también
  es informativo: indicaría que las fotos de vías principales de USME
  no contienen suficiente señal sobre el nivel socioeconómico de los
  barrios adyacentes — lo que sería en sí mismo un hallazgo válido.
""")
    print("  Script completado.")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpretabilidad PovertyNet")
    parser.add_argument(
        "--n_gradcam", type=int, default=8,
        help="Número de imágenes para el grid Grad-CAM (default: 8)"
    )
    parser.add_argument(
        "--n_shap", type=int, default=4,
        help="Número de imágenes para el análisis SHAP (default: 4)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PASO 6 — Interpretabilidad: Grad-CAM + SHAP")
    print("=" * 60)

    analizar(n_gradcam=args.n_gradcam, n_shap=args.n_shap)

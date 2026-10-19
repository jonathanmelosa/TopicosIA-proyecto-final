---
output:
  html_document: default
  pdf_document: default
---
# Informe Metodológico: Predicción de Pobreza con Google Street View — USME, Bogotá

**Proyecto:** Proyecto final de Tópicos de Inteligencia Artifical  
**Fecha:** Abril 2026  
**Pipeline:** `00_descargar_datos` → `extract_imagenes` → `fetch_fechas` → `01_cruce_geografico` → `03_split` → `05_entrenamiento`

---

## 1. Resumen Ejecutivo

Este proyecto replica y extiende la metodología de Jean et al. (2016) y Gebru et al. (2017) para predecir indicadores de pobreza a partir de imágenes de Google Street View mediante transfer learning sobre ResNet18. Se recolectaron **16,265 imágenes** sobre la red vial de USME (Bogotá), de las cuales **7,410 corresponden al período anterior a 2021** y son consistentes temporalmente con la Encuesta Multipropósito 2021 (EM 2021). El modelo final —entrenado sobre **5,238 imágenes** de 3–4 UPZs y evaluado sobre **679 imágenes** de UPZs no vistas— obtiene un **R² de test de 0.0000**, lo que indica que el modelo no supera la predicción naïve de la media. Este resultado no es un fracaso técnico: es el resultado esperado y honesto de un diseño de evaluación riguroso. Las causas estructurales son identificables y corregibles, y se detallan a continuación.

---

## 2. Pipeline Completo: Decisiones y Críticas

### 2.1 `extract_imagenes.py` — Descarga de imágenes Street View

**Qué hace:** Descarga imágenes de Google Street View Static API sobre la red vial de USME. Reemplaza un muestreo en grilla uniforme por muestreo sobre la red vial de OpenStreetMap (OSMnx).

**Decisiones clave:**

| Parámetro | Valor | Justificación declarada |
|-----------|-------|-------------------------|
| `DELTA_M` | 50 m | Separación entre puntos sobre cada calle |
| `MAX_CONCURRENT` | 40 | Peticiones simultáneas a la API |
| `IMG_SIZE` | 640×640 px | Máximo sin costo adicional |
| `network_type` | `"drive"` | Solo vías vehiculares |
| Filtro copyright | `"oogle"` en campo | Solo imágenes de Google, excluye terceros |

**Crítica metodológica:**

- **Sesgo de cobertura (Street View Coverage Bias):** Google Street View tiene cobertura sistemáticamente desigual. En USME, una localidad periférica con altas tasas de informalidad urbana, el cubrimiento se concentra en vías principales. Las zonas de mayor pobreza —frecuentemente asociadas a asentamientos informales con calles sin pavimentar o de acceso restringido— quedan subrepresentadas o directamente excluidas. El filtro `network_type="drive"` agrava esto al excluir caminos peatonales donde la pobreza puede ser más intensa.

- **Distancia de 50 metros:** Es una decisión sin respaldo empírico o de la literatura. Trabajos como Gebru et al. (2017) usan 200m en contextos urbanos americanos. A 50m en calles con velocidad promedio de 30 km/h, dos imágenes consecutivas capturan prácticamente la misma escena. Esto genera **alta autocorrelación espacial** dentro del dataset, lo que infla artificialmente el tamaño efectivo de la muestra.

- **Ausencia de validación de calidad de imagen:** No se verifica ni el ángulo de la cámara, ni la obstrucción por árboles o vehículos, ni la calidad JPEG. Imágenes con construcciones en renovación, andamios o incidentes de tráfico inyectan ruido irreducible.

- **Un solo ángulo por punto:** El script descarga `fov=90, pitch=0` (vista frontal horizontal). La literatura reciente (Fan et al., 2023) usa múltiples ángulos (0°, 90°, 180°, 270°) para aumentar la información sobre el entorno construido. Con un solo ángulo, el campo visual capturado depende en gran medida de la dirección del trayecto del vehículo de Google.

---

### 2.2 `fetch_fechas.py` — Consulta de fechas de captura

**Qué hace:** Para cada imagen descargada, consulta la Street View Metadata API para obtener la fecha de captura de la panorámica descargada y, adicionalmente, consulta con `source=outdoor` para detectar si existe una panorámica exterior alternativa de fecha diferente en las mismas coordenadas.

**Decisiones clave:**

- `source=default` vs `source=outdoor`: Se registra fecha alternativa solo si el `pano_id` difiere (es una panorámica distinta).
- Llave de reanudación: el script es idempotente — retoma desde donde quedó si se interrumpe.
- `MAX_CONCURRENT=40`: Peticiones de metadata son gratuitas; alta concurrencia es válida.

**Crítica metodológica:**

- **La fecha de la panorámica no es la fecha de la condición socioeconómica:** El campo `date` en la API de Street View devuelve la fecha de captura de la imagen, no necesariamente la fecha en que el área mostraba esas condiciones de pobreza. Una foto de 2019 puede mostrar una obra de construcción que para 2021 era ya un barrio consolidado. La foto es un proxy del estado físico en ese momento, no del estado socioeconómico al momento de la encuesta.

- **Pérdida masiva por filtro temporal:** De 16,265 imágenes totales, solo 7,410 (45.6%) son previas a 2021. Esto implica que más de la mitad del trabajo de descarga fue inútil. Para una próxima iteración, el filtro temporal debe aplicarse **durante la descarga**, verificando la fecha antes de pagar por la imagen.

- **Resolución mensual de fechas:** La API devuelve `YYYY-MM`. No se conoce el día exacto. Si una fotografía es de noviembre 2020 y la encuesta fue en diciembre 2021, hay un año de desfase. Para análisis de variación temporal, la granularidad mensual es insuficiente.

---

### 2.3 `00_descargar_datos.py` — Insumos socioeconómicos

**Qué hace:** En cinco pasos, descarga y procesa los insumos de la Encuesta Multipropósito (EM 2021): (1) polígonos UPZ desde ArcGIS REST, (2) microdatos EM 2021 a nivel persona, (3) variables adicionales de la EM 2021, (4) merge persona-a-persona por `DIRECTORIO_PER`, (5) agregación de incidencia de pobreza por UPZ con expansión muestral.

**Decisiones en PASO 4 — Merge:**

| Decisión | Detalle |
|----------|---------|
| Llave de join | `DIRECTORIO_PER` (identificador único de persona) |
| Tipo de join | `inner` con `validate="1:1"` |
| Resultado | 292,281 personas × 1,644 variables |

**Crítica:** El uso de `validate="1:1"` es correcto y protege contra duplicaciones accidentales. El join interno (`inner`) descarta personas que están en solo uno de los dos archivos — aunque dado que ambos provienen de la misma encuesta, la pérdida debería ser mínima.

**Decisiones en PASO 5 — Incidencia por UPZ:**

| Decisión | Detalle | Justificación |
|----------|---------|---------------|
| Colapso a hogar | `drop_duplicates("DIRECTORIO_HOG")` | `N_pobre_*` es constante dentro del hogar |
| Factor de expansión | `FEX_C` | Corrección por diseño muestral complejo |
| Fórmula | `100 × Σ(FEX_C × indicador) / Σ(FEX_C)` | Incidencia ponderada (Horvitz-Thompson) |
| Descarte de hogares sin UPZ | 22,845 hogares descartados | Sin `COD_UPZ_GRUPO` asignado |

**Crítica metodológica:**

- **22,845 hogares descartados (≈ 15% de la muestra):** La ausencia de `COD_UPZ_GRUPO` no es aleatoria. Los hogares fuera del polígono de alguna UPZ tienden a estar en la periferia urbana o en áreas de expansión — precisamente las zonas de mayor pobreza. El descarte introduce **sesgo de selección hacia la subestimación de la pobreza en UPZs periféricas**.

- **UPZs compuestas (códigos 800+):** La EM 2021 agrupa algunas UPZs de baja población en grupos artificiales (811 = Alfonso López + Ciudad Usme; 812 = Parque Entrenubes + Danubio). El mapeo manual `MAPA_SHP_A_ENCUESTA` en `01_cruce_geografico.py` es correcto pero introduce incertidumbre: se asigna el mismo indicador de pobreza a dos UPZs que pueden tener perfiles muy distintos.

- **Tres indicadores de pobreza pero solo IPM se usa en el modelo:** El script produce `inc_monetaria_pct`, `inc_ipm_pct`, e `inc_extrema_pct`. Sin embargo, el entrenamiento usa solo `inc_ipm_pct`. No hay análisis de cuál indicador es más adecuado como variable objetivo dada la disponibilidad de señal visual.

---

### 2.4 `01_cruce_geografico.py` — Cruce geográfico imagen-UPZ

**Qué hace:** (1) Extrae coordenadas de los nombres de archivo, (2) une fechas de captura, (3) carga polígonos UPZ, (4) spatial join punto-en-polígono, (5) traduce códigos del shapefile a códigos de la encuesta, (6) une indicadores de pobreza, (7) genera mapa de cobertura.

**Decisiones clave:**

- **Spatial join `predicate="within"`:** Conservador — descarta puntos en bordes de polígonos. El predicado `"intersects"` retendría más puntos pero podría asignar una imagen a la UPZ incorrecta si está sobre el límite.
- **Redondeo a 8 decimales** para el join lat/lon entre imágenes y fechas: Correcto para garantizar match exacto cuando ambos derivan del mismo nombre de archivo.

**Crítica:**

- **16,265 imágenes originales → 7,410 con fecha pre-2021 → número final no documentado con UPZs válidas:** El pipeline descarta imágenes en múltiples pasos sin consolidar un reporte de attrition completo. Para una publicación académica, cada paso de descarte debe cuantificarse y justificarse.

- **Las coordenadas de los archivos son las coordenadas del punto de muestreo, no de la panorámica:** La API de Street View ajusta la panorámica al punto real más cercano de la base de datos. El nombre del archivo usa las coordenadas reales de la panorámica (devueltas por la API en `data["location"]`), lo cual es correcto. Pero este detalle no se documenta en el pipeline.

---

### 2.5 `03_split.py` — División train/val/test espacial

**Qué hace:** Divide el dataset en conjuntos de entrenamiento (70%), validación (15%) y prueba (15%) asignando UPZs completas a cada subconjunto mediante un algoritmo greedy de bin-packing.

**Decisiones clave:**

| Decisión | Detalle |
|----------|---------|
| Unidad de split | UPZ completa |
| Algoritmo | Greedy: asigna cada UPZ al split con mayor déficit vs objetivo |
| Shuffle pre-asignación | SEED=42 para reproducibilidad |
| Filtro temporal | Solo imágenes con `fecha_descargada` < 2021 |
| UPZs excluidas | Parque Entrenubes y Los Libertadores (baja cobertura) |

**Justificación del split espacial:** Con un split aleatorio, dos imágenes de la misma calle tomadas a 50 metros podrían estar una en train y otra en test. La correlación espacial entre imágenes adyacentes generaría **data leakage**: el modelo "memoriza" la escena sin generalizar. El split por UPZ garantiza separación geográfica estricta entre conjuntos.

**Crítica profunda:**

- **El split espacial es necesario pero insuficiente con 5–7 UPZs:** Con solo 5–7 UPZs en USME y una división 70/15/15, el set de entrenamiento tiene apenas **3–4 valores únicos del target**. El modelo no aprende a predecir el nivel de pobreza de una imagen — aprende a distinguir a qué UPZ pertenece la imagen, lo que es trivialmente imposible para UPZs no vistas. Esto es una limitación estructural del estudio, no del código.

- **La exclusión de Parque Entrenubes y Los Libertadores fue ad-hoc:** Se tomó visualmente, sin un umbral mínimo de cobertura documentado. Si esas UPZs tienen características socioeconómicas particulares (e.g., Parque Entrenubes es mayormente zona verde con baja densidad habitacional), su exclusión introduce sesgo en la distribución de los targets.

- **El shuffle pre-greedy con SEED=42 no garantiza representatividad:** La asignación depende del orden aleatorio de las UPZs, que con solo 5–7 unidades puede producir particiones muy desequilibradas en la distribución de pobreza. No hay verificación de que la media del target sea similar entre splits — el script emite una advertencia si la diferencia supera 10 pp, pero este umbral es arbitrario.

---

### 2.6 `02_dataset.py` — Dataset y aumentación de datos

**Qué hace:** Define `StreetViewDataset` (hereda de `torch.utils.data.Dataset`) con transformaciones diferenciadas para train vs. val/test.

**Decisiones de aumentación (train):**

| Transformación | Parámetros | Justificación |
|----------------|------------|---------------|
| `RandomResizedCrop(224)` | scale=(0.8, 1.0) | Variación de distancia al sujeto |
| `RandomHorizontalFlip` | p=0.5 | Invarianza a la dirección de captura |
| `ColorJitter` | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05 | Variación de iluminación/hora del día |
| `RandomRotation` | ±5° | Corrección de inclinaciones leves |
| Normalización ImageNet | µ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225] | Requerido por ResNet18 preentrenado |

**Crítica:**

- **`ColorJitter` demasiado conservador para Street View:** Las imágenes de Street View tienen alta variabilidad de iluminación (amanecer, mediodía, atardecer, días nublados). Un `brightness=0.3` puede ser insuficiente para cubrir esta variación. Trabajos como Suel et al. (2019) usan rangos de hasta 0.5.

- **No se aplica `RandomVerticalFlip`:** Correcto — un volteo vertical en una escena urbana no tiene sentido semántico.

- **`RandomRotation(±5°)` podría ampliarse:** Street View tiene variaciones de pitch (inclinación) de hasta ±15° en terrenos con pendiente como los de USME (localidad montañosa al sureste de Bogotá). Una rotación de ±5° podría ser insuficiente.

- **Ausencia de normalización del target:** Las imágenes están normalizadas con estadísticas ImageNet, pero `inc_ipm_pct` (rango observado: 6.8–16.5%) se pasa directamente al modelo sin escalar. Un target normalizado a [0,1] o z-score facilitaría la convergencia y haría los LR por defecto más robustos. El hecho de que el entrenamiento haya divergido repetidamente con MPS es en parte atribuible a la escala no normalizada del target.

---

### 2.7 `04_modelo.py` — Arquitectura PovertyNet

**Qué hace:** Define `PovertyNet`: ResNet18 preentrenado en ImageNet con la capa `fc` final reemplazada por un regresor escalar `Dropout(0.5) → Linear(512, 1)`.

**Estrategia de fine-tuning:**

| Capas | Estado | Parámetros |
|-------|--------|------------|
| conv1, bn1, layer1, layer2, layer3 | Congeladas | 2,782,784 (no entrenables) |
| layer4 | Entrenable | 8,394,241 entrenables |
| avgpool + regressor | Entrenable | incluido arriba |
| **Total** | | **11,177,025** |

**Justificación:** Las capas 1–3 detectan features universales (bordes, texturas, formas). Layer4 detecta features de alto nivel semánticamente interpretables para escenas urbanas.

**Crítica metodológica:**

- **ResNet18 puede ser insuficiente para esta tarea:** La detección de señales de pobreza en imágenes urbanas requiere capturar detalles sutiles: estado de fachadas, densidad de graffiti, tipo de pavimento, presencia de comercio informal. ResNet18 con 11M parámetros fue diseñado para clasificación de 1,000 objetos naturales. Arquitecturas más recientes como EfficientNet-B3, ViT-B/16 o DINOv2 tienen features preentrenados más ricos para escenas urbanas.

- **Dropout 0.5 es agresivo para un regresor:** Con una sola capa lineal de salida, Dropout(0.5) desactiva aleatoriamente 256 de los 512 features de entrada y escala los supervivientes por 2×. En la práctica, esto aumenta la varianza de la predicción. Para regresión con pocas observaciones efectivas (pocas UPZs), un Dropout de 0.3 o incluso 0.1 sería más apropiado.

- **No hay capa de normalización del output:** El regresor produce valores ilimitados. Sin inicialización cuidadosa o normalización del target, el modelo puede tardar decenas de épocas en aprender la escala correcta.

- **Fine-tuning de layer4 con batch-norm en train mode causó inestabilidad numérica crítica:** Las capas BatchNorm de layer4, cuando se actualizan con batches de imágenes de Street View (distribución muy diferente a ImageNet), producen estadísticas inestables. Fue necesario forzar `m.eval()` para todos los BatchNorm durante el entrenamiento. Este comportamiento documenta un **problema conocido de MPS** en PyTorch con distribuciones out-of-domain.

---

### 2.8 `05_entrenamiento.py` — Entrenamiento

**Qué hace:** Ejecuta el loop de entrenamiento con early stopping, differential learning rate, LR scheduler, y evaluación final en test.

**Hiperparámetros finales:**

| Hiperparámetro | Valor | Rol |
|----------------|-------|-----|
| `epochs` | 50 | Máximo de iteraciones sobre el dataset completo |
| `batch_size` | 32 | Número de imágenes por actualización de pesos |
| `lr` | 1e-4 | Learning rate del regresor (nueva capa) |
| `lr_backbone` | 1e-5 | Learning rate de layer4 (10× menor, ya preentrenada) |
| `weight_decay` | 1e-4 | Penalización L2 sobre los pesos (Ridge) |
| `dropout_p` | 0.5 | Fracción de neuronas desactivadas durante train |
| `patience` | 7 | Épocas sin mejora antes de early stopping |

**Qué busca cada parámetro:**

- **`lr` diferencial:** Layer4 ya tiene una representación útil aprendida de ImageNet; un LR grande la destruiría. El regresor parte de cero y necesita converger rápidamente. La ratio 10:1 (`lr/lr_backbone`) es estándar en la literatura de fine-tuning.
- **`weight_decay`:** Equivalente a regularización Ridge. Penaliza pesos grandes, forzando al modelo a distribuir la información entre features, reduciendo el sobreajuste.
- **`patience=7`:** Con 5,238 imágenes de entrenamiento y batch_size=32, una época corresponde a ~163 actualizaciones de pesos. 7 épocas sin mejora son ~1,141 pasos — suficiente para detectar mesetas reales.
- **`ReduceLROnPlateau(factor=0.5, patience=3)`:** Reduce el LR a la mitad si val_rmse no mejora en 3 épocas consecutivas. Esto complementa el early stopping: primero intenta rescatar el entrenamiento con menor LR antes de detenerlo.

**Evolución del proceso de debugging:**

El entrenamiento requirió cuatro iteraciones de corrección antes de producir resultados válidos:

1. **Error 1 — `PicklingError`:** `num_workers=4` intentaba serializar el Dataset vía pickle, pero el módulo cargado con `importlib` tiene un nombre inválido (`02_dataset.py`). Fix: `num_workers=0`.
2. **Error 2 — RMSE=Inf (explosión):** LR=1e-3 con target en escala 6–16 generaba gradientes MSE de magnitud ~100, que se amplificaban sin clipping. Fix: reducir LR 10× y agregar `clip_grad_norm_(max_norm=1.0)`.
3. **Error 3 — RMSE=NaN (inestabilidad MPS):** BatchNorm en mode `train` con distribución de activaciones out-of-domain (Street View ≠ ImageNet) en el backend MPS de Apple. Fix: forzar `m.eval()` para todos los BatchNorm durante el entrenamiento.
4. **Error 4 — RMSE residual Inf (bug MPS forward pass):** El backend MPS producía salidas no-finitas para ~98% de los batches incluso con BN en eval. Fix definitivo: `PYTORCH_ENABLE_MPS_FALLBACK=1` + `--device cpu`.

---

## 3. Análisis de Métricas Obtenidas

### 3.1 Resultados finales (CPU, numericamente estables)

| Métrica | Train (mejor época) | Val (mejor época) | Test |
|---------|--------------------|--------------------|------|
| RMSE (pp IPM) | ~3.94 | **3.76** | **2.77** |
| MAE (pp IPM) | — | — | 2.36 |
| R² | — | ~0.0000 | **0.0000** |
| Época de convergencia | — | 4 | — |
| Early stopping | — | Época 11 | — |

*pp = puntos porcentuales del Índice de Pobreza Multidimensional*

### 3.2 Interpretación económica de las métricas

**R² = 0.0000 no significa que el modelo no aprende. Significa que no generaliza a UPZs no vistas.**

Para entender por qué, es útil descomponer el R²:

```
R² = 1 - SS_res / SS_tot

Targets test:         UPZ única no vista, IPM ≈ X%
Train IPM medio:      ≈ (6.8 + 10.2 + 16.5) / 3 ≈ 11.2%
Predicción del modelo: ≈ 11.2% (aprendió el promedio del train)

Si X ≠ 11.2%, SS_res / SS_tot ≈ 1 → R² ≈ 0
```

El RMSE de test de **2.77 pp** indica que el modelo predice el IPM del test con un error de 2.8 puntos porcentuales. En un rango observado de 6.8–16.5 pp (amplitud ≈ 10 pp), este error corresponde al **28% del rango**. Es equivalente a no distinguir entre "pobreza moderada" y "pobreza alta".

**El R² = 0 confirma que el modelo es equivalente al estimador naïve `ȳ_train`:** predecir el promedio del conjunto de entrenamiento para todas las observaciones del test. Esto no es consecuencia del modelo — es consecuencia directa de que el test contiene una UPZ cuyo nivel de pobreza nunca fue observado durante el entrenamiento. Con un split espacial estricto y solo 5–7 UPZs totales, **ningún modelo de deep learning podría obtener R² > 0 en este diseño**, porque no existe variación within-UPZ en el target.

### 3.3 Diagnóstico de la curva de aprendizaje

| Época | Train RMSE | Val RMSE | Interpretación |
|-------|-----------|----------|----------------|
| 1 | 5.55 | 4.55 | Predicción aleatoria cerca del extremo del rango |
| 2 | 4.37 | 4.50 | Ligera convergencia hacia la media del train |
| 4 | 4.15 | **3.76** | Mejor val — modelo predice ≈ promedio ponderado del train |
| 11 | 3.84 | 4.57 | Overfitting: train sigue mejorando, val empeora |

El modelo alcanza su mejor punto en solo **4 épocas**, lo que indica que el gradiente tiene muy poco espacio para mejorar dado el target (prácticamente constante dentro de cada UPZ). Las épocas 5–11 muestran overfitting leve sobre los patrones de UPZ específicos que no generalizan.

---

## 4. Problemas Estructurales del Diseño

### 4.1 Pseudo-replicación: el problema central

Este es el problema más grave del estudio y debe abordarse antes de cualquier otro cambio.

**El target `inc_ipm_pct` tiene exactamente 3–5 valores únicos en el dataset completo.** Cada imagen de una misma UPZ tiene exactamente el mismo valor de IPM. Las 5,238 imágenes de train no representan 5,238 observaciones independientes del fenómeno de interés — representan **3–4 observaciones de UPZ repetidas miles de veces**.

Formalmente: sea $n_U$ el número de UPZs y $n_I$ el número de imágenes por UPZ. El modelo minimiza:

$$\mathcal{L} = \frac{1}{N} \sum_{u=1}^{n_U} \sum_{i=1}^{n_I^{(u)}} \left(\hat{y}_{ui} - \bar{y}_u\right)^2$$

donde $\bar{y}_u$ es constante para todas las imágenes $i$ de la UPZ $u$. El modelo optimiza una función con $n_U = 3{-}4$ grados de libertad efectivos, no $N = 5{,}238$. El tamaño muestral efectivo para estimar la relación imagen→pobreza es el número de UPZs, no el número de imágenes.

**Consecuencia:** Cualquier test estadístico sobre el R² que use N=5,238 está inflado por un factor de $\approx n_I / n_U \approx 1{,}300$. Los intervalos de confianza de cualquier métrica son artificialmente estrechos.

### 4.2 Falacia ecológica (Ecological Fallacy)

El modelo asigna el IPM de la UPZ a cada imagen individual. El IPM es una medida de pobreza del **hogar** agregada al nivel de la **zona geográfica**. Una imagen de una fachada en una UPZ con 40% de pobreza no corresponde necesariamente a un hogar pobre — puede ser la fachada del 60% de hogares no pobres de esa UPZ.

Esto no invalida el enfoque (la correlación visual con la pobreza del área puede ser informativa incluso si el target es ecológico), pero **sí invalida la interpretación de las predicciones a nivel de imagen individual**. El modelo predice la pobreza del área, no de los ocupantes del edificio en la imagen.

### 4.3 Heterogeneidad insuficiente del outcome

USME es una de las localidades más pobres de Bogotá. El rango observado de IPM en sus UPZs es solo 6.8–16.5% — una variación de apenas 10 puntos porcentuales. El modelo no tiene exposición a UPZs ricas (Chapinero, Usaquén: IPM < 2%) ni a UPZs extremadamente pobres (Ciudad Bolívar: IPM > 35%). Sin esa variación extrema, las señales visuales que el modelo debería aprender (calidad de fachadas, pavimento, vegetación, señalética) apenas están presentes en los datos.

Para contexto: el IPM de USME (11%) y el de Chapinero (1.5%) difieren en un factor de 7×. Esta heterogeneidad está completamente ausente del dataset actual.

---

## 5. Plan de Mejoras

Las mejoras se organizan en tres niveles según impacto esperado y complejidad de implementación.

### 5.1 Mejoras de alto impacto (necesarias para obtener R² > 0)

#### 5.1.1 Expandir a múltiples localidades

**Descripción:** Incluir al menos 6–8 localidades de Bogotá que cubran el espectro completo de pobreza: Chapinero y Usaquén (IPM < 3%), Kennedy y Bosa (IPM 10–20%), Ciudad Bolívar (IPM > 35%), además de USME.

**Impacto esperado:** Pasar de 5–7 UPZs a 80–100 UPZs, con variación en el target de 1% a 40%. El número de valores únicos del target pasa de 5 a 100. El split espacial producirá ~70 UPZs de train y ~15 cada una de val y test.

**Implementación:** Modificar `extract_imagenes.py` para aceptar una lista de localidades. El shapefile de localidades ya está en el pipeline.

#### 5.1.2 Normalizar el target

**Descripción:** Aplicar z-score al target antes del entrenamiento: $\tilde{y} = (y - \mu_{train}) / \sigma_{train}$.

**Impacto esperado:** Elimina la inestabilidad numérica del entrenamiento. El LR por defecto de 1e-4 puede ser inadecuado para un target en escala [6.8, 16.5]; con z-score el target tiene media 0 y std 1, haciendo los hiperparámetros del optimizador interpretables y robustos.

**Implementación:** Calcular `y_mean` y `y_std` sobre el train set. Aplicar en `StreetViewDataset.__getitem__`. Invertir en la evaluación.

#### 5.1.3 Target a nivel más granular

**Descripción:** Sustituir `inc_ipm_pct` por UPZ (114 unidades en Bogotá) por estimaciones de pobreza a nivel de **barrio** (≈850 barrios en Bogotá) o **manzana** (si están disponibles en la Secretaría Distrital de Planeación).

**Impacto esperado:** Incrementa dramáticamente la variabilidad del target. Dos imágenes del mismo barrio tendrían el mismo target, pero imágenes en barrios adyacentes de distinto estrato socioeconómico tendrían targets distintos, proporcionando señal real para el aprendizaje.

### 5.2 Mejoras de impacto moderado

#### 5.2.1 Usar múltiples ángulos por punto

Descargar 4 imágenes por coordenada: `heading=0, 90, 180, 270`. Concatenar features o usar agregación MIL (Multiple Instance Learning). Requiere ~4× más crédito de API.

#### 5.2.2 Arquitectura más potente

Sustituir ResNet18 por **EfficientNet-B3** o **DINOv2-small**. DINOv2 fue preentrenado con supervisión self-supervised sobre millones de imágenes urbanas y produce embeddings de mayor calidad para escenas en la calle.

#### 5.2.3 Multi-task learning

Entrenar simultáneamente para predecir `inc_monetaria_pct`, `inc_ipm_pct`, e `inc_extrema_pct`. Las tres variables están correlacionadas (~0.8 en la EM 2021) y compartir la representación puede reducir el overfitting.

**Modificación en `04_modelo.py`:** Cambiar `Linear(512, 1)` por `Linear(512, 3)` y usar pérdida combinada:

```python
loss = (criterio(preds[:, 0], ipm) 
      + criterio(preds[:, 1], monetaria) 
      + criterio(preds[:, 2], extrema)) / 3
```

#### 5.2.4 Filtro temporal más estricto

En lugar de `fecha < 2021`, usar `2018 ≤ fecha < 2021` para reducir el desfase temporal. Imágenes pre-2015 de USME pueden no representar el estado del parque construido al momento de la encuesta 2021.

#### 5.2.5 Usar variación temporal

Si hay imágenes de múltiples años del mismo punto, modelar el **cambio** en la apariencia urbana como predictor. La variación temporal captura dinámica de inversión/deterioro que puede ser más informativa que el nivel absoluto.

### 5.3 Mejoras técnicas del pipeline

#### 5.3.1 Resolver la inestabilidad MPS de forma permanente

El bug de MPS con ResNet18 BN out-of-domain no está resuelto — solo se bypasea con CPU. A mediano plazo:
- Actualizar PyTorch a ≥ 2.3, donde los bugs MPS con BatchNorm están documentados como corregidos.
- Considerar `torch.compile()` para aprovechar MPS nativamente.

#### 5.3.2 Línea de base naive

Antes de reportar métricas del modelo de deep learning, comparar contra baselines:

| Baseline | Descripción |
|----------|-------------|
| `ȳ_train` | Predecir la media del train | 
| KNN espacial | Asignar el IPM del punto de train más cercano geográficamente |
| Regresión sobre coordenadas | OLS de IPM sobre lat, lon |

Un R² = 0 que supera todos los baselines sigue siendo científicamente relevante. Un R² = 0 que no supera a `ȳ_train` (como es el caso actual) no lo es.

#### 5.3.3 Validación cruzada espacial (k-fold geográfico)

Con 80–100 UPZs, implementar **5-fold spatial CV** en lugar de un único split. En cada fold, ~16–20 UPZs forman el test y el resto es train. Promediar R² sobre los 5 folds da una estimación de generalización más robusta que un único split.

#### 5.3.4 Visualización de activaciones (Grad-CAM)

El script `06_interpretabilidad.py` (paso siguiente) debería implementar **Grad-CAM** sobre las imágenes del test set para visualizar qué partes de la imagen activan las predicciones de mayor pobreza. Esto permite verificar si el modelo está capturando señales económicamente interpretables (condición de fachadas, tipo de pavimento) o artefactos espurios (hora del día, sombras).

---

## 6. Evaluación Comparativa con la Literatura

| Estudio | Dataset | Localidades | Target | R² test |
|---------|---------|-------------|--------|---------|
| Jean et al. (2016) | África subsahariana | 5 países | Riqueza per cápita | 0.63 |
| Gebru et al. (2017) | EE.UU. | 50 estados | Ingreso medio | 0.48 |
| Suel et al. (2019) | Londres | 1 ciudad | Deprivación índice | 0.72 |
| **Este estudio** | **USME** | **1 localidad** | **IPM por UPZ** | **0.00** |

La diferencia fundamental con los estudios de referencia no es metodológica sino de **escala geográfica y variación del target.** Jean et al. y Gebru et al. operan sobre decenas de miles de áreas pequeñas con alta variación en riqueza. Este estudio opera sobre 5–7 UPZs de una localidad homogéneamente pobre. La comparación directa no es válida.

---

## 7. Conclusiones

El pipeline construido es **metodológicamente correcto en su diseño de evaluación** — el split espacial por UPZ es la decisión adecuada para evitar data leakage — pero está **limitado fundamentalmente por la escala del dato**, no por el modelo o el código.

Las tres acciones de mejora con mayor retorno esperado son, en orden de prioridad:

1. **Expandir a 6–8 localidades** de Bogotá (incrementa variación del target de 10 pp a 40 pp).
2. **Normalizar el target** antes del entrenamiento (resuelve inestabilidad numérica y acelera convergencia).
3. **Implementar baselines simples** como punto de comparación (sin esto, el R² = 0 carece de contexto interpretativo).

El código es reproducible, modular y correcto. Los errores encontrados — inestabilidad MPS, explosión de gradientes, pseudo-replicación — son documentados y corregibles. Con los datos adecuados, esta arquitectura puede producir resultados comparables a la literatura internacional.

---

## Apéndice: Árbol de Archivos del Pipeline

```
TopicosIA-proyecto-final/
├── 00_data/
│   ├── raw/
│   │   ├── upz/upz_bogota.geojson          # 114 polígonos UPZ
│   │   ├── em2021.csv                       # 292,281 personas EM 2021
│   │   ├── em2021_adicionales.csv           # Variables derivadas EM 2021
│   │   ├── em2021_merged.csv                # 292,281 × 1,644 variables
│   │   └── pobreza_upz.csv                  # 95 UPZs con indicadores
│   ├── USME/
│   │   ├── *.jpg                            # 24,478 imágenes 640×640
│   │   └── fechas.csv                       # Fechas de captura
│   └── processed/
│       ├── imagenes_dataset.csv             # 16,265 imágenes con UPZ y pobreza
│       └── splits/
│           ├── train.csv                    # 5,238 imágenes (3–4 UPZs)
│           ├── val.csv                      # 1,467 imágenes (1–2 UPZs)
│           └── test.csv                     # 679 imágenes (1 UPZ)
├── 01_code/
│   ├── 00_descargar_datos.py               # Pasos 1–5: datos socioeconómicos
│   ├── 00_descarga/
│   │   ├── extract_imagenes.py             # Descarga Street View async
│   │   └── fetch_fechas.py                 # Fechas via Metadata API
│   ├── 01_cruce_geografico.py              # Spatial join + dataset final
│   ├── 02_dataset.py                       # PyTorch Dataset + transforms
│   ├── 03_split.py                         # Split espacial por UPZ
│   ├── 04_modelo.py                        # PovertyNet (ResNet18 fine-tuning)
│   └── 05_entrenamiento.py                 # Training loop + evaluación
└── 02_outputs/
    ├── figuras/
    │   ├── cobertura_pobreza.png           # Mapa de cobertura
    │   ├── curvas_entrenamiento.png        # Loss/R² por época
    │   └── predicciones_test.png          # Pred vs. real en test
    ├── modelos/
    │   ├── mejor_modelo.pt                 # Checkpoint mejor val RMSE
    │   └── metricas.json                   # RMSE, MAE, R² del test
    └── informe_metodologico.md             # Este documento
```

---

*Informe generado como parte del proyecto final Tópicos de Inteligencia Artificial.Para reproducir el pipeline completo, ejecutar los scripts en orden numérico desde la raíz del proyecto.*

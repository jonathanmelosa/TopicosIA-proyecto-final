# Poverty prediction from Google Street View images (Usme, Bogotá)

Final course project, Master's in Applied Economics (MeCA), Universidad de los Andes (2026). Can street-level images predict how poor an area is? The project collects Google Street View images along the road network of Usme (Bogotá), links each image to the multidimensional poverty (IPM) incidence of its UPZ from the 2021 Bogotá Multipurpose Survey (EM 2021), and fine-tunes a ResNet18 to predict it.

Code, comments and the methodological report (`informe_metodologico_1.md`) are in Spanish. The original exercise design is in `diseño_detallado.pdf`.

## Result

The model does not beat a naive mean prediction on unseen UPZs: test R² = 0.00, RMSE = 2.77 pp, MAE = 2.36 pp (`02_outputs/modelos/metricas.json`). The report explains why. Usme has only 5–7 UPZs, so with a spatial split the training set contains 3–4 distinct target values; the target is an area-level average assigned to every image (ecological fallacy); and consecutive images are highly autocorrelated. It also lays out the changes needed to get a positive R² (more localities, a more granular target, target normalization, spatial k-fold validation).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0a | `01_code/00_descarga/extract_imagenes.py` | Samples points along the OpenStreetMap road network (OSMnx) and downloads Street View images asynchronously (aiohttp); resumable. |
| 0b | `01_code/00_descarga/verificar_cobertura.py` | Re-queries points without an image with a wider radius to check whether coverage is exhausted. |
| 0c | `01_code/00_descarga/refinar_muestreo.py` | Adds a finer sampling grid only in uncovered gaps, if the check recommends it. |
| 0d | `01_code/00_descarga/fetch_fechas.py` | Retrieves the capture date of each image (to keep images consistent in time with EM 2021). |
| 0e | `01_code/00_descargar_datos.py` | Downloads the UPZ boundaries and the EM 2021 microdata from Bogotá's open data portal and builds poverty indicators by UPZ. |
| 1 | `01_code/01_cruce_geografico.py` | Spatial join: assigns each image the IPM of its UPZ. |
| 2 | `01_code/02_dataset.py` | PyTorch `Dataset` with ImageNet normalization and data augmentation. |
| 3 | `01_code/03_split.py` | 70/15/15 split by whole UPZs to avoid spatial leakage. |
| 4 | `01_code/04_modelo.py` | ResNet18 (ImageNet weights) with a new regression head. |
| 5 | `01_code/05_entrenamiento.py` | MSE loss, Adam, weight decay, dropout, gradient clipping and early stopping. |
| 6 | `01_code/06_interpretabilidad.py` | Grad-CAM and Integrated Gradients attributions. |

## Setup

```bash
pip install -r requirements.txt
export GSV_API_KEY="your-key"   # Google Street View Static API, only for step 0
```

Paths are relative to the project root. The data (`00_data/`: about 24,500 Street View images and the EM 2021 microdata) and the trained model weights are not included: Street View terms do not allow redistributing images, and the survey can be downloaded with step 0e.

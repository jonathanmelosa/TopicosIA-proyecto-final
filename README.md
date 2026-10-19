# Street View image collection for poverty prediction (Usme, Bogotá)

Data-collection pipeline for a course project on predicting poverty from street-level imagery: Google Street View photos of the Usme locality (Bogotá) are later used to train a convolutional neural network against DANE poverty measures. Course: MECA 4107, *Big Data and Machine Learning for Applied Economics*, Universidad de los Andes (2026). The exercise design is described in `diseño_detallado.pdf` (Spanish).

This repository covers the image-collection stage. Code comments are in Spanish.

## Scripts (`01_code/`)

| Script | What it does |
|---|---|
| `extract_imagenes.py` | Samples points along the OpenStreetMap road network of the locality and downloads Street View images asynchronously. |
| `verificar_cobertura.py` | Checks whether coverage is exhausted: re-queries points without an image using a wider search radius and reports the share of "missed" images. |
| `refinar_muestreo.py` | Run only if the check finds missed images: adds a finer sampling grid in uncovered gaps and merges results into the same state file. |

## Design choices

- **Road network instead of a uniform grid.** Street View only exists on streets, so most points on a uniform grid return no image. Sampling along OSM roads (OSMnx) cuts the number of queries sharply and raises the hit rate.
- **Asynchronous requests (aiohttp).** A single thread handles 40+ concurrent requests.
- **Resumable.** Progress is saved periodically to a pickle; an interrupted run resumes with only the pending points.
- **Coverage diagnosis before refinement.** Points without an image are re-queried with a radius proportional to the sampling distance (3× spacing), so refinement is only run when it can add images.

## Setup

```bash
pip install geopandas osmnx aiohttp requests shapely scikit-learn scipy matplotlib pandas numpy
export GSV_API_KEY="your-key"          # Google Street View Static API
export RUTA_BASE="/path/to/data"       # optional; defaults to ./data next to 01_code/
```

`RUTA_BASE` must contain the Bogotá localities shapefile at `loca/Loca.shp` (column `LocNombre`). Downloaded images and the sampling state are written there. Images are not included in this repository because Street View terms do not allow redistribution.

## Run order

```bash
python 01_code/extract_imagenes.py
python 01_code/verificar_cobertura.py
python 01_code/refinar_muestreo.py     # only if the check recommends it
```

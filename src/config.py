"""
Configuracion central del proyecto.

Todo parametro que se barre en experimentos vive aqui, no disperso en los
notebooks. Cambiar un experimento = cambiar esto, no editar codigo.
"""

from pathlib import Path

# --- Reproducibilidad ---
RANDOM_STATE = 42
N_INIT = 10           # inicializaciones de K-Means; no bajar a 1 por "velocidad"

# --- Rutas ---
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SEGMENTED = RESULTS / "segmented_images"

# --- Preprocesamiento ---
# Se reduce por velocidad. La imagen va bilineal; la MASCARA va nearest-neighbor.
RESIZE_TO = (128, 128)   # (alto, ancho); None para no redimensionar

# --- Experimentos ---
K_PRIMARY = 2                        # pregunta A: foreground/background
K_SWEEP = [2, 3, 4, 5, 6, 8, 10]     # pregunta B: efecto de K

# Espacios de caracteristicas a comparar.
FEATURE_SPACES = ["rgb", "hsv", "hsv_circular", "rgb_spatial", "hsv_circular_spatial"]

# Peso de la informacion espacial (X,Y ya normalizadas a [0,1]).
# lambda alto -> las regiones importan mas que el color.
LAMBDA_SPATIAL_GRID = [0.1, 0.25, 0.5, 1.0, 2.0]
LAMBDA_SPATIAL_DEFAULT = 0.5

# --- Trimap Oxford-IIIT Pet ---
# La codificacion oficial es 1=foreground, 2=background, 3=borde/ambiguo,
# PERO hay que verificarla empiricamente (ver evaluation.verify_trimap_values).
# El borde se IGNORA en el IoU; se mantiene el mismo protocolo en todo el proyecto.
TRIMAP_FOREGROUND = 1
TRIMAP_BACKGROUND = 2
TRIMAP_BOUNDARY = 3
IGNORE_BOUNDARY_IN_IOU = True

# --- Evaluacion ---
SILHOUETTE_SAMPLE_SIZE = 5000   # muestreo de pixeles; silhouette completo es O(n^2)

# --- Split development / test ---
# Se elige K/lambda/config en development y se reporta en test, para no
# seleccionar hiperparametros mirando el IoU de las mismas imagenes.
DEV_FRACTION = 0.5

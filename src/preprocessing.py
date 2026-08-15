"""
Carga de imagenes/mascaras y construccion de vectores de caracteristicas.

Reglas clave:
- La imagen se redimensiona con interpolacion bilineal; la MASCARA con
  nearest-neighbor (nunca bilineal: interpolar etiquetas crea valores invalidos).
- Todas las caracteristicas se estandarizan (media 0, varianza 1) para que la
  distancia euclidiana de K-Means no quede dominada por la escala de un canal.
- El Hue se trata de forma circular y ponderado por saturacion, porque en
  pixeles poco saturados (grises) el Hue es inestable.
"""

from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Verificacion del trimap
# ----------------------------------------------------------------------
def verify_trimap_values(trimap_paths, max_files=200):
    """Inspecciona varios trimaps y devuelve el conjunto de valores unicos.

    Sirve para confirmar empiricamente la codificacion (esperado: {1, 2, 3})
    antes de asumirla. No confiar en la doc: verificar.
    """
    values = set()
    for p in list(trimap_paths)[:max_files]:
        mask = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        values.update(np.unique(mask).tolist())
    return sorted(values)


# ----------------------------------------------------------------------
# Listado de archivos (filtrando basura ._ y .mat)
# ----------------------------------------------------------------------
def list_pairs(images_dir, trimaps_dir):
    """Devuelve pares (imagen, mascara) emparejados por nombre.

    Ignora archivos que empiezan con '._' (basura de macOS) y todo lo que no
    sea .jpg en imagenes (p.ej. .mat). Solo conserva pares completos.
    """
    images_dir = Path(images_dir)
    trimaps_dir = Path(trimaps_dir)

    pairs = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        if img_path.name.startswith("._"):
            continue
        mask_path = trimaps_dir / (img_path.stem + ".png")
        if not mask_path.exists() or mask_path.name.startswith("._"):
            continue
        pairs.append((img_path, mask_path))
    return pairs


# ----------------------------------------------------------------------
# Carga y redimensionado
# ----------------------------------------------------------------------
def load_image(path, resize_to=None):
    """Carga una imagen en RGB (0-255). Resize bilineal si se pide."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if resize_to is not None:
        h, w = resize_to
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return rgb


def load_mask(path, resize_to=None):
    """Carga un trimap como enteros. Resize NEAREST para no inventar etiquetas."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"No se pudo leer la mascara: {path}")
    if mask.ndim == 3:            # por si viniera con canales, quedarse con uno
        mask = mask[:, :, 0]
    if resize_to is not None:
        h, w = resize_to
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.int32)


# ----------------------------------------------------------------------
# Construccion de caracteristicas
# ----------------------------------------------------------------------
def _standardize(features):
    """Estandariza cada columna a media 0, varianza 1. Evita dividir por 0."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    return (features - mean) / std


def _color_features(rgb_image, space):
    """Devuelve la matriz (N, D) de color sin estandarizar, segun el espacio.

    - 'rgb'          -> (R, G, B)
    - 'hsv'          -> (H, S, V) crudo
    - 'hsv_circular' -> (S*cosH, S*sinH, V): Hue circular ponderado por saturacion
    """
    h, w, _ = rgb_image.shape
    if space == "rgb":
        return rgb_image.reshape(-1, 3).astype(np.float64)

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float64)
    # OpenCV para 8-bit: H en [0,179], S y V en [0,255].
    H = hsv[:, :, 0] * (2.0 * np.pi / 180.0)   # a radianes
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0

    if space == "hsv":
        return np.stack([hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]], axis=-1).reshape(-1, 3)
    if space == "hsv_circular":
        feat = np.stack([S * np.cos(H), S * np.sin(H), V], axis=-1)
        return feat.reshape(-1, 3)

    raise ValueError(f"Espacio de color desconocido: {space}")


def _spatial_features(h, w):
    """Coordenadas (x, y) normalizadas a [0,1]. Matriz (N, 2)."""
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs / max(w - 1, 1)
    ys = ys / max(h - 1, 1)
    return np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float64)


def build_features(rgb_image, space, lambda_spatial=0.5):
    """Construye la matriz de caracteristicas (N, D) lista para K-Means.

    space admite: 'rgb', 'hsv', 'hsv_circular', 'rgb_spatial',
    'hsv_circular_spatial'. Cuando lleva '_spatial', se anexan (x,y)
    estandarizadas y multiplicadas por lambda_spatial.

    Todo se estandariza; lo espacial se pondera DESPUES de estandarizar,
    para que lambda controle su peso relativo de forma interpretable.
    """
    h, w, _ = rgb_image.shape
    use_spatial = space.endswith("_spatial")
    color_space = space.replace("_spatial", "") if use_spatial else space

    color = _standardize(_color_features(rgb_image, color_space))

    if not use_spatial:
        return color

    spatial = _standardize(_spatial_features(h, w)) * lambda_spatial
    return np.concatenate([color, spatial], axis=1)

"""
Segmentacion por K-Means sobre una sola imagen.

K-Means se corre POR IMAGEN (no sobre el dataset entero): cada imagen tiene sus
propios clusters. La salida es un mapa de etiquetas (alto, ancho) donde cada
pixel trae el id de su cluster (0..K-1). Ese id NO significa foreground/background
todavia; el mapeo a clases se hace en evaluation.py.
"""

import numpy as np
from sklearn.cluster import KMeans


def segment_image(features, image_shape, k, random_state=42, n_init=10):
    """Corre K-Means sobre 'features' y devuelve el mapa de etiquetas 2D.

    features: matriz (N, D) donde N = alto*ancho (ya construida por preprocessing).
    image_shape: (alto, ancho) para reconstruir el mapa.
    k: numero de clusters.

    Devuelve (labels_2d, model):
      - labels_2d: array (alto, ancho) con el cluster de cada pixel.
      - model: el objeto KMeans ajustado (util para inertia y centroides).
    """
    h, w = image_shape
    if features.shape[0] != h * w:
        raise ValueError(
            f"features tiene {features.shape[0]} filas pero la imagen es {h}x{w}={h*w}"
        )

    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(features)
    labels_2d = labels.reshape(h, w)
    return labels_2d, model


def labels_to_color(labels_2d, rgb_image):
    """Colorea cada cluster con el color RGB promedio de sus pixeles.

    Sirve para visualizar la segmentacion: la imagen reconstruida usando solo
    K colores (el promedio de cada cluster). No interviene en la evaluacion.
    """
    h, w = labels_2d.shape
    out = np.zeros((h, w, 3), dtype=np.float64)
    for cluster_id in np.unique(labels_2d):
        mask = labels_2d == cluster_id
        mean_color = rgb_image[mask].mean(axis=0)
        out[mask] = mean_color
    return out.astype(np.uint8)

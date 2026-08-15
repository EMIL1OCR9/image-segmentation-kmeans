"""
Evaluacion de la segmentacion.

Contiene todo lo que convierte clusters en numeros comparables:
- mapeo oracle cluster->clase (muchos-a-uno) usando la mascara real.
- IoU foreground, ignorando la region de borde del trimap.
- baselines (Otsu, todo-foreground, aleatorio) para dar contexto al IoU.
- metricas internas de clustering (silhouette con muestreo, DB, CH, inertia).
- metrica de coherencia espacial (fragmentacion / componentes conexas).
- correlacion de rangos por imagen entre metrica interna e IoU.

NOTA IMPORTANTE: el IoU que se reporta es "oracle-mapping": para decidir que
cluster es foreground se mira la mascara real. Es una COTA SUPERIOR (mejor caso),
no lo que lograria un sistema sin acceso a la mascara. Reportarlo siempre asi.
"""

import cv2
import numpy as np
from scipy import stats
from scipy import ndimage
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# ----------------------------------------------------------------------
# Mascara binaria de referencia a partir del trimap
# ----------------------------------------------------------------------
def trimap_to_binary(trimap, fg=1, bg=2, boundary=3, ignore_boundary=True):
    """Convierte el trimap (1/2/3) en referencia binaria + mascara de validos.

    Devuelve (gt_fg, valid):
      - gt_fg: booleano, True donde hay foreground segun la referencia.
      - valid: booleano, True donde el pixel cuenta para el IoU. Si
        ignore_boundary=True, la region de borde queda fuera (valid=False).
    """
    gt_fg = trimap == fg
    if ignore_boundary:
        valid = trimap != boundary
    else:
        valid = np.ones_like(trimap, dtype=bool)
    return gt_fg, valid


# ----------------------------------------------------------------------
# Mapeo oracle cluster -> clase (muchos-a-uno)
# ----------------------------------------------------------------------
def map_clusters_to_foreground(labels_2d, gt_fg, valid):
    """Asigna cada cluster a foreground o background por voto mayoritario.

    Para cada cluster mira, SOLO en pixeles validos, si se solapa mas con
    foreground o con background en la referencia, y lo asigna a esa clase.
    Varios clusters pueden ir a la misma clase (muchos-a-uno). Esto usa la
    mascara real -> es el paso 'oracle'.

    Devuelve pred_fg: booleano, prediccion foreground/background por pixel.
    """
    pred_fg = np.zeros_like(labels_2d, dtype=bool)
    for cluster_id in np.unique(labels_2d):
        cluster_mask = labels_2d == cluster_id
        sel = cluster_mask & valid
        if sel.sum() == 0:
            # cluster totalmente dentro del borde ignorado: no asignar fg
            continue
        fg_overlap = (sel & gt_fg).sum()
        bg_overlap = sel.sum() - fg_overlap
        if fg_overlap >= bg_overlap:
            pred_fg[cluster_mask] = True
    return pred_fg


# ----------------------------------------------------------------------
# IoU
# ----------------------------------------------------------------------
def iou_foreground(pred_fg, gt_fg, valid):
    """IoU de la clase foreground, contando solo pixeles validos.

    IoU = |pred AND gt| / |pred OR gt|, restringido a 'valid'.
    Si la union es 0 (no hay foreground en ninguno) devuelve NaN.
    """
    p = pred_fg & valid
    g = gt_fg & valid
    inter = (p & g).sum()
    union = (p | g).sum()
    if union == 0:
        return np.nan
    return inter / union


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
def baseline_otsu(rgb_image):
    """Segmentacion por umbral de Otsu sobre la imagen en escala de grises.

    Metodo clasico basado en intensidad. Devuelve booleano foreground.
    Otsu no sabe cual lado es el objeto; se decide despues con el mapeo oracle
    igual que a K-Means, para comparar en igualdad de condiciones.
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(bool)


def baseline_all_foreground(shape):
    """Prediccion trivial: todo es foreground. Cota inferior de referencia."""
    return np.ones(shape, dtype=bool)


def baseline_random(shape, random_state=42):
    """Prediccion aleatoria 50/50. Referencia adicional."""
    rng = np.random.default_rng(random_state)
    return rng.random(shape) > 0.5


def otsu_as_two_cluster_labels(rgb_image):
    """Devuelve Otsu como mapa de etiquetas 0/1 para pasarlo por el mapeo oracle."""
    fg = baseline_otsu(rgb_image)
    return fg.astype(np.int32)


# ----------------------------------------------------------------------
# Metricas internas de clustering
# ----------------------------------------------------------------------
def internal_metrics(features, labels_2d, sample_size=5000, random_state=42):
    """Calcula metricas internas de clustering.

    Silhouette es O(n^2): se calcula sobre una muestra de pixeles.
    Devuelve dict con silhouette, davies_bouldin, calinski_harabasz.
    Si hay un solo cluster, las metricas no estan definidas -> NaN.
    """
    labels = labels_2d.ravel()
    n_clusters = len(np.unique(labels))
    result = {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    if n_clusters < 2:
        return result

    n = features.shape[0]
    if sample_size is not None and n > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_size, replace=False)
        f_s, l_s = features[idx], labels[idx]
    else:
        f_s, l_s = features, labels

    # tras el muestreo puede quedar un solo cluster; proteger cada metrica
    if len(np.unique(l_s)) >= 2:
        result["silhouette"] = float(silhouette_score(f_s, l_s))
        result["davies_bouldin"] = float(davies_bouldin_score(f_s, l_s))
        result["calinski_harabasz"] = float(calinski_harabasz_score(f_s, l_s))
    return result


# ----------------------------------------------------------------------
# Coherencia espacial (fragmentacion)
# ----------------------------------------------------------------------
def spatial_coherence(pred_fg):
    """Mide que tan fragmentada esta la prediccion de foreground.

    El IoU no ve el ruido sal-y-pimienta; esto si. Devuelve dict con:
      - n_components: numero de componentes conexas de foreground.
      - largest_ratio: fraccion del foreground que cae en el componente mas
        grande (1.0 = una sola region limpia; bajo = muy fragmentado).
    """
    labeled, n_components = ndimage.label(pred_fg)
    total = pred_fg.sum()
    if total == 0:
        return {"n_components": 0, "largest_ratio": np.nan}
    sizes = np.bincount(labeled.ravel())[1:]   # descarta el fondo (label 0)
    largest_ratio = sizes.max() / total
    return {"n_components": int(n_components), "largest_ratio": float(largest_ratio)}


# ----------------------------------------------------------------------
# Relacion metrica interna vs IoU (correlacion de rangos por imagen)
# ----------------------------------------------------------------------
def rank_correlation_per_image(internal_by_k, iou_by_k, method="spearman"):
    """Correlacion de rangos entre una metrica interna e IoU, a lo largo de K.

    internal_by_k, iou_by_k: listas alineadas con el valor de cada metrica para
    los distintos K de UNA imagen. Devuelve el coeficiente (rho o tau).

    Se calcula por imagen y luego se agrega la distribucion entre imagenes.
    Mezclar todas las imagenes en un solo scatter confunde las fuentes de
    variacion; por eso es por imagen.
    """
    a = np.asarray(internal_by_k, dtype=float)
    b = np.asarray(iou_by_k, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 2:
        return np.nan
    a, b = a[ok], b[ok]
    if np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan   # varianza cero: correlacion indefinida
    if method == "spearman":
        return float(stats.spearmanr(a, b).correlation)
    if method == "kendall":
        return float(stats.kendalltau(a, b).correlation)
    raise ValueError(f"metodo desconocido: {method}")


# ----------------------------------------------------------------------
# Resumen de distribucion (media, mediana, IQR)
# ----------------------------------------------------------------------
def summarize(values):
    """Resumen robusto de una lista de valores (ignora NaN).

    Reporta media, mediana e IQR porque el IoU por imagen es muy variable y
    sesgado; la media sola engania.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "iqr": np.nan}
    q1, q3 = np.percentile(v, [25, 75])
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "iqr": float(q3 - q1),
    }

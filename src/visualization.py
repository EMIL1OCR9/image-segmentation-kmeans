"""
Visualizacion de resultados.

Funciones para comparar de un vistazo: imagen original, segmentacion por
K-Means (coloreada), mascara de referencia y prediccion foreground/background.
Nada aqui interviene en las metricas; es solo para inspeccion visual.
"""

import matplotlib.pyplot as plt
import numpy as np


def show_comparison(rgb_image, segmented, gt_fg=None, pred_fg=None,
                    title=None, save_path=None):
    """Muestra en fila: original, segmentada y (si se pasan) referencia y prediccion.

    rgb_image: imagen original RGB.
    segmented: imagen coloreada por cluster (de kmeans_segmentation.labels_to_color).
    gt_fg: booleano opcional, foreground de referencia.
    pred_fg: booleano opcional, foreground predicho.
    save_path: si se da, guarda la figura ahi en vez de (o ademas de) mostrarla.
    """
    panels = [("Original", rgb_image), ("K-Means", segmented)]
    if gt_fg is not None:
        panels.append(("Referencia", gt_fg))
    if pred_fg is not None:
        panels.append(("Prediccion", pred_fg))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, panels):
        if data.ndim == 2:                 # mascara booleana o de etiquetas
            ax.imshow(data, cmap="gray")
        else:
            ax.imshow(data)
        ax.set_title(name)
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def show_k_sweep(rgb_image, segmented_by_k, ks, save_path=None):
    """Muestra la misma imagen segmentada con distintos valores de K, en fila.

    segmented_by_k: lista de imagenes coloreadas, una por cada K en 'ks'.
    Sirve para ver de un vistazo como cambia la segmentacion al subir K.
    """
    n = len(ks) + 1
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for ax, k, seg in zip(axes[1:], ks, segmented_by_k):
        ax.imshow(seg)
        ax.set_title(f"K={k}")
        ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_metric_vs_k(ks, metric_values, metric_name, save_path=None):
    """Grafica una metrica (IoU, silhouette, etc.) contra K para una imagen."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ks, metric_values, marker="o")
    ax.set_xlabel("K")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs K")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig

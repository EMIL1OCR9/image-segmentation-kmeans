# ¿Puede K-Means segmentar objetos?

Estudio experimental de clustering visual para segmentación de imágenes.
Se evalúa hasta dónde llega K-Means para separar objeto/fondo usando solo
características de color, sin aprendizaje supervisado, y dónde se rompe.

## Pregunta central

K-Means no "comprende" imágenes: agrupa píxeles por similitud de color.
El proyecto mide qué tan lejos llega ese enfoque y compara configuraciones.

## Dataset

Oxford-IIIT Pet (7390 imágenes con máscaras trimap). No se incluye en el
repo; se descarga desde https://www.robots.ox.ac.uk/~vgg/data/pets/ y se
coloca en `data/raw/`.

## Metodología

- K-Means por imagen (no sobre el dataset completo).
- K=2 como experimento principal foreground/background; K=2..10 como
  secundario (el IoU sube con K por más grados de libertad, se interpreta
  con cautela).
- Mapeo cluster→clase muchos-a-uno usando la máscara real. Por eso el IoU
  reportado es cota superior (oracle-mapping), no rendimiento sin supervisión.
- Espacios comparados: RGB, HSV crudo, HSV circular ponderado por saturación,
  y variantes con información espacial (X,Y) ponderada por λ.
- Baselines: Otsu, todo-foreground, aleatorio.
- Evaluación: IoU + métricas internas (silhouette, Davies-Bouldin,
  Calinski-Harabasz) + coherencia espacial (componentes conexas).

## Estructura

- `src/` módulos reutilizables (preprocessing, kmeans, evaluación, visualización).
- `notebooks/` experimentación y análisis.
- `data/` datos crudos (no versionado).
- `results/` figuras e imágenes segmentadas.

## Uso

Instalar dependencias con `pip install -r requirements.txt`, descargar el
dataset en `data/raw/` y correr los notebooks en orden.

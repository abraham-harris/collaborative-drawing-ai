"""
Variety metric for the collaborative-drawing-ai project.

Pipeline: load N outputs generated from the SAME seed -> embed each image ->
compute pairwise cosine distance across all embeddings -> report mean/std as
the variety score for that seed.

Embedding used here: HOG (Histogram of Oriented Gradients), via scikit-image.
This is a classical, dependency-light stand-in chosen because torch (needed
for DreamSim/CLIP) couldn't be installed in time in this environment. HOG is
a reasonable placeholder for line-art/sketch content since it captures edge
and gradient structure rather than raw pixels or color.

To upgrade later to DreamSim or CLIP: only `embed_image()` needs to change.
Everything downstream (the distance matrix, the aggregation, the report)
stays the same -- that's the point of keeping embedding and scoring separate.
"""

import glob
import os
import sys

import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.metrics.pairwise import cosine_similarity

IMG_SIZE = (256, 256)  # resize so every image feeds HOG the same shape


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    return np.asarray(img)


def embed_image(img_array: np.ndarray) -> np.ndarray:
    """Turn one image into a fixed-length feature vector (the 'embedding')."""
    gray = rgb2gray(img_array)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    return features


def variety_score(image_paths: list[str]) -> dict:
    """Given N image paths (outputs from the same seed), return the variety
    report: the full pairwise cosine-distance matrix plus summary stats."""
    embeddings = np.stack([embed_image(load_image(p)) for p in image_paths])

    sim_matrix = cosine_similarity(embeddings)       # higher = more similar
    dist_matrix = 1.0 - sim_matrix                    # higher = more varied

    n = len(image_paths)
    iu = np.triu_indices(n, k=1)                       # upper triangle, no diagonal, no double-count
    pairwise_distances = dist_matrix[iu]

    return {
        "n_images": n,
        "distance_matrix": dist_matrix,
        "pairwise_distances": pairwise_distances,
        "mean_distance": float(np.mean(pairwise_distances)),
        "std_distance": float(np.std(pairwise_distances)),
        "min_distance": float(np.min(pairwise_distances)),
        "max_distance": float(np.max(pairwise_distances)),
    }


def print_report(report: dict, labels: list[str]):
    print(f"Variety report over {report['n_images']} images")
    print(f"  mean pairwise cosine distance: {report['mean_distance']:.4f}")
    print(f"  std  pairwise cosine distance: {report['std_distance']:.4f}")
    print(f"  min / max pairwise distance:   {report['min_distance']:.4f} / {report['max_distance']:.4f}")
    print()
    print("Full distance matrix:")
    dm = report["distance_matrix"]
    header = "         " + " ".join(f"{l:>9s}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = " ".join(f"{dm[i, j]:9.4f}" for j in range(len(labels)))
        print(f"{row_label:>9s} {row}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    paths = sorted(glob.glob(os.path.join(folder, "variant_*.png")))
    if not paths:
        print(f"No variant_*.png files found in {folder}")
        sys.exit(1)

    labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    report = variety_score(paths)
    print_report(report, labels)

"""
Variety metric for the collaborative-drawing-ai project.

Three complementary, separately-reported signals:

1. Visual variety: load N outputs generated from the SAME seed -> embed each
   image -> compute pairwise cosine distance across all embeddings -> report
   mean/std. Two embedding backends are supported (see embed_image below):

     - "hog" (default): Histogram of Oriented Gradients via scikit-image.
       Classical, dependency-light, works completely offline. Captures edge
       / gradient structure (shape, orientation) -- not semantics. This is
       what's been used so far.

     - "clip": real semantic embedding (OpenAI CLIP ViT-B/32, via the
       onnx_clip package -- no torch required). This is the DreamSim/CLIP
       upgrade. NOTE: the first time it runs, onnx_clip downloads ~350MB of
       model weights from an S3 bucket. That download is blocked in the
       sandbox this was built in (network allowlist), so it could only be
       written and smoke-tested with a dummy/offline path here, NOT run
       end-to-end against the real weights. It should work fine on a normal
       machine with normal internet -- run `pip install onnx_clip` and pass
       --embedding clip. If the weights can't download, this script prints
       a warning and automatically falls back to HOG rather than crashing.

   To add a third backend later, only embed_image() needs a new branch --
   everything downstream (variety_score, reporting) is backend-agnostic.

2. Isolated (AI-only) variety: same as above, but run against ai_only_*.png
   instead of variant_*.png. Those images contain ONLY the strokes the AI
   itself added (rendered on a blank canvas by make_variants.py from the
   aiActions the backend returns) -- not the fixed human seed stroke. This
   answers "how much does the AI's own contribution vary", with the
   constant human input not diluting the signal. Reported separately from
   (1), right below it, when ai_only_*.png files are present.

3. Subject diversity: reads subjects.json (written by make_variants.py,
   which records the "subject" field the backend returns for each run) and
   reports how many distinct subjects the model drew across the N runs.
   Independent signal, not merged into the visual score.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.metrics.pairwise import cosine_similarity

IMG_SIZE = (256, 256)  # resize so every image feeds the embedder the same shape

# Lazily-created singleton so the (possibly slow/network-dependent) CLIP model
# is only ever loaded once, and only if --embedding clip is actually used.
_clip_model = None


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    return np.asarray(img)


def _get_clip_model():
    global _clip_model
    if _clip_model is None:
        from onnx_clip import OnnxClip  # pip install onnx_clip
        _clip_model = OnnxClip(model="ViT-B/32")
    return _clip_model


def embed_image_hog(img_array: np.ndarray) -> np.ndarray:
    gray = rgb2gray(img_array)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )


def embed_image_clip(img_array: np.ndarray) -> np.ndarray:
    model = _get_clip_model()
    embeddings = model.get_image_embeddings([Image.fromarray(img_array)])
    return embeddings[0]


_clip_warned = False


def embed_image(img_array: np.ndarray, method: str = "hog") -> tuple[np.ndarray, bool]:
    """Turn one image into a fixed-length feature vector (the 'embedding').
    Returns (embedding, fell_back) where fell_back is True if clip was
    requested but this particular image had to use HOG instead."""
    global _clip_warned
    if method == "clip":
        try:
            return embed_image_clip(img_array), False
        except Exception as e:
            if not _clip_warned:
                print(f"  [warning] CLIP embedding failed ({e!r}); falling back to HOG. "
                      f"Check your internet connection / onnx_clip install. "
                      f"(further failures this run will stay silent and also fall back)",
                      file=sys.stderr)
                _clip_warned = True
            return embed_image_hog(img_array), True
    return embed_image_hog(img_array), False


def variety_score(image_paths: list[str], method: str = "hog") -> dict:
    """Given N image paths (outputs from the same seed), return the visual
    variety report: the full pairwise cosine-distance matrix plus summary
    stats."""
    results = [embed_image(load_image(p), method) for p in image_paths]
    embeddings = np.stack([r[0] for r in results])
    n_fallback = sum(1 for r in results if r[1])
    effective_method = method
    if method == "clip" and n_fallback:
        complete = "fully" if n_fallback == len(results) else f"{n_fallback}/{len(results)} images"
        effective_method = f"clip (fell back to hog for {complete})" if n_fallback != len(results) \
            else "clip requested, but unavailable -> ran as hog for every image"

    sim_matrix = cosine_similarity(embeddings)       # higher = more similar
    dist_matrix = 1.0 - sim_matrix                    # higher = more varied

    n = len(image_paths)
    iu = np.triu_indices(n, k=1)                       # upper triangle, no diagonal, no double-count
    pairwise_distances = dist_matrix[iu]

    return {
        "n_images": n,
        "method": effective_method,
        "distance_matrix": dist_matrix,
        "pairwise_distances": pairwise_distances,
        "mean_distance": float(np.mean(pairwise_distances)),
        "std_distance": float(np.std(pairwise_distances)),
        "min_distance": float(np.min(pairwise_distances)),
        "max_distance": float(np.max(pairwise_distances)),
    }


def subject_diversity(subjects: list[str]) -> dict:
    """Given the list of subjects the model reported across N runs, return
    how many distinct subjects appeared and a breakdown of counts."""
    counts: dict[str, int] = {}
    for s in subjects:
        counts[s] = counts.get(s, 0) + 1

    n = len(subjects)
    n_distinct = len(counts)

    return {
        "n_runs": n,
        "n_distinct": n_distinct,
        "diversity_ratio": n_distinct / n if n else 0.0,
        "counts": counts,
    }


def print_report(report: dict, labels: list[str], title: str = "Visual variety report"):
    print(f"{title} over {report['n_images']} images  (embedding: {report['method']})")
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


def print_subject_report(report: dict):
    print()
    print(f"Subject diversity report over {report['n_runs']} runs")
    print(f"  distinct subjects: {report['n_distinct']} / {report['n_runs']}  "
          f"(ratio: {report['diversity_ratio']:.2f})")
    print("  breakdown:")
    for subject, count in sorted(report["counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {subject}: {count}")


def _run_set(folder: str, prefix: str, method: str, title: str):
    paths = sorted(glob.glob(os.path.join(folder, f"{prefix}_*.png")))
    if not paths:
        return None
    labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    report = variety_score(paths, method)
    print_report(report, labels, title=title)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute variety score over N same-seed outputs.")
    parser.add_argument("folder", nargs="?", default=".", help="folder containing variant_*.png (and optionally ai_only_*.png, subjects.json)")
    parser.add_argument("--embedding", choices=["hog", "clip"], default="hog",
                         help="hog = offline, shape-only (default). clip = real semantic embedding, "
                              "requires `pip install onnx_clip` and internet access to download weights once.")
    args = parser.parse_args()
    folder = args.folder

    full_report = _run_set(folder, "variant", args.embedding, "FULL CANVAS -- visual variety report")
    if full_report is None:
        print(f"No variant_*.png files found in {folder}")
        sys.exit(1)

    ai_only_report = _run_set(folder, "ai_only", args.embedding,
                               "AI-ONLY (isolated strokes) -- visual variety report")
    if ai_only_report is not None:
        print()
        print("-" * 60)
        print()

    subjects_path = os.path.join(folder, "subjects.json")
    if os.path.exists(subjects_path):
        with open(subjects_path) as f:
            subjects_map = json.load(f)
        paths = sorted(glob.glob(os.path.join(folder, "variant_*.png")))
        subjects = [subjects_map[os.path.basename(p)] for p in paths if os.path.basename(p) in subjects_map]
        if subjects:
            subject_report = subject_diversity(subjects)
            print_subject_report(subject_report)
    else:
        print()
        print(f"(no subjects.json found in {folder} -- run make_variants.py to generate it, "
              f"or skip if these images didn't come from the real model)")
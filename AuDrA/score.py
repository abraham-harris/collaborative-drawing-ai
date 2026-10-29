"""Run the supplied AuDrA checkpoint on images without the training stack.

Model architecture and preprocessing follow the supplied AuDrA code by
John D. Patterson (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18


HERE = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_model(checkpoint_path: Path) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Sequential(
        OrderedDict(
            [
                ("bc1", nn.BatchNorm1d(model.fc.in_features)),
                ("relu1", nn.ReLU()),
                ("fc1", nn.Linear(model.fc.in_features, 1)),
            ]
        )
    )

    with torch.serialization.safe_globals(
        [(dict, "pytorch_lightning.utilities.parsing.AttributeDict")]
    ):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True, mmap=True
        )
    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def score_image(image: Image.Image, model: nn.Module) -> float:
    """Score one PIL image using the same preprocessing as the batch CLI."""
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.1612] * 3, [0.4075] * 3),
        ]
    )
    prepared = transform(ImageOps.invert(image.convert("RGB"))).unsqueeze(0)
    with torch.inference_mode():
        return model(prepared).item()


def score_images(input_dir: Path, checkpoint_path: Path) -> list[tuple[str, float]]:
    paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found in {input_dir}")

    model = load_model(checkpoint_path)
    scores = []
    for path in paths:
        with Image.open(path) as image:
            scores.append((path.name, score_image(image, model)))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Score drawings with AuDrA")
    parser.add_argument("--input-dir", type=Path, default=HERE / "user_images")
    parser.add_argument("--checkpoint", type=Path, default=HERE / "AuDrA_trained.ckpt")
    parser.add_argument("--output", type=Path, default=HERE / "AuDrA_predictions.csv")
    args = parser.parse_args()

    scores = score_images(args.input_dir, args.checkpoint)
    with args.output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filenames", "predictions"])
        writer.writerows(scores)
    print(f"Scored {len(scores)} image(s) with CPU; wrote {args.output}")


if __name__ == "__main__":
    main()

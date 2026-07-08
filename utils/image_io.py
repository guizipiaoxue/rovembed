from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def collect_images(path: str | Path) -> list[Path]:
    image_path = Path(path)
    if image_path.is_file():
        return [image_path]

    if not image_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {image_path}")

    return sorted(
        file
        for file in image_path.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image_tensor(path: str | Path, device: torch.device | str = "cpu") -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = image_to_tensor(image).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    array = array / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def save_image_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    output = tensor.detach().float().cpu().squeeze(0)
    output = output.clamp(-1.0, 1.0)
    image = tensor_to_image(output)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    output = (tensor + 1.0) * 127.5
    array = output.permute(1, 2, 0).numpy().round().astype(np.uint8)
    return Image.fromarray(array)

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models import UWCNN
from utils import collect_images, load_image_tensor, save_image_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UWCNN image enhancement with PyTorch.")
    parser.add_argument(
        "--input",
        default="test_real",
        help="Input image or directory. Defaults to the original TestCode/test_real layout.",
    )
    parser.add_argument(
        "--output",
        default="sample",
        help="Directory for enhanced images.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to a converted PyTorch .pth checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, for example cpu or cuda.",
    )
    parser.add_argument(
        "--suffix",
        default="_out.png",
        help="Suffix appended to each input stem for the output filename.",
    )
    return parser.parse_args()


def load_weights(model: UWCNN, weights_path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)


class UWCNNEnhancer:
    def __init__(self, weights_path: str | Path | None = None, device: str | torch.device | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = UWCNN().to(self.device)
        if weights_path:
            load_weights(self.model, weights_path, self.device)
        else:
            print("Warning: running UWCNN with randomly initialized weights. Use --weights for trained results.")
        self.model.eval()

    def enhance_tensor(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            return self.model(image).clamp(-1.0, 1.0)

    def enhance_bgr_frame(self, frame: np.ndarray) -> np.ndarray:
        rgb = frame[:, :, ::-1].astype(np.float32)
        tensor = torch.from_numpy(rgb / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
        output = self.enhance_tensor(tensor).detach().cpu().squeeze(0)
        output = (output.permute(1, 2, 0).numpy() + 1.0) * 127.5
        rgb_out = output.round().clip(0, 255).astype(np.uint8)
        return rgb_out[:, :, ::-1].copy()

    def enhance_file(self, image_path: str | Path, output_path: str | Path) -> None:
        image = load_image_tensor(image_path, device=self.device)
        output = self.enhance_tensor(image)
        save_image_tensor(output, output_path)

    def enhance_directory(self, input_path: str | Path, output_dir: str | Path, suffix: str = "_out.png") -> None:
        images = collect_images(input_path)
        if not images:
            raise RuntimeError(f"No images found in {input_path}")

        output_dir = Path(output_dir)
        for image_path in images:
            output_path = output_dir / f"{image_path.stem}{suffix}"
            self.enhance_file(image_path, output_path)
            print(f"{image_path} -> {output_path}")


def main() -> None:
    args = parse_args()
    enhancer = UWCNNEnhancer(weights_path=args.weights, device=args.device)
    enhancer.enhance_directory(args.input, args.output, suffix=args.suffix)


if __name__ == "__main__":
    main()

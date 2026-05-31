from __future__ import annotations

import argparse
import time
from typing import Sequence

import torch


DEFAULT_REPO = "facebookresearch/dinov2"
DEFAULT_MODEL_NAME = "dinov2_vits14"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download DINOv2 weights ahead of time via torch.hub."
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="torch.hub repository.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="DINOv2 model name.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for the warm-up forward pass.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of download attempts.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=2.0,
        help="Wait time between retries.",
    )
    return parser


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return "cuda"
    return "cpu"


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    device = _resolve_device(args.device)

    print(f"repo: {args.repo}")
    print(f"model: {args.model_name}")
    print(f"device: {device}")
    print()

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            model = torch.hub.load(
                repo_or_dir=args.repo,
                model=args.model_name,
                trust_repo=True,
            ).to(device)
            model.eval()

            # Warm-up inference forces the checkpoint download and validates execution.
            dummy = torch.zeros((1, 3, 224, 224), device=device)
            with torch.no_grad():
                output = model(dummy)

            shape = (
                tuple(output.shape)
                if isinstance(output, torch.Tensor)
                else type(output).__name__
            )
            print(f"loaded successfully, output: {shape}")
            return 0
        except Exception as exc:
            last_error = exc
            print(f"attempt {attempt}/{args.retries} failed: {exc}")
            if attempt < args.retries:
                time.sleep(args.retry_wait_seconds)

    raise RuntimeError(
        f"Failed to download DINOv2 weights after {args.retries} attempts."
    ) from last_error


if __name__ == "__main__":
    raise SystemExit(main())

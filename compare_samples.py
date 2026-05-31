from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:
    from dinov2_vectorizer import DINOv2Vectorizer, DINOv2VectorizerConfig


DEFAULT_SAMPLE_DIR = Path(__file__).resolve().parent / "sample"
SCORE_MODES: tuple[str, ...] = ("zero_to_one", "raw_cosine")
AGGREGATION_MODES: tuple[str, ...] = ("max", "mean", "diag_mean")


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return image


def _parse_rotations(value: str) -> tuple[int, ...]:
    rotations = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not rotations:
        raise argparse.ArgumentTypeError("rotations must not be empty.")
    return rotations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare sample images against sample/ref.png using DINOv2 embeddings."
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=DEFAULT_SAMPLE_DIR,
        help="Directory containing ref.png and sample*.png.",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=None,
        help="Reference image path. Defaults to <sample-dir>/ref.png.",
    )
    parser.add_argument(
        "--model-name",
        default="dinov2_vits14",
        help="torch.hub DINOv2 model name.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--rotations",
        type=_parse_rotations,
        default=(0, 90, 180, 270),
        help="Comma-separated rotation angles. Example: 0,90,180,270",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize DINOv2 embeddings before returning them.",
    )
    parser.add_argument(
        "--score-mode",
        choices=SCORE_MODES,
        default="zero_to_one",
        help="Similarity score output mode.",
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=AGGREGATION_MODES,
        default="max",
        help="How to aggregate the rotation similarity matrix.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    DINOv2Vectorizer, _ = _load_vectorizer_types()

    sample_dir = args.sample_dir
    ref_path = args.ref if args.ref is not None else sample_dir / "ref.png"
    sample_paths = sorted(sample_dir.glob("sample*.png"))

    if not sample_paths:
        raise FileNotFoundError(f"No sample*.png files found in: {sample_dir}")

    vectorizer = DINOv2Vectorizer(_create_config(args))
    ref_image = _read_image(ref_path)
    ref_vectors = vectorizer.extract(ref_image)

    print(f"reference: {ref_path}")
    print(f"device: {vectorizer.device}")
    print(f"model: {args.model_name}")
    print(f"embedding dim: {vectorizer.embedding_dim()}")
    print(f"rotations: {args.rotations}")
    print(f"normalize: {args.normalize}")
    print(f"score mode: {args.score_mode}")
    print(f"aggregation mode: {args.aggregation_mode}")
    print()

    _print_table(
        sample_paths=sample_paths,
        ref_vectors=ref_vectors,
        vectorizer=vectorizer,
    )
    return 0


def _load_vectorizer_types():
    from dinov2_vectorizer import DINOv2Vectorizer, DINOv2VectorizerConfig

    return DINOv2Vectorizer, DINOv2VectorizerConfig


def _create_config(args: argparse.Namespace):
    _, DINOv2VectorizerConfig = _load_vectorizer_types()
    return DINOv2VectorizerConfig(
        model_name=args.model_name,
        device=args.device,
        rotations=args.rotations,
        normalize=args.normalize,
        score_mode=args.score_mode,
        aggregation_mode=args.aggregation_mode,
    )


def _print_table(
    sample_paths: Sequence[Path],
    ref_vectors: np.ndarray,
    vectorizer: "DINOv2Vectorizer",
) -> None:
    print(
        f"{'sample':<16} {'score':>10} {'size':>12} "
        f"{'vector shape':>16} {'time ms':>10}"
    )
    print("-" * 69)

    for sample_path in sample_paths:
        start = time.perf_counter()
        sample_image = _read_image(sample_path)
        sample_vectors = vectorizer.extract(sample_image)
        score = vectorizer.compare(ref_vectors, sample_vectors)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        height, width = sample_image.shape[:2]
        size = f"{width}x{height}"
        vector_shape = f"{sample_vectors.shape}"
        print(
            f"{sample_path.name:<16} {score:>10.6f} {size:>12} "
            f"{vector_shape:>16} {elapsed_ms:>10.1f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

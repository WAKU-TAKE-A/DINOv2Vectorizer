from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


__version__ = "v0.9.0"


DeviceMode = Literal["auto", "cpu", "cuda"]
ScoreMode = Literal["zero_to_one", "raw_cosine"]
AggregationMode = Literal["max", "mean", "diag_mean"]


@dataclass(frozen=True)
class DINOv2VectorizerConfig:
    """
    DINOv2画像特徴量の設定。

    前提:
        入力画像は、すでにOBBなどで水平矩形化されたcrop画像。
        OpenCV形式のBGR画像を入力する。

    model_name:
        torch.hubで読み込むDINOv2モデル名。
        初期値は "dinov2_vits14"。

    hub_repo:
        torch.hubの取得元リポジトリ。

    device:
        "auto" : CUDAが使えればCUDA、なければCPU
        "cuda" : CUDAを明示使用
        "cpu"  : CPUを明示使用

    rotations:
        入力crop画像を何度回転して特徴量を取るか。
        初期値は (0, 90, 180, 270)。

    normalize:
        Trueなら特徴量をL2正規化する。
        compareではコサイン類似度を使うため、通常Trueでよい。

    score_mode:
        "zero_to_one" : compare結果を 0.0〜1.0 に変換する。
        "raw_cosine"  : 生のコサイン類似度を返す。

    aggregation_mode:
        compare時に回転ごとの類似度行列をどう集約するか。
        "max"       : 全組み合わせの最大値
        "mean"      : 全組み合わせの平均値
        "diag_mean" : 同じ回転どうしの対角成分平均値

    image_size:
        事前処理時のcenter cropサイズ。

    resize_size:
        事前処理時の短辺リサイズサイズ。

    output_dtype:
        出力ベクトルのdtype。
    """

    model_name: str = "dinov2_vits14"
    hub_repo: str = "facebookresearch/dinov2"
    device: DeviceMode = "auto"
    rotations: tuple[int, ...] = (0, 90, 180, 270)
    normalize: bool = True
    score_mode: ScoreMode = "zero_to_one"
    aggregation_mode: AggregationMode = "max"
    image_size: int = 224
    resize_size: int = 256
    output_dtype: type = np.float32


class DINOv2Vectorizer:
    """
    水平矩形化済みcrop画像からDINOv2特徴量を作成し、
    特徴量同士の類似度を計算するクラス。
    """

    VERSION = __version__

    def __init__(self, config: DINOv2VectorizerConfig | None = None) -> None:
        self.config = config or DINOv2VectorizerConfig()
        self._validate_config()

        self.device = self._resolve_device(self.config.device)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    self.config.resize_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.model = torch.hub.load(
            repo_or_dir=self.config.hub_repo,
            model=self.config.model_name,
            trust_repo=True,
        ).to(self.device)
        self.model.eval()

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        self._validate_image(crop_bgr)

        images = [
            self._to_preprocessed_tensor(self._rotate_bgr(crop_bgr, angle))
            for angle in self.config.rotations
        ]
        batch = torch.stack(images, dim=0).to(self.device)

        with torch.no_grad():
            features = self._encode_image(batch)
            if self.config.normalize:
                features = torch.nn.functional.normalize(features, dim=-1)

        vectors = features.detach().cpu().numpy()
        return vectors.astype(self.config.output_dtype, copy=False)

    def compare(self, a: np.ndarray, b: np.ndarray) -> float:
        self._validate_vector_pair(a, b)

        a_norm = self._l2_normalize_rows(a.astype(np.float32, copy=False))
        b_norm = self._l2_normalize_rows(b.astype(np.float32, copy=False))
        similarity_matrix = a_norm @ b_norm.T
        aggregated_cosine = self._aggregate_similarity_matrix(similarity_matrix)

        if self.config.score_mode == "raw_cosine":
            return self._clamp_raw_cosine(aggregated_cosine)

        if self.config.score_mode == "zero_to_one":
            return self._cosine_to_zero_to_one(aggregated_cosine)

        raise ValueError(f"Unsupported score_mode: {self.config.score_mode}")

    def embedding_dim(self) -> int:
        embed_dim = getattr(self.model, "embed_dim", None)
        if embed_dim is not None:
            return int(embed_dim)

        num_features = getattr(self.model, "num_features", None)
        if num_features is not None:
            return int(num_features)

        raise RuntimeError("Could not determine DINOv2 embedding dimension.")

    def _encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        output = self.model(batch)
        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, dict):
            for key in ("x_norm_clstoken", "cls_token", "pooler_output"):
                value = output.get(key)
                if isinstance(value, torch.Tensor):
                    return value

        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(batch)
            if isinstance(features, dict):
                for key in ("x_norm_clstoken", "cls_token"):
                    value = features.get(key)
                    if isinstance(value, torch.Tensor):
                        return value

        raise RuntimeError("Could not extract a DINOv2 image embedding tensor.")

    def _to_preprocessed_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        return self.preprocess(pil_image)

    @staticmethod
    def _rotate_bgr(image: np.ndarray, angle: int) -> np.ndarray:
        angle = angle % 360

        if angle == 0:
            return image
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        raise ValueError(f"Unsupported rotation angle: {angle}")

    @staticmethod
    def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms <= 0.0, 1.0, norms)
        return x / norms

    @staticmethod
    def _cosine_to_zero_to_one(cosine: float) -> float:
        cosine = DINOv2Vectorizer._clamp_raw_cosine(cosine)
        score = (cosine + 1.0) * 0.5
        return DINOv2Vectorizer._clamp_zero_to_one(score)

    def _aggregate_similarity_matrix(self, similarity_matrix: np.ndarray) -> float:
        if self.config.aggregation_mode == "max":
            return float(np.max(similarity_matrix))
        if self.config.aggregation_mode == "mean":
            return float(np.mean(similarity_matrix))
        if self.config.aggregation_mode == "diag_mean":
            return float(np.mean(np.diag(similarity_matrix)))

        raise ValueError(
            f"Unsupported aggregation_mode: {self.config.aggregation_mode}"
        )

    @staticmethod
    def _clamp_raw_cosine(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        if value < -1.0:
            return -1.0
        if value > 1.0:
            return 1.0
        return float(value)

    @staticmethod
    def _clamp_zero_to_one(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return float(value)

    @staticmethod
    def _resolve_device(device: DeviceMode) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested, but torch.cuda.is_available() is False."
                )
            return "cuda"
        if device == "cpu":
            return "cpu"
        raise ValueError("device must be one of: 'auto', 'cpu', 'cuda'.")

    def _validate_config(self) -> None:
        if not self.config.model_name:
            raise ValueError("model_name must not be empty.")
        if not self.config.hub_repo:
            raise ValueError("hub_repo must not be empty.")
        if self.config.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be one of: 'auto', 'cpu', 'cuda'.")
        if not self.config.rotations:
            raise ValueError("rotations must not be empty.")
        if self.config.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.config.resize_size < self.config.image_size:
            raise ValueError("resize_size must be greater than or equal to image_size.")

        allowed_angles = {0, 90, 180, 270}
        normalized_angles = [angle % 360 for angle in self.config.rotations]
        for angle in normalized_angles:
            if angle not in allowed_angles:
                raise ValueError("rotations must contain only 0, 90, 180, or 270.")
        if len(set(normalized_angles)) != len(normalized_angles):
            raise ValueError("rotations must not contain duplicates after modulo 360.")
        if self.config.score_mode not in ("zero_to_one", "raw_cosine"):
            raise ValueError("score_mode must be one of: 'zero_to_one', 'raw_cosine'.")
        if self.config.aggregation_mode not in ("max", "mean", "diag_mean"):
            raise ValueError(
                "aggregation_mode must be one of: 'max', 'mean', 'diag_mean'."
            )

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("crop_bgr must be a numpy.ndarray.")
        if image.ndim != 3:
            raise ValueError("crop_bgr must have shape (height, width, channels).")
        if image.shape[2] != 3:
            raise ValueError("crop_bgr must have 3 channels.")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("crop_bgr must have positive width and height.")
        if image.dtype != np.uint8:
            raise ValueError("crop_bgr dtype must be np.uint8.")

    def _validate_vector_pair(self, a: np.ndarray, b: np.ndarray) -> None:
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise TypeError("Both vectors must be numpy.ndarray.")
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Both vectors must be 2-dimensional.")
        if a.shape[1] != b.shape[1]:
            raise ValueError(f"Embedding dimension mismatch: {a.shape[1]} != {b.shape[1]}.")

        expected_rotations = len(self.config.rotations)
        if a.shape[0] != expected_rotations:
            raise ValueError(
                f"First vector rotation count mismatch: got {a.shape[0]}, expected {expected_rotations}."
            )
        if b.shape[0] != expected_rotations:
            raise ValueError(
                f"Second vector rotation count mismatch: got {b.shape[0]}, expected {expected_rotations}."
            )


if __name__ == "__main__":
    image_path_1 = "sample_crop_1.jpg"
    image_path_2 = "sample_crop_2.jpg"

    crop_1 = cv2.imread(image_path_1)
    crop_2 = cv2.imread(image_path_2)

    if crop_1 is None:
        raise FileNotFoundError(f"Image not found: {image_path_1}")
    if crop_2 is None:
        raise FileNotFoundError(f"Image not found: {image_path_2}")

    vectorizer = DINOv2Vectorizer(
        DINOv2VectorizerConfig(
            model_name="dinov2_vits14",
            device="auto",
            rotations=(0, 90, 180, 270),
            normalize=True,
            score_mode="zero_to_one",
        )
    )

    vecs_1 = vectorizer.extract(crop_1)
    vecs_2 = vectorizer.extract(crop_2)
    score = vectorizer.compare(vecs_1, vecs_2)

    print("device:", vectorizer.device)
    print("embedding dim:", vectorizer.embedding_dim())
    print("vecs_1 shape:", vecs_1.shape)
    print("vecs_2 shape:", vecs_2.shape)
    print("DINOv2 similarity:", score)

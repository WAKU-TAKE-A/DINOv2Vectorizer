# DINOv2VectorizerConfig

`DINOv2 small` (`dinov2_vits14`) を使って、画像 crop から特徴ベクトルを作り、ベクトル同士の類似度を比較します。

## できること

- OpenCV の BGR 画像から DINOv2 埋め込みを抽出
- `0, 90, 180, 270` 度の回転バリエーションをまとめて扱う
- ベクトル間のコサイン類似度を `max` / `mean` / `diag_mean` で集約
- サンプル画像一式で比較スコアを確認

## ファイル構成

- [dinov2_vectorizer.py](c:/UserData/nishida/Git/DINOv2VectorizerConfig/dinov2_vectorizer.py): ベクトル化と比較の本体
- [compare_samples.py](c:/UserData/nishida/Git/DINOv2VectorizerConfig/compare_samples.py): `sample/ref.png` と `sample*.png` の比較スクリプト
- [download_weights.py](c:/UserData/nishida/Git/DINOv2VectorizerConfig/download_weights.py): DINOv2 重みの事前ダウンロード
- [setup_venv.bat](c:/UserData/nishida/Git/DINOv2VectorizerConfig/setup_venv.bat): Windows 用セットアップ

## セットアップ

```bat
setup_venv.bat
```

手動で入れる場合:

```bat
python -m pip install -r requirements.txt
python download_weights.py --device cpu
```

## 使い方

### サンプル比較

```bat
python compare_samples.py --device cpu
```

CUDA が使える環境なら:

```bat
python compare_samples.py --device cuda
```

### コードから使う

```python
import cv2

from dinov2_vectorizer import DINOv2Vectorizer, DINOv2VectorizerConfig

vectorizer = DINOv2Vectorizer(
    DINOv2VectorizerConfig(
        model_name="dinov2_vits14",
        device="auto",
        rotations=(0, 90, 180, 270),
        normalize=True,
        score_mode="raw_cosine",
        aggregation_mode="max",
    )
)

image_a = cv2.imread("sample/ref.png")
image_b = cv2.imread("sample/sample01.png")

vectors_a = vectorizer.extract(image_a)
vectors_b = vectorizer.extract(image_b)
score = vectorizer.compare(vectors_a, vectors_b)

print(DINOv2Vectorizer.VERSION)
print(score)
```

## スコアの見方

`compare()` は画像そのものではなく、`extract()` で得たベクトルを比較します。  
`score_mode="zero_to_one"` は単にコサイン類似度 `[-1, 1]` を `[0, 1]` に線形変換しただけなので、確率のようには解釈できません。

他の特徴量と組み合わせる前提なら、まず `score_mode="raw_cosine"` で分布を見るのがおすすめです。

## 備考

- この環境では CPU 実行で動作確認済みです。
- `xFormers is not available` 警告は、Transformer の attention などを高速化する追加ライブラリ `xFormers` が未導入という意味です。
- `xFormers` が無くても推論は動作し、精度は変わりません。主な違いは実行速度とメモリ効率で、特に CUDA/GPU 実行時の影響が大きいです。
- CPU 実行が前提なら、`xFormers` は無理に入れなくて大丈夫です。

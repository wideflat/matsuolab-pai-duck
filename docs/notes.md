# ノート

## 環境の版（確定したら記入）
- Mac: Apple M4 Pro, 24 GB, macOS arm64, uv 0.10.4, システム Python 3.13.6, git-lfs 未導入
- jax / jaxlib / mujoco / mujoco_playground / brax: 未確定（GPU 環境で `uv sync` 後に記入）
- GPU / マシン: 未確定（AWS を使う。種類・予算は未決）

## 参考リポジトリ（公式で最新を確認する）
- Open_Duck_Playground
- Open_Duck_reference_motion_generator
- Open_Duck_Mini_Runtime

## メモ

## 2026-10-27 フェーズ 0: clone と依存の確認
clone 先: `external/`（このリポジトリの中。`.gitignore` 済みで git 管理外。無ければ同じ URL から clone し直す）
- https://github.com/apirrone/Open_Duck_Playground
- https://github.com/apirrone/Open_Duck_reference_motion_generator

| リポジトリ | HEAD | 最終コミット日 |
|---|---|---|
| Open_Duck_Playground | b9be205ac64488c23504ca42e5ec790337adeec3 | 2025-08-05 |
| Open_Duck_reference_motion_generator | 3d7bc6fd80e6c35587b2a5279314d946cf63456b | 2025-04-03 |

`uv sync --dry-run` の結果:
- **Playground は Mac で入らない。** `jax[cuda12]` の `jax-cuda12-plugin` が Linux のホイールしか無い。本学習は Linux + NVIDIA GPU（AWS）で行う。Mac 側は再生・確認用に CPU 版 jax の環境を別に作る必要がある（未着手）。要求 Python は >=3.11。
- **reference motion generator は Mac で解決できた。** ただし要求 Python が `==3.10.12`、`placo==0.6.3`（実際のインストールは未実行）。git-lfs が必要で、未導入。
- **`playground/open_duck_mini_v2/data/polynomial_coefficients.pkl` が Playground に同梱されている。** ベースライン学習（フェーズ 5）には Placo での生成（フェーズ 4）が必須ではない可能性がある。生成物との違いは未確認（要確認）。
- ONNX 書き出し: `playground/common/export_onnx.py` が存在する（README にコマンドの記載なし。コードを読んで確認する）。

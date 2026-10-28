# ノート

## 環境の版（確定したら記入）
- Mac: Apple M4 Pro, 24 GB, macOS arm64, uv 0.10.4, システム Python 3.13.6, git-lfs 未導入
- jax / jaxlib / mujoco / mujoco_playground / brax: 未確定（GPU 環境で `uv sync` 後に記入）
- GPU / マシン: AWS（us-east-1）。予算上限 $20（2026-10-27 決定）。候補は g6.xlarge（L4）

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

## 2026-10-27 GPU 環境の調査（GPU: AWS に決定）
**AWS の状態（読み取りのみで確認）**
- CLI 2.34.56、`default` プロファイル、リージョン us-east-1、IAM ユーザー `kkudoadmin`。利用可。
- G / VT のオンデマンド vCPU 枠: 16（g6.xlarge / g5.xlarge は 4 vCPU なので足りる）。G / VT のスポット枠も 16。
- us-east-1 に起動中のインスタンスと EBS ボリュームは無かった。
- 価格（us-east-1、2026-10-27 に API で取得）: g6.xlarge（L4）オンデマンド $0.8048/h。スポットは約 $0.48〜0.79/h（AZ により変動）で、割引は小さく中断リスクがあるためオンデマンドを基本にする。g5.xlarge のスポットは約 $0.47〜0.60/h、g4dn.xlarge のスポットは約 $0.29〜0.32/h。
- 予算 $20 ÷ $0.8048 ≒ 約 24 時間分。EBS 等を含め実質 20 時間前後と見ておく。

**Colab Pro（第三者情報。Google 公式の単位消費表は無い。要・実機確認）**
- 月 $9.99 で 100 コンピュートユニット。1 ユニット約 $0.10。使い切ると無料枠の制限に戻る。
- 消費の目安（単位/時間）: T4 約 1.2〜1.9、L4 約 3、A100 約 10〜15。100 ユニットなら T4 で約 50〜80 時間、L4 で約 33 時間、A100 で約 7〜10 時間。
- ランタイムは最長 12〜24 時間、アイドル約 3 時間で切断。GPU の割り当ては保証されない。
- 結論: 月 $10 は AWS の $20 より安いが、GPU の種類が選べず、長い学習は切断の恐れがある。今回は AWS を主とし、Colab は使わない。

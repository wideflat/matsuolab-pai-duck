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

## 2026-10-27 フェーズ 2: MJCF の確認（Open Duck Mini v2）
確認コマンド: `.venv/bin/python scripts/inspect_mjcf.py external/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml`
- 総質量 2.107 kg、timestep 0.002 s、アクチュエーター 14（脚 5×2 + 首/頭 4）。qpos 31 / qvel 30。
- 脚の関節範囲（度）: hip_yaw ±30、hip_roll ±25、hip_pitch 左 [-70,30] / 右 [-30,70]（左右で符号が逆）、knee ±90、ankle ±90。頭: neck_pitch [-20,65]、head_pitch ±45、head_yaw ±160、head_roll ±30。
- アクチュエーターはすべて position 制御、kp=17.11、出力範囲 ±3.23（単位は MJCF のとおり、トルクとして扱う前に要確認）。
- backlash 版は、各脚関節に ±0.01 rad の受動関節（バックラッシュ）が付く（10 個）。公式の学習タスク名 `flat_terrain_backlash` はこれを使うと推測（要確認）。
- 初期姿勢での接触 0、センサー 15 個（gyro, accelerometer, linvel, 足の位置/速度, 足の向き など）、keyframe は `home`。
- 静止画: `videos/duck_still.png`（git 管理外）。

## 2026-10-27 フェーズ 3: 公開ポリシーの再生
- 入手元: https://github.com/apirrone/Open_Duck_Mini/raw/v2/BEST_WALK_ONNX_2.onnx（Open_Duck_Mini_Runtime の README が案内）。sha256 `3c606f9381a1710cc8fecdb7442787dcbfce3ee9bc02a6f1224774ab2b3a1067`。`external/published_policy/` に置いた（git 管理外）。
- Open_Duck_Mini_Runtime HEAD: 376de65435c93486347cd601a31aa96951799896（2026-09-08）。
- Mac の環境: `.venv`（uv, Python 3.12）に mujoco 3.17.0 / playground 0.2.0 / jax 0.12.0.1（CPU）/ onnxruntime 1.31.1 / numpy 2.5.4 / brax 0.14.2。PyPI のパッケージ名は `mujoco-playground` ではなく `playground`。
- 公式の `mujoco_infer.py` は GUI の viewer 専用のため、同じ制御ループを再実装した `scripts/replay_headless.py` を使い、数値と GIF（`videos/`、git 管理外）を出した。結果は docs/experiments.md の E001。
- 公式の viewer 版（`mjpython` が必要）は未実行。

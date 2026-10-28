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

## 2026-10-27 GPU 環境の決定
AWS（us-east-1、g6.xlarge）を使う。Colab Pro は使わない（ユーザー決定）。予算上限 $20。

## 2026-10-27 フェーズ 4: リファレンスモーション
**同梱の `polynomial_coefficients.pkl`（Open_Duck_Playground, open_duck_mini_v2）の中身**（自分で読んで確認）
- 240 個のモーション。キーは `dx_dy_dtheta`（例 `0.074_-0.037_0.704`）。格子は dx 6 値（-0.148〜0.222）× dy 4 値（±0.111）× dtheta 10 値（-1.111〜1.222）。
- どのモーションも period 0.54 s、50 fps、1 周期 27 ステップ。各モーションは 40 次元 × 16 係数（多項式）。
- 現在の generator の `robots/open_duck_mini_v2/auto_gait.json` の掃引範囲（x -0.04〜0.06、y ±0.03、theta ±0.3）とは**一致しない**。同梱 pkl は別の（古い）設定で作られたもので、出所の記録は無い。**generator を今の設定で回しても同じ pkl にはならない。**

**generator（Placo）は Mac では動かない**
- `placo==0.6.3` の配布 wheel は `manylinux_2_35_x86_64`（cp310）のみ。Mac(arm64) ではソースビルドになり、CMake の設定段階で失敗した。
- Linux x86_64 なら動くはず（未実行）。git-lfs が必要で、STL は `brew install git-lfs`（brew 経由、`git lfs install --local` と `git lfs pull`）で取得済み。
- Python は `==3.10.12` を要求。

**結論**
- ベースライン学習（フェーズ 5）は同梱の pkl をそのまま使えばよい。Placo での再生成は必須ではない。
- 再生成するなら、Linux x86_64 の小さい CPU インスタンスで行う（GPU 時間を使わない）。GPU 予算 $20 の外で費用がかかるため、実施前にユーザーの承認を取る。再生成した pkl は「別条件の実験」として experiments.md に残し、同梱 pkl とは混ぜない。

## 2026-10-27 GPU 起動ログ
- 21:28 -0400 **GPU 起動**: g6.xlarge (L4), us-east-1c, `i-076e1704e8f8776fc`, AMI `ami-041bfdc361a4a68ef`（DLAMI Ubuntu 22.04 20261027）, EBS gp3 100 GB。目的: スモーク学習で steps/s を測る。
- 作成したリソース（すべて `project=matsuolab-pai-duck`）: キーペア `matsuolab-pai-duck`（秘密鍵 `~/.ssh/matsuolab-pai-duck.pem`）、セキュリティグループ `sg-0c41d5282a8d863ab`（22 番を 67.247.48.58/32 のみ）。インスタンスは `instance-initiated-shutdown-behavior=terminate`。
- 終了時に削除するもの: インスタンス（ボリュームは連動削除）、SG、キーペア（再利用するなら残す）。
- 21:48 -0400 **GPU 停止（インスタンス削除）**: `i-076e1704e8f8776fc` を terminate（2026-10-28 01:48:26 GMT）。稼働約 20 分、概算 $0.3（$0.8048/h）。関連ボリュームは残っていないことを確認。残したもの: キーペア `matsuolab-pai-duck`、セキュリティグループ `sg-0c41d5282a8d863ab`（どちらも無料。次回の学習で再利用）。
- 自分の IP が変わったら SG の許可 IP を更新する必要がある。

## 2026-10-27 フェーズ 5 準備: スモーク学習（AWS）
**動く環境（Open_Duck_Playground HEAD b9be205 + 下記の固定）。`configs/requirements-aws-train.txt` に全パッケージの版を保存。**
- Python 3.12.16（uv が既定で 3.15 を選び onnxruntime が入らないので `--python 3.12` が必要）
- `jax[cuda12]==0.7.0`, `jaxlib==0.7.0`, `brax==0.12.5`, `mujoco==3.3.4`, `mujoco-mjx==3.3.4`, `playground==0.0.5`（他は uv の解決どおり: flax / orbax / tensorflow 2.21.0 / onnxruntime 1.31.1 など）
- **公式の `uv sync` のままでは動かない。** 理由: (1) `playground` の最新 0.2.0 には `mujoco_playground._src.collision` が無い（`joystick.py` が import に失敗）。(2) `jax` 0.12 では `jax.device_put_replicated` が削除されており brax 0.14.2 が落ちる。2025 年夏に揃う版へ固定して解決した。
- 学習は `playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps N --output_dir ...`。runner の既定は **1.5 億ステップ**（README の例は 3 億）。
- PPO の設定（ログから）: `clipping_epsilon` 0.2、`num_envs` 8192、`batch_size` 256、`num_minibatches` 32、`num_updates_per_batch` 4、`unroll_length` 20、`discounting` 0.97、`entropy_cost` 0.005、`learning_rate` 3e-4、`episode_length` 1000、`num_evals` 15。観測 101 次元、行動 14 次元。
- 評価ごとにチェックポイント（orbax）と ONNX が自動で保存される。`--restore_checkpoint_path` で再開できる。
- コンパイル込みの立ち上がりは約 3 分。以降 49 万ステップごとに約 18 秒＝**約 2.7 万ステップ/秒**（g6.xlarge, L4）。ここから見積り: **1.5 億ステップ ≒ 1.6 時間（約 $1.3）、3 億ステップ ≒ 3.1 時間（約 $2.5）**。予算 $20 に十分収まる。
- 回収: `checkpoints/smoke_20261028/`（git 管理外、39 MB）。

- 23:13 -0400 **GPU 起動**: g6.xlarge (L4), us-east-1, `i-042e05bd0a5f2b8fc`, 目的: E003 ベースライン学習（5,000 万ステップ）。同じキーペアと SG を再利用。
- E003 の準備で詰まった点: Mac のシェルは zsh で、`ssh $O host`（オプションを変数にまとめる）は**単語分割されず**、鍵が渡らずに `Permission denied (publickey)` になる。ssh の設定ファイル（`Host duck` に HostName / IdentityFile / UserKnownHostsFile をまとめ、`ssh -F <file> duck`）を使えば解決。ホスト鍵は専用の `~/.ssh/known_hosts_matsuolab-pai-duck` に記録する（AWS の IP は使い回されるため、既定の known_hosts を汚さない）。
- 23:17 -0400 E003 学習開始（`--num_timesteps 50000000 --output_dir checkpoints_baseline`）。
- 23:42 -0400 E003 学習終了。速度は 360 万ステップ（評価 1 回分）あたり約 85 秒＝**約 4.2 万ステップ/秒**（スモークの 2.7 万より速い。スモークは評価・保存の負担の比率が大きかった）。回収: `checkpoints/baseline_e003/`（40 MB、15 チェックポイント＋ONNX＋ログ＋TensorBoard）。
- 23:43 -0400 **GPU 停止（インスタンス削除）**: `i-042e05bd0a5f2b8fc` を terminate（03:43:20 GMT）。稼働約 30 分、概算 $0.4。累計の概算 $0.7 / $20。

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
- 10/28 E003 の全 15 チェックポイントを Mac で再生（vx=0.15、10 秒、ノイズなし）→ `docs/data/e003_checkpoints.csv`（`scripts/eval_checkpoints.py`）。前進速度は 3,240 万ステップまで 0.003 m/s 以下で、その後 0.006 → 0.021 → 0.027 → 0.039 → 0.056 m/s と伸びる。評価報酬は約 2,500 万ステップで頭打ち（約 220〜265）なので、**報酬が横ばいのあいだも歩容は改善している**。理由の仮説: 報酬の大半は「立っているだけでもらえる alive=20」で、前進の効き（tracking_lin_vel=2.5）が小さい。0 ステップの方策も転倒しない（立ったまま）。
- 比較動画 v2: `scripts/render_compare.py` で 0 / 25.2M / 39.6M / 50.5M の 4 画面を作成（`videos/e003_compare_v2.mp4`、12 秒、定点カメラ、床に 0.1 m ごとの目印、下に報酬と速度のグラフ）。12 秒で各 +0.00 / +0.01 / +0.21 / +0.69 m。
- 10/28 experiments.md に E004 の行を追加（未実施、判定基準だけ先に登録）。案 A＝前進報酬 2 倍。上書きは `scripts/train_variant.py`（Playground の `config_overrides` を使い、external は改造しない）。playground 0.0.5 の `MjxEnv.__init__` が `config.lock()` → `update_from_flattened_dict` で上書きすることを wheel で確認済み。
- 10:59 -0400 **GPU 起動**: g6.xlarge (L4), us-east-1, `i-0148caba8dc964cd6`, 目的: E004（前進報酬 2 倍、スモーク＋5,000 万ステップ）。ユーザー承認済み。停止条件: インスタンス内で `shutdown -h +90`（終了時は terminate）。
- 11:10 -0400 E004 スモーク（`train_variant.py`、100 万ステップ指定で 2,293,760 まで実行）完走。上書きの反映を確認（`tracking_lin_vel: 5.0`）、NaN なし、報酬 14〜18。続けて 5,000 万ステップの本番を開始。
- 10/28 手本（参照動作）の確認動画: `scripts/render_reference.py` → `videos/reference_vs_e003.mp4`（左＝pkl の手本を物理なしで再生、右＝E003 最終を物理ありで再生、指令 vx=0.15、8 秒）。手本は 8 秒で 1.23 m（0.154 m/s）、E003 は 0.45 m。pkl の 40 次元の並びは generator の `scripts/fit_poly.py` で確認（関節角 16 / 関節速度 16 / 足の接地 2 / 胴体の並進速度 3 / 回転速度 3）。dy の格子に 0 がない（±0.037）ので、「まっすぐ」の手本も横へ 0.044 m/s ずれる。
- 10/28 ダンスの予備確認（E003 最終、Mac、ノイズなし、各 30 秒 1 回）: 7 種の指令（前・左・後・右・左旋回・右旋回・停止）を 1 / 2 / 4 / 8 拍（1 拍 0.54 s）ごとに切り替えても転倒なし。2 拍ごとの切り替えに加えて首（head_yaw ±0.8 rad、head_roll ±0.3 rad）をサーボへ直接正弦波で与えても転倒なし。学習中の指令の切り替えは 10 秒（500 制御ステップ）ごとで、首の指令には報酬がない（観測に入るだけ）ことも joystick.py で確認。
- 10/28 ダンス用の曲: FreePD（CC0）は 2025 年に閉鎖済み。Kevin MacLeod（incompetech.com、CC BY 4.0、YouTube 可・収益化可、クレジット必須）から候補を取得 → `videos/music/`（git 管理外）。クレジット表記: `<曲名> Kevin MacLeod (incompetech.com) / Licensed under Creative Commons: By Attribution 4.0 / https://creativecommons.org/licenses/by/4.0/`（概要欄で可）。
- テンポ実測（`scripts/estimate_bpm.py`、numpy のみ）: Fluffing a Duck **122 BPM**（Shazam の 122 と一致。帯域を区切らない推定は 81 と誤る＝2:3 の取り違え → 帯域ごとに候補を出すよう修正）、Monkeys Spinning Monkeys 144、Pixel Peeker Polka - faster 145、Sneaky Snitch 87。確認用のクリック音入り音声: `videos/music/check_*.m4a`。
- 歩行リズムの変更の予備確認（E003 最終、各 1 回、ノイズなし）: 位相の進み `phase_rate` を 0.8 / 0.9 / 1.0 / 1.098 / 1.2 / 1.3 倍にしても、ダンス（2 拍ごとに 7 種の指令を切り替え、30 秒）、直進（vx=0.15、10 秒）とも転倒なし。直進 10 秒の距離は 0.245 / 0.429 / 0.562 / 0.621 / 0.588 / 0.542 m。→ 歩行の 1 周期を曲の 1 拍に合わせられる（122 BPM なら 1.098 倍、144 BPM なら 1.296 倍）。
- 11:33 -0400 E004 学習終了（50,462,720 ステップ、本番の学習は 11:10〜11:33 の約 23 分）。回収: `checkpoints/e004/`（40 MB、15 チェックポイント＋ONNX＋ログ）。最終 ONNX の sha256 は 2f9d1418。
- 11:34 -0400 **GPU 停止（インスタンス削除）**: `i-0148caba8dc964cd6` を terminate。稼働約 35 分、概算 $0.5。累計の概算 $1.2 / $20。
- 10/28 experiments.md の E004 を結果で更新: **成功**。最終 0.119 m/s（E003 の約 2.1 倍、公開ポリシー 0.089 も上回る）、歩き始めは 28.8M（E003 は 39.6M）。ただし左へ曲がる癖（30 秒で 87°、E003 は 20°）。0 ステップの ONNX が E003 と同じハッシュなので、seed 既定でも初期値は再現している。
- 11:41 -0400 E004 のインスタンス terminate 完了を確認、ボリュームは残っていない。
- 10/28 E003 と E004 の比較動画: `scripts/render_runs.py` → `videos/e003_vs_e004.mp4`（上段 E003・下段 E004、各 28.8M と 50.5M、12 秒、下に速度の推移と上から見た軌跡）と `videos/e003_vs_e004_traj30s.png`（30 秒の軌跡）。12 秒で E003 50.5M は x 0.67 m、E004 50.5M は x 1.41 m・y 0.34 m。
- 動画のコマ送りの不具合を修正: 1 コマ＝17 物理ステップ（0.034 s、1/30 s を丸めた値）で進めていたため、動画が実時間より約 2% 遅く、"12 秒" が実際は 12.2 秒だった。`DuckRollout.advance_to(t)` で時刻に合わせて進めるよう直し、既存の動画（e003_compare_v2、reference_vs_e003）も作り直した。数値の表（eval_checkpoints.py）は物理ステップ数で回していたので影響なし。
- 10/28 ダンスの仕組み: `scripts/dance.py` ＋ `configs/dance_fluffing_a_duck.json`（拍単位の振り付け、歩行の時計を 122 BPM に合わせる phase_rate=1.098、位相 0 を最初の拍に置く、首は sin 波でサーボへ直接、音楽を ffmpeg で合成）。試作 `videos/dance_fluffing_{e004,e003,e004_39m}.mp4`（40.3 秒）はいずれも転倒なし。
- **指令への追従が前進以外は弱い**（10 秒、期待値は後退 -1.5 m／横 2 m／旋回 573°）: E003 は後退 +0.01 m、横 ±0.02 m、旋回 +60° / -126°。E004 は後退 0.00 m、横 +0.24 / -0.20 m、旋回 +101° / -30°。公開ポリシーは後退 -0.31 m、横 +0.74 / -0.68 m、旋回 +86° / -71°。→ 5,000 万ステップでは前進以外がほぼ未習得。ダンスでも後退・カニ歩き・回転は小さくしか動かない。横の報酬には許容幅 y_tol=0.1 m/s があり、指令 0.2 でも 0.1 出れば満点なのも一因の可能性。
- **E004 の途中チェックポイントのほうがまっすぐ歩く**（vx=0.15、30 秒）: 32.4M x 0.54・向き -137°、36.0M x 3.16・-21°、**39.6M x 3.31・+13°**、43.3M x 2.70・+64°、46.9M x 3.01・+62°、50.5M x 2.67・+87°。左へ曲がる癖は 40M 以降に出てきた。30 秒で最も遠くへまっすぐ進むのは 39.6M（sha256 先頭 673a53be）。
- 10/28 後退・横・旋回が育たない理由の仮説（コード読みと計算、未検証）: 指令は学習中にランダムに出され（前後 ±0.15、横 ±0.2、旋回 ±1.0、10 秒ごとに引き直し、1 割は 0）、追従の報酬は exp(-誤差²/0.01)。突っ立ったままでもらえる割合は、前後 0.15 で 0.105、横 0.2 で 0.37（y_tol=0.1 のおかげ）、旋回 1.0 で exp(-100)≈0（0.9 rad/s 回っても 0.37）。**旋回は完璧に近づくまでほぼ点がもらえない**。tracking_sigma は前後・横・旋回で共通なので、緩めると前進の報酬も甘くなる（0.25 なら突っ立ちで 0.91）。
- 10/28 experiments.md に E005 の行を追加（未実施、判定基準だけ先に登録）。E004 の設定のまま 1.5 億ステップ。後退・横・旋回が学習量で身につくかを見る。ユーザー承認済み。
- 12:34 -0400 **GPU 起動**: g6.xlarge (L4), us-east-1, `i-09c5c564ed455814c`, 目的: E005（E004 の設定で 1.5 億ステップ）。ユーザー承認済み。停止条件: インスタンス内で `shutdown -h +110`（終了時は terminate）。
- 10/28 指令ごとの追従を全チェックポイントで測定（`scripts/eval_commands.py`、各 10 秒）→ `docs/data/e003_commands.csv`、`docs/data/e004_commands.csv`。E004 最終は後退 -0.005 m、左 +0.239 m、右 -0.200 m、左回り 101°、右回り -30°。E003 最終は後退 +0.006 m、左 +0.017 m、右 -0.021 m、左回り 60°、右回り -126°（43.3M で -178°）。両方とも後退は全チェックポイントでほぼ 0、転倒は 0 件。E005 の判定はこの表と同じ測り方で行う。
- 10/28 方策が指令をどう使っているかの感度分析（E004 39.6M、前進 vx=0.15 で歩いた 5 秒分の観測 250 個。指令の値だけを差し替えて ONNX を再実行し、関節目標の変化 |Δ| を平均。1 ステップだけの比較で、閉ループではない）: vx を -0.15 にすると平均 3.5°（股関節ピッチ・足首が大きい）、vy=+0.2 で 1.2°（股関節ロール）、wz=+1.0 で 1.7°（股関節ヨー）、首の 4 指令は 0.3〜0.4°（ほぼ無視）。前後は脚を前後に振る関節、横は脚を開く関節、旋回は脚をひねる関節が反応しており、学習で指令と関節の対応はついている。横・旋回は反応が小さく、追従が弱いことと合う。首の指令は報酬がないので、ほぼ無視されている。観測 101 次元の内訳（joystick.py `_get_obs`）: ジャイロ 3、加速度 3、指令 7、関節角 14、関節速度 14、過去 3 回の動作 42、モーター目標 14、足の接地 2、歩行の位相 2。ONNX は入力を (x−平均)/標準偏差で正規化し、512→256→128 の swish 層を経て、出力の前半 14 を tanh。関節目標＝既定姿勢＋0.25×出力 [rad]。
- 13:30 -0400 E005 途中経過（**まだ学習中、108M までの暫定値**、Mac で各指令 10 秒）: 108.1M で後退 -0.587 m、左 +0.624 m、右 -0.238 m、左回り 274°、右回り -217°、前進 0.980 m、転倒なし。後退・左・旋回は判定ラインを超えた。右カニ歩き（0.5 m）と前進（0.1 m/s）がまだ届いていない。50M 以降も後退・横・旋回は伸び続けている。スループット約 4.5 万ステップ/秒、終了は 13:43 ごろの見込み。
- 13:40 -0400 E005 学習終了（151,388,160 ステップ、12:38〜13:40 の約 62 分、約 4.5 万ステップ/秒）。回収: `checkpoints/e005/`（40 MB、15 チェックポイント＋ONNX＋ログ）。最終 ONNX の sha256 は cd46aff2。
- 13:41 -0400 **GPU 停止（インスタンス削除）**: `i-09c5c564ed455814c` を terminate。稼働約 67 分、概算 $0.9。累計の概算 $2.1 / $20。
- 10/28 experiments.md の E005 を結果で更新: **部分的成功**。後退 0.71 m・旋回 348° / -270°（151.4M）が身についたが、全項目を同時に満たすチェックポイントはない（横歩きが左右どちらかに偏る）。13:48 にインスタンスの terminate 完了を確認、ボリュームは残っていない。
- 10/28 E005 最終（151.4M）でダンス → `videos/dance_fluffing_e005.mp4`（40.3 秒、転倒なし）。E004 39.6M と比べ、後退 8 拍が -0.02 → -0.28 m、左回り 8 拍が +24° → +146°、右回り 8 拍が -18° → -108° と、動きがはっきり見えるようになった。カニ歩き 4 拍は ±0.1 m 前後で、まだ小さい。
- 10/28 ダンス v2（Fluffing a Duck、曲 1 本分 67.9 秒、`configs/dance_fluffing_a_duck_v2.json`）。E005 の 151.4M と 129.8M で比較 → 129.8M（sha256 先頭 cbbee2e3）を採用: 8 拍のカニ歩きが左右とも約 0.3 m（151.4M は左 0.09 m）。どちらも転倒なし。**首を振りながら回ると回転が止まる**（8 拍で 32〜38°、首なしは 136〜153°）ので、その動きは外した。最終版 `videos/dance_fluffing_v2.mp4`。終わりの 1.5 秒は音をフェードアウト。
- 10/28 ダンス 2 曲目（Monkeys Spinning Monkeys、144 BPM、曲の頭 60.8 秒、`configs/dance_monkeys_spinning_monkeys.json`、E005 129.8M）→ `videos/dance_monkeys.mp4`。歩行の時計 1.296 倍（試した範囲の端）でも転倒なし。16 拍（6.7 秒）のスピンで左 +222° / +226°、右 -156°、左に弧を描く前進で +206°。斜め 4 方向（4 拍ずつ）を追加。dance.py の回転量の集計が 180° を超えると折り返していた（-138° と表示）のを、毎ステップ積算に直した。
- 10/28 experiments.md に E006 を追加: 道案内役（`scripts/navigator.py`、学習なし、wz=2.5α・20° 以内で前進）で E005 129.8M を 4 つの目標点へ歩かせ、**全点到達・転倒なし**（6.5〜8.9 秒、直線からのずれ最大 7 cm）。まっすぐ歩けない癖は上位の制御で打ち消せる。次はサッカー（E007: ボールに当たっても転ばないか）。
- 10/28 E007（ダックサッカー）途中経過・**3 回続けて失敗したので止めてユーザーに相談**。(a) 当たり試験: 公式モデルは足の裏しか当たり判定がないので、太もも・すね・胴体に「ボールとだけ当たる質量 0 の形」を追加（ロボットの質量・関節・観測は不変を assert）。ボール半径 4 cm・6.5 cm は足の間（11.7 cm）と胴体の下（床から 11.5 cm）をすり抜け、8 cm は胴体の前の下端に乗り上げてダックが持ち上がり 5.2 秒で転倒。**12 cm（ビーチボール、50 g）で転ばずに胴体で押せる**（0.6 m 押した）→ 合格。ゴールも幅 0.6 m・奥行き 0.35 m に広げた（事前登録の 0.44 m から変更）。(b) 本番 5 通り: striker 3 版とも **0/5 ゴール、転倒 0**。原因: ダックはその場旋回で胴体の基準点が 5〜8 cm ずれる（足を軸に回るため）。押し始めにボールの方へ向き直るとこのずれで「ボールが外れた」と判定され、並び直し→向き直りを延々と繰り返す（試行 1 で 7〜45 秒のあいだ前進ゼロ）。次の案: 位置の基準を胴体ではなく両足の中点にする（その場旋回で動かない点）＋押し始めの判定を「ボールの後ろの円錐に入ったか」にする。
- 10/28 experiments.md に E008 を追加: ジャンプの実測（方策なし、手で書いた「しゃがむ→伸ばす」を 72 通り）。**両足が同時に浮いた時間は最大 0.02 秒、足の裏は最大 0.15 cm しか浮かない**。72 回中 61 回は倒れた。判定（1 cm）未満 → ジャンプ・バク転は打ち切り。レポートで「なぜバク転をやらなかったか」の根拠に使える。
- 10/28 E007 の結果を experiments.md に記入: **部分的成功**。striker 4 版（位置を両足の中点で測る＋その場旋回中は判定しない）で、調整に使った 5 か所は 4/5 ゴール・1 回転倒（ボールがゴールポストに挟まり押し続けて倒れた）、**調整に使っていない 3 か所は 3/3 ゴール・転倒なし**。学習なし・方策は E005 129.8M のまま。ユーザー承認の再挑戦 1 回で打ち切り。
- 10/28 サッカーの動画（`scripts/render_soccer.py`、eval_soccer と同じ制御ループ）: `videos/soccer_held_out_03_06.mp4`（調整に使っていない置き場所 (0.3, 0.6)、2 倍速、25.9 秒でゴール）、`videos/soccer_behind_-04_02.mp4`（ボールがダックの後ろ (-0.4, 0.2)、4 倍速、回り込んで 83.5 秒でゴール）。どちらも CSV と同じゴール時刻で再現。
- 10/28 スキップの検討（実験なし）: 本物のスキップは片足で跳ぶ瞬間があり、両足で跳んでも 0.15 cm（E008）なので不可能。参照動作生成器（Open_Duck_reference_motion_generator の gait_generator.py）に `--walk_foot_height` などの引数があるので、足を高く上げる「スキップ風の歩き」なら、参照動作を作り直して学習 1 回（約 1 時間・約 $1）で狙える見込み。ただし生成器の実行に placo が要る（新しい依存）。
- 10/28 E009 準備: 参照動作の生成環境。placo 0.6.3 は manylinux x86_64 の wheel しかなく Mac では build 失敗 → Docker（linux/amd64、python 3.10.12、公式 uv.lock から書き出した `configs/requirements-refgen.txt`）。生成器は読み取り専用でマウントし、external/ は書き換えない。既定値で 240 動作を再生成して公式 pkl と比較: 210 動作は一致（関節角の差の中央値 0.0009 rad）、**30 動作は右ひざの符号が逆**（逆向きに曲がった IK 解）。同じ動作を単独で作ると正しい解になるので、10 並列のマルチスレッド計算の揺らぎが原因と判断。各プロセスを 1 スレッドに固定し、ひざが 0 以下になる動作があれば止める検査を `scripts/refgen_sweep.py` に追加して作り直し中。参考: E005 は前進時に足の裏が 0.8 cm（E003 は 0.4 cm）しか上がらず、参照の 4.5 cm に遠い。
- 10/28 参照動作の再生成を検証: 単独スレッドでも 240 中 23〜30 動作で IK が「ひざが逆に曲がった解」から始まる（全フレームで負）。負荷の小さい状態で作り直すと正しい解になるので、最初のフレームでひざ ≤0 の動作だけ 2 並列で作り直す処理を `scripts/refgen_sweep.py` に入れた。結果、既定値の再生成は公式 pkl と **239/240 動作が 0.002 rad 以内で一致**。残り 1 動作 (0.222, 0.111, 0.963) は**公式 pkl の方が右ひざ逆向き**（公式 -1.3〜-1.8 rad、こちらは +）。足先の上下幅（1 周期）: 既定 中央値 4.52 cm → 8 cm 版 8.92 cm（`runs/refgen/skip/`）。8 cm 版の手本と E005 を並べた動画 `videos/reference_skip_vs_e005.mp4`。
- 21:46 -0400 **GPU 起動**: g6.xlarge `i-071312267ee3ae786`（us-east-1、オンデマンド $0.8048/h）。E009（スキップ風の参照で学習、スモーク後に 1 億ステップ）。停止条件: shutdown -h +120。
- 22:01 -0400 E009 スモーク（300 万ステップ、--reference 8 cm 版）完走: 評価報酬 14→31 で有限・上昇。1 億ステップ本番を開始（`checkpoints_e009`、終了は 22:45 ごろの見込み）。
- 22:08 -0400 E009 本番の設定をログで確認: reference motion ~/ref_skip.pkl（27 steps/period）、imitation 2.0、tracking_lin_vel 5.0。まだ STEP 0 のみ（コンパイル直後）。

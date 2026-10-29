# Open Duck Mini でダンスと高度な動きに挑む

東京大学 松尾・岩澤研究室「Physical AI 応用編1」(知能ロボティクス講座)の最終課題。
MuJoCo Playground (MJX) + PPO (Brax) で小型二足歩行ロボット Open Duck Mini v2 の歩行を学習し、ONNX に書き出してシミュレーター上で動かす。
**実機は使わず、シミュレーターのみ。** 説明資料(レポート)の元は、提出用の 2 ページ版 [docs/report/report_short.html](docs/report/report_short.html) と、詳細版 [docs/report/report.html](docs/report/report.html)(PDF は `scripts/build_report.sh` で作る。PDF は .gitignore で git 管理外、提出は omnicampus)。

## できたこと(すべて自分で測った値。実験の正本は [docs/experiments.md](docs/experiments.md))
| 内容 | 結果 |
|---|---|
| 公式設定で学習(E003) | 5,000 万ステップ、0.056 m/s |
| 前進の報酬を 2 倍(E004) | 歩き始めが約 1,100 万ステップ早く、0.119 m/s(約 2.1 倍) |
| 学習を 1.5 億ステップに延長(E005) | 後退・旋回・横歩きが身についた(全項目同時は満たさず、部分的成功) |
| 道案内役(E006、学習なし) | 4 点とも到達、転倒なし |
| サッカー(E007、学習なし) | 調整した 5 か所で 4 回、未調整の 3 か所で 3 回ゴール(転倒 1 回) |
| ダンス 2 曲(学習なし) | 転倒なし |
| ジャンプ(E008)、スキップ風(E009) | できなかった(失敗として記録) |
| ONNX の一致検査(E010) | JAX との最大差 5.1e-07 |

## 出典(土台にしたもの)
- [apirrone/Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground)(b9be205)、[apirrone/Open_Duck_reference_motion_generator](https://github.com/apirrone/Open_Duck_reference_motion_generator)(3d7bc6f)、Open Duck Mini v2 のモデルと公開ポリシー
- MuJoCo / MuJoCo Playground(Google DeepMind)、Brax、Placo
- 音楽: Fluffing a Duck、Monkeys Spinning Monkeys — Kevin MacLeod (incompetech.com)、CC BY 4.0
- これらのコードは書き換えていない(`external/` は git 管理外で未変更)。**自分の変更は `configs/` と `scripts/`** にある。

## ディレクトリ
- `configs/` 学習設定、ダンスの振り付け、手本生成の Docker 設定
- `scripts/` 学習の上書き(`train_variant.py`)、評価、上位の制御(`navigator.py`、`striker.py`、`dance.py`)、動画・レポートの作成
- `docs/experiments.md` 実験ログ(1 実験 1 行、失敗も残す)、`docs/notes.md` 出来事の記録、`docs/specs/` 計画、`docs/data/` 測定値の CSV、`docs/report/` レポート
- `checkpoints/`, `videos/` は git 管理外

## 再現手順
学習は GPU マシン、再生・評価は手元(Mac)で行った。実験ごとの条件・版・乱数の種・チェックポイントのハッシュは `docs/experiments.md`。
1. 参照リポジトリを取得: `external/` に Open_Duck_Playground(b9be205)と Open_Duck_reference_motion_generator(3d7bc6f)。
2. 学習(AWS g6.xlarge、版は `configs/requirements-aws-train.txt`):
   `python train_variant.py --overrides configs/e004_tracking_lin_vel_x2.json --task flat_terrain_backlash --num_timesteps 150000000 --output_dir checkpoints_e005`
   (Open_Duck_Playground のフォルダで、その venv を使って実行。使い方は `scripts/train_variant.py` の先頭。公式設定のままなら Open_Duck_Playground の `runner.py`)。ONNX は評価ごとに自動で出力される。
3. 評価(使い方は各スクリプトの先頭): `scripts/eval_checkpoints.py`(前進)、`scripts/eval_commands.py`(6 つの指令)。結果は `docs/data/*.csv`。
4. 上位の制御: `scripts/eval_goto.py`(道案内役)、`scripts/eval_soccer.py`(サッカー)、`scripts/dance.py --config configs/dance_*.json`(ダンス)、`scripts/jump_test.py`。
5. 一致検査: `scripts/check_onnx_parity.py`(使い方は先頭の説明)。
6. 動画とレポート: `scripts/assemble_video.py`、`scripts/make_report_figs.py`、`scripts/build_report.sh`。

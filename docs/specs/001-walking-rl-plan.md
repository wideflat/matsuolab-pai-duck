# Spec 001: Open Duck Mini / Microduck 歩行RL 全体計画

> **Date:** 2026-10-27
> **Author:** wideflat & Claude
> **Status:** Planned

## Context
松尾研「フィジカルAI応用編」最終課題。MuJoCo Playground (MJX) + PPO (Brax) で小型二足歩行ロボットの歩行を学習し、ONNX に書き出してシミュレーター上で再生する。
**実機はなく、スコープはシミュレーターまで**（2026-10-27 に確定）。ロボットは Open Duck Mini（v2 を想定）で、Microduck への差し替えと実機実行は行わない。流れの原案は README.md。

## 方針
- **最短で「歩く動画」を作る**ことを最優先にする。まず公開済みの Open_Duck_Playground をそのまま再現し、そのあとで自分の変更（報酬・sim2real 対策）を足す。ゼロから環境を書かない（YAGNI）。
- 1 段階ごとに判定基準を先に決め、`docs/experiments.md` に 1 行残す（`.claude/rules/experiments.md`）。
- GPU は使い終わったらすぐ止める（`.claude/rules/gpu-cost.md`）。学習は GPU 環境、確認・再生は手元の Mac で行う（下の「実行場所」）。

## 事前に確認済みの事実（2026-10-27、公式 README より。版は未確認）
- Open_Duck_Playground: パッケージ管理は `uv`。学習は `uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000`、ログは TensorBoard。
- 模倣報酬を使うには、リファレンスモーション生成リポジトリの `polynomial_coefficients.pkl` を `playground/<robot>/data/` に置き、`joystick.py` の `USE_IMITATION_REWARD=True` にする。
- 再生は `mujoco_infer.py -o <onnx>`（open_duck_mini_v2 のみ対応と記載）。ONNX 書き出しは `playground/common/export_onnx.py` があるが、コマンドは README に無い（**要確認: コードを読む**）。
- Reference motion generator: `uv run scripts/auto_waddle.py --duck open_duck_mini_v2 --sweep --output_dir ...` → `recordings/*.json` → `scripts/fit_poly.py --ref_motion recordings/` で `polynomial_coefficients.pkl`。git-lfs が必要。
- **未確認（着手時に確認）**: Python / jax / mujoco の版、学習に必要な GPU メモリ・所要時間、Placo の要件。

## 実行場所
- **Mac（Apple M4 Pro, 24 GB）**: 環境構築、MJCF の確認、リファレンスモーション生成（CPU）、ポリシーの再生、ONNX 書き出し、スモーク学習（数ステップ）。
- **GPU マシン**: 本学習（フェーズ 1, 5, 6, 7）。公式例は 3 億ステップで、Mac では現実的でない（理由は docs/notes.md に記録）。GPU 環境は未決（AWS か Colab）。

## フェーズ

| # | フェーズ | 場所 | 完了条件（先に決めた判定） |
|---|----------|------|----------------------------|
| 0 | 準備: 2 リポジトリを取得し、版と依存を `docs/notes.md` に固定。GPU 環境（Colab / クラウド）を決定 | 手元 | 依存が入り、`uv run` で runner が `--help` まで動く |
| 1 | 基礎: Playground 付属の歩行環境（例: Go1 Joystick）を短時間学習 | GPU | 報酬曲線が上昇し、動画で前進する。目的は流れの理解なので打ち切り可 |
| 2 | モデル確認: Open Duck Mini の MJCF を `mujoco.viewer` で開く | 手元 | 関節数・可動域・接触・アクチュエーターを `docs/notes.md` に表で残す |
| 3 | 公開ポリシー再生: 公開済みの学習済みポリシーを再生 | 手元 | ビューアで歩く。入手元と版を記録（**自己申告値として扱い、自分の測定と混ぜない**） |
| 4 | リファレンスモーション生成 | 手元（CPU） | `polynomial_coefficients.pkl` が生成され、プロットで滑らかに当てはまっている |
| 5 | ベースライン学習: 公式設定のまま PPO で学習（seed 固定、チェックポイントごとにロールアウト動画） | GPU | 指定速度で 10 秒以上転ばず歩く動画。**転倒率・速度追従誤差を数値で記録** |
| 6 | 自分の変更: 報酬設計の説明と改善（例: 滑らかさ/エネルギー罰の係数）。1 変更ずつ実験 | GPU | ベースラインに対し、事前に決めた指標（転倒率・速度追従誤差・エネルギー）で比較。**悪化も残す** |
| 7 | ロバスト化（sim2real 対策の再現。実機は使わず、シミュレーター内でのランダム化への耐性で評価）: ドメインランダマイゼーション、制御遅延、センサーノイズ、バックラッシュ/アクチュエーターモデルの確認と調整 | GPU | ランダム化範囲を広げた条件でも転倒率が基準以内 |
| 8 | ONNX 書き出しとシミュレーター再生 | 手元 | JAX 出力と ONNX 出力が許容誤差内で一致（`testing.md`）。`mujoco_infer.py` で再生できる |
| 9 | まとめ: 動画、学習曲線、報酬設計の説明、再現手順 | 手元 | README の提出物チェックがすべて埋まる |

**最小ライン**はフェーズ 0, 2, 3, 4, 5, 8, 9。ここまでで「シミュレーションで歩く動画 + 学習曲線 + 再現手順」の3点が揃う。6, 7 は時間と GPU 予算に応じて深める。

## 次の一手
フェーズ 0 から着手する。まず決めること: (a) GPU 環境、(b) 提出期限。

## 未決事項
- 提出期限（逆算してフェーズ 6, 7 の深さを決める）
- GPU 環境と予算上限
- 評価指標の具体値（転倒率・速度追従誤差の合格ライン）は、フェーズ 5 のベースライン実測のあとに、**フェーズ 6 を始める前に**決める

## Simpler alternatives considered (and why rejected)
- **環境を自作する**: 公開環境の再現で目的（歩行、ロバスト化、ONNX）に足りるため採用しない。
- **Placo 生成を飛ばして模倣報酬なしで学習**: 最も簡単だが、講座の流れ（リファレンス追従）と歩容の質を満たさないため、ベースラインは模倣報酬あり。比較用に、余裕があればなしの実験を足す。

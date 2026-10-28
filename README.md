# Microduck / Open Duck Mini 歩行の強化学習

## 背景
東京大学 松尾・岩澤研究室の講座「フィジカルAI応用編」を受講しており、その最終課題として取り組むプロジェクトです。このリポジトリで課題を進めます。

## 概要
MuJoCo Playground (MJX) + PPO (Brax) で小型二足歩行ロボットの歩行を学習し、ONNX に書き出してシミュレーター上で再生する。
実機はないため、スコープはシミュレーターまで。ロボットは Open Duck Mini。

## 進め方
1. 環境構築（mujoco, mujoco_playground, JAX GPU 版。学習は GPU マシン、確認は手元）
2. Playground 付属の歩行環境を 1 つ学習して流れを掴む
3. MJCF を `mujoco.viewer` で確認（関節・可動域・接触）
4. Open Duck Playground の公開ポリシーを再生
5. Placo でリファレンスモーション生成
6. PPO 学習（観測: 関節角・角速度・IMU・速度指令・歩行位相・直前の行動 / 行動: 関節目標角 / 報酬: 参照追従 + 速度追従 + 姿勢 − 滑らかさ・エネルギー罰）
7. 学習の可視化（TensorBoard / wandb、チェックポイントごとのロールアウト動画）
8. sim2real 対策（ドメインランダマイゼーション、制御遅延、センサーノイズ、アクチュエーターモデル）
9. ONNX 書き出しとシミュレーター再生

## ディレクトリ
- `docs/experiments.md` 実験ログ（1 実験 1 行）
- `docs/notes.md` 環境の版・メモ
- `configs/` 学習設定
- `scripts/` 学習・再生・ONNX 書き出し
- `checkpoints/`, `videos/` git 管理外

## 提出物チェック
- [ ] シミュレーションで歩く動画
- [ ] 学習曲線と報酬設計の説明
- [ ] 再現手順

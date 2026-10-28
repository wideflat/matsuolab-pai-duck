# CLAUDE.md

Open Duck Mini の歩行を MuJoCo Playground (MJX) + PPO (Brax) で学習する課題。概要は README.md。**実機はなく、スコープはシミュレーターまで。**

## 背景
東大 松尾研の講座「フィジカルAI応用編」の最終課題として、このリポジトリで進める。提出物は README.md のチェックリスト参照。

## ルール
- 実験は 1 行（条件・結果・結論）を `docs/experiments.md` に追記する。失敗も残す。
- seed、ライブラリの版、チェックポイントのハッシュを必ず記録する。
- `checkpoints/` と `videos/` は git 管理外。
- 外部リポジトリ（Open_Duck_*）の API・設定は、使う前に公式の最新を確認する。記憶で書かない。
- 学習は GPU マシン、手元は確認用（viewer・ロールアウト）。

## Rules and handoff
- Detailed rules live in `.claude/rules/` (English): `experiments.md`, `workflow.md`, `testing.md`, `coding-style.md`, `gpu-cost.md`. Read the relevant one before acting.
- Branches: `main` (default, presentable) and `dev` (daily work). Never commit directly to `main`; never push unless asked.
- Session handoff: `docs/handoff/HANDOFF.md`. Compact hooks in `.claude/hooks/` point at it.
- Progress / events go in `docs/notes.md` (no separate progress notes).

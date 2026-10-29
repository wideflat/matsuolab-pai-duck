# Development Workflow

## Branches and commits

- Two long-lived branches: **`main`** (default; always in a presentable state) and **`dev`**
  (day-to-day work). Committing directly to `dev` is fine. **Never commit directly to `main`.**
- Land on `main` only by merging `dev` into it, as a deliberate step -- never as a side effect
  of finishing a task. Use `--no-ff`.
- Use a `feat/<slug>` branch only when the work is genuinely separable (e.g. a risky reward
  rewrite). Merge it into `dev` with `--no-ff`, then delete it with `git branch -d` (never `-D`).
- Commit with explicit paths (`git add <path> ...`), never `git add -A`, so a stray file
  (checkpoint, video, key) cannot slip in.
- **Commit whenever files change** (authorized by the user, 2026-10-27): after each logical change,
  on `dev`, with explicit paths. Small commits; never batch unrelated changes. This covers
  commits only; push and merging into `main` still need the user's request.
- Never `git push` unless the user asks.

## Commit message format

`<type>: <description>`, optionally followed by a body.
Types: `feat` `fix` `refactor` `docs` `test` `chore` `perf` `exp`
(`exp` = an experiment config/result change that is not a code feature).

## Per-step cycle

1. State what the step produces and how you will know it worked (see `experiments.md`).
2. Smoke-run first (a few training steps) before any long run. Skipping the smoke run is how
   hours of GPU time get wasted.
3. Run the real thing; log the events listed in `experiments.md`.
4. Update `docs/handoff/HANDOFF.md` before stopping.

## Long-running work

- Never edit a shell script while it is running (bash reads scripts lazily by byte offset).
  Copy it to a new name and edit the copy.
- Decide completion by output (final checkpoint exists, log end marker), not by process
  presence (`pgrep -f` matches its own command line).
- If the same step fails 3 times in a row, stop, record the blocker and what was tried in
  `docs/notes.md`, and ask the user. Do not keep retrying one wall.

## Safety rails (every session)

- Never delete user files. Never run destructive commands (`rm -rf` outside the repo,
  `git reset --hard`, force pushes).
- Do not edit files outside this repository.
- No new dependencies unless the user or a documented step names them.
- Never commit checkpoints, videos, API keys, or wandb tokens.
- Scope is simulation only (no real robot). Nothing here sends commands to hardware.

# Running Experiments

Applies to any training run, evaluation sweep, or measurement that will decide something.

- **Write the decision rule before you measure.** Record what counts as success and failure
  *before* looking at the result. If you cannot name what you would do differently under each
  outcome, don't run it.
- **Never mix self-reported numbers with your own measurements.** Figures from papers, repos
  (e.g. Open_Duck_Playground's published results) or other people are claims until reproduced.
  Keep them in their own column, labelled as claims.
- **Keep negative results.** A failed experiment goes in the same place, in the same format.
- **Pin the seed and the environment.** Record the random seed and the versions of Python, jax,
  jaxlib, mujoco, mujoco_playground, brax. Where something cannot be pinned, say so.
- **Identify checkpoints by fingerprint, not filename.** Record sha256 (first 8 hex digits in
  the ledger) plus where it came from. A path can be overwritten; a hash cannot.
- **One experiment, one row** in `docs/experiments.md`: conditions, result, conclusion.

## Events that must be logged the moment they happen

Do not decide for yourself what counts as an event. These three are the list. When one happens,
append a line to `docs/notes.md` **before doing anything else**.

| Event | Where |
|---|---|
| A row was added to `docs/experiments.md` | one line in `docs/notes.md` (what and why it matters) |
| A training run finished | one line in `docs/notes.md` (run, steps, wall time, ckpt hash) |
| A GPU machine was started or stopped | one line in `docs/notes.md`, containing the word `GPU` (the compact hook greps for it) |

**Telling the user a number is not recording it.** If you report a result in chat, write the
line in the same turn. Partial results count: write "only X is out" now and rewrite later.
Chat does not persist; the ledger does.

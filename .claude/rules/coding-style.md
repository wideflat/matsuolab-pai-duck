# Coding Style

## Fail fast, with a narrow scope

Silent defaults hide misconfiguration (a wrong joint order trains fine and fails on the robot).

- **Config values and credentials: no defaults.** Read them explicitly; a missing key must raise.
  `os.environ["WANDB_API_KEY"]`, never `.get("KEY", "")` or `or "default"`.
- No `try/except` that swallows an error and continues. Catch only to add context, then re-raise.
- Validate shapes, joint names/order and units at boundaries (config load, MJCF load, ONNX export).
- Library idioms (function default arguments, `dict.get` inside third-party-style code, JAX/Brax
  patterns) are fine **outside** config/credential handling. Do not fight the framework.

## Immutability

Prefer returning new objects over mutating a caller's. This also matches JAX: functional,
pure functions, `jax.random` keys passed explicitly, no hidden global state.

## File organization

- Many small files over few large ones; 200-400 lines typical, 800 max.
- Put a short header at the top of each file: what it is for and what it guarantees.
  Update it with every change. Read a file's header (and its README) before changing it.
- Organize by feature (env, reward, train, export), not by type.

## Notebooks (Colab)

- Notebooks are for exploration. Logic anything depends on moves into a module in `scripts/`
  and is imported back.
- Re-runnable top to bottom from clean state; no reliance on cell execution order.
- Never overwrite raw inputs (reference motions, MJCF); write derived files to a new path.
- Pin the exact input (path + version/date, or hash) in code, not in memory.
- Check notebook outputs for keys/tokens before committing.

## Timing

Do not use sleeps or timeouts to fix race conditions. Find the root cause or simplify the design.

## Secrets and untrusted input

- Never hardcode a key or token. Never commit one.
- Never load untrusted pickles (`pickle.load`, `torch.load`) -- only checkpoints from a source
  you can name. Prefer formats that cannot execute code on load (ONNX, safetensors, JSON, npz).
- Never build a path from external input without validating it; never `eval`/`exec` it.
- If you find something serious, stop and report it first.

## Debugging

- "Only add logs" means exactly that: do not change logic, structure or execution order.
- Log structured summaries (shape, dtype, min/max, NaN count), not whole arrays.
- Remove temporary `print()` / `breakpoint()` before committing; use `logging` for what stays.

# GPU Machines -- use, then stop

Applies to every GPU environment used for training (Colab, cloud VMs, any rented machine).

## Rules

1. **Stop or delete the machine as soon as the run is done.** A forgotten GPU bills by the hour,
   and a stopped cloud VM still bills its disk. Prefer delete over stop once checkpoints are
   safely copied out.
2. **Copy out before you stop.** Checkpoints, logs and the exact config go to a durable place
   (and the ckpt sha256 into `docs/experiments.md`) *before* the machine goes away. Colab
   runtimes disappear without notice, so save checkpoints periodically, not only at the end.
3. **Log start and stop** in `docs/notes.md` with the word `GPU`: which machine/type, when, why.
   The compact hook greps for it.
4. **Smoke-run first** (`testing.md`) so a bad config is caught in minutes, not hours.
5. **Starting a paid machine needs the user's approval** each time, unless they have said it is
   pre-approved for the current task. State the expected cost and duration.
6. Never leave a long run unattended without a stop condition (max steps / max wall time).

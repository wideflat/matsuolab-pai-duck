"""Draw the figures of the report (docs/report/fig_*.png) from the measured CSVs in docs/data/ and from video frames.

Usage (from the repo root):
  .venv/bin/python scripts/make_report_figs.py

Figures:
  fig_curves.png    evaluation reward and forward speed against training steps (three runs).
  fig_commands.png  distance / angle covered in 10 s under six commands: base run vs long run.
  fig_nav.png, fig_soccer.png, fig_ref.png  single frames cut from videos/final with ffmpeg (needs those clips).
Guarantees: reads only docs/data/*.csv and videos/final; every CSV row used is looked up by its training step and
raises if missing; writes only docs/report/fig_*.png. Palette: dataviz categorical order, text in ink tokens.
"""
import csv
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/report")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e0"
RUNS = [("基本の学習", "e003"), ("前進の報酬を 2 倍", "e004"), ("報酬 2 倍で長く学習", "e005")]
plt.rcParams.update({"font.family": ["Hiragino Sans", "sans-serif"], "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def rows(name, kind):
    with open(f"docs/data/{name}_{kind}.csv") as f:
        return list(csv.DictReader(f))


def row_at(data, steps):
    hit = [r for r in data if int(r["steps"]) == steps]
    if len(hit) != 1:
        raise KeyError(f"steps {steps}: {len(hit)} rows")
    return hit[0]


def curves():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=200)
    for ax, key, label in ((axes[0], "eval_reward", "評価時の報酬"), (axes[1], "mean_vx", "前進の速さ [m/s](指令 0.15)")):
        for (name, tag), c in zip(RUNS, COLORS):
            d = rows(tag, "checkpoints")
            ax.plot([int(r["steps"]) / 1e6 for r in d], [float(r[key]) for r in d], color=c, lw=2, label=name)
        ax.set_xlabel("学習ステップ数 [百万]")
        ax.set_ylabel(label)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[1].axhline(0.15, color=INK2, lw=1, ls=":")
    axes[1].text(2, 0.153, "指令 0.15", color=INK2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_curves.png")


def commands():
    base, long_ = row_at(rows("e003", "commands"), 50462720), row_at(rows("e005", "commands"), 129761280)
    dist = [("前進", "fwd_x_m"), ("後退", "back_x_m"), ("左へ横歩き", "left_y_m"), ("右へ横歩き", "right_y_m")]
    turn = [("左旋回", "turn_l_deg"), ("右旋回", "turn_r_deg")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), dpi=200, gridspec_kw={"width_ratios": [2, 1]})
    for ax, items, unit in ((axes[0], dist, "10 秒で進んだ距離 [m](向きは指令の向きを正)"), (axes[1], turn, "10 秒で回った角度 [度]")):
        x = np.arange(len(items))
        for k, (r, c, lab) in enumerate(((base, COLORS[0], "基本の学習(5,000 万)"), (long_, COLORS[2], "報酬 2 倍で長く学習(1.3 億)"))):
            v = [abs(float(r[key])) for _, key in items]
            b = ax.bar(x + (k - 0.5) * 0.38, v, 0.36, color=c, label=lab)
            for rect, val in zip(b, v):
                ax.text(rect.get_x() + rect.get_width() / 2, val, f"{val:.2f}" if unit.startswith("10 秒で進") else f"{val:.0f}",
                        ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x, [n for n, _ in items])
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_commands.png")


def frame(clip, t, out, crop=None):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", clip, "-frames:v", "1"]
    if crop:
        cmd += ["-vf", crop]
    subprocess.run(cmd + [str(OUT / out)], check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves()
    commands()
    frame("videos/final/clip_navigator.mp4", 9.3, "fig_nav.png")
    frame("videos/final/src/soccer_goal.mp4", 10.0, "fig_soccer.png")
    frame("videos/final/src/reference.mp4", 4.0, "fig_ref.png", "crop=1280:300:0:44")
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()

"""Assemble the final YouTube video from the clips in videos/ and videos/final/ (1280x720, 30 fps, AAC 48 kHz stereo).

Usage (from the repo root):
  PYTHONPATH=scripts .venv/bin/python scripts/assemble_video.py --out videos/final/open_duck_mini_final.mp4

Storyboard (STORY below): title, reference vs E003, E003 vs E004, E003 vs E005 under four commands, navigator,
soccer, dance 1 (Monkeys Spinning Monkeys, the three spins), failures card + two failure clips, dance 2
(Fluffing a Duck, from the second half), end card. Clips that already carry their own captions are used as is;
reference_vs_e003 (344 px high, no explanation) gets a caption through video_cards.caption.
Only the two dance clips keep their music; every other part gets a silent track so the parts can be joined
with stream copy. Dance cut points come from the beat grid in configs/dance_*.json (first_beat + beats/bpm),
see CUTS.

Guarantees: deterministic; reads inputs, writes only videos/final/parts/ and --out; raises if an input is missing
or the result is outside 60-180 s (the course asks for 1-3 minutes).
"""
import argparse
import subprocess
from pathlib import Path

from video_cards import caption

PARTS = Path("videos/final/parts")
# (name, input, cut start s, cut length s or None, fade audio s). Cut times: Monkeys spin_l begins at
# 0.412 + 32 * 60/144 = 13.75 s and spin_r ends at 0.412 + 80 * 60/144 = 33.75 s; Fluffing march_shake begins at
# 0.473 + 64 * 60/122 = 31.95 s and the dance ends at 67.4 s.
CUTS = {
    "dance_monkeys": ("videos/dance_monkeys.mp4", 13.3, 20.5, 0.5),
    "dance_fluffing": ("videos/dance_fluffing_v2.mp4", 31.65, 35.7, 0.3),
}
REF_CAPTION = ("左は手本(Placo の歩行生成器)、右は 5,000 万ステップ学習した E003。同じ指令 0.15 m/s でも、"
               "4 秒で手本は 0.62 m、学習結果は 0.21 m しか進まない")
STORY = [
    "videos/final/card_title.mp4",
    "CAPTION:videos/reference_vs_e003.mp4",
    "videos/e003_vs_e004.mp4",
    "videos/final/clip_e003_vs_e005_commands.mp4",
    "videos/final/clip_navigator.mp4",
    "videos/soccer_held_out_03_06.mp4",
    "CUT:dance_monkeys",
    "videos/final/card_failures.mp4",
    "videos/final/clip_skip_failed.mp4",
    "videos/final/clip_soccer_fall.mp4",
    "CUT:dance_fluffing",
    "videos/final/card_end.mp4",
]
AUDIO_FMT = ["-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k"]
VIDEO_FMT = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", "30"]


def run(cmd):
    subprocess.run(cmd, check=True)


def probe(path, entry="format=duration", stream=None):
    cmd = ["ffprobe", "-v", "error"] + (["-select_streams", stream] if stream else []) + ["-show_entries", entry, "-of", "csv=p=0", str(path)]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def normalize(src, dst, start=None, length=None, fade=0.0):
    """Re-encode one part to the common format. Silent track if the source has no audio."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)
    has_audio = probe(src, "stream=codec_type", "a") != ""
    dur = length if length is not None else float(probe(src)) - (start or 0.0)
    cut = (["-ss", f"{start}"] if start is not None else []) + ["-t", f"{dur}"]
    if has_audio:
        af = f"afade=t=in:st=0:d={fade},afade=t=out:st={dur - fade}:d={fade}" if fade else "anull"
        cmd = ["ffmpeg", "-y", "-loglevel", "error"] + cut + ["-i", str(src), "-af", af, "-map", "0:v", "-map", "0:a"]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error"] + cut + ["-i", str(src), "-f", "lavfi", "-t", f"{dur}",
                                                         "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v", "-map", "1:a"]
    run(cmd + VIDEO_FMT + AUDIO_FMT + ["-shortest", str(dst)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)

    parts = []
    for i, item in enumerate(STORY):
        dst = PARTS / f"{i:02d}.mp4"
        if item.startswith("CAPTION:"):
            tmp = PARTS / f"{i:02d}_caption.mp4"
            caption(item[len("CAPTION:"):], REF_CAPTION, "出発点: 手本どおりに歩く", str(tmp))
            normalize(tmp, dst)
        elif item.startswith("CUT:"):
            src, start, length, fade = CUTS[item[len("CUT:"):]]
            normalize(src, dst, start, length, fade)
        else:
            normalize(item, dst)
        parts.append(dst)
        print(f"{dst}: {float(probe(dst)):5.1f} s  <- {item}")

    listing = PARTS / "list.txt"
    listing.write_text("".join(f"file '{q.name}'\n" for q in parts))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", a.out])
    total = float(probe(a.out))
    print(f"wrote {a.out}: {total:.1f} s")
    if not 60 <= total <= 180:
        raise RuntimeError(f"length {total:.1f} s is outside the 60-180 s the course allows")


if __name__ == "__main__":
    main()

"""Assemble the final YouTube video from the clips in videos/ and videos/final/ (1280x720, 30 fps, AAC 48 kHz stereo).

Usage (from the repo root):
  PYTHONPATH=scripts .venv/bin/python scripts/assemble_video.py --out videos/final/open_duck_mini_final.mp4

Storyboard (STORY below): title, reference vs trained, doubled forward reward, four commands, navigator, soccer,
dance 1 (Monkeys Spinning Monkeys), failures card + two failure clips, dance 2 (Fluffing a Duck, second half),
end card. Every content clip gets a one-line title and an explanation through video_cards.titled; the clips come
from videos/final/src/ (rendered with plain-language labels, no experiment IDs).
Only the two dance clips keep their music; every other part gets a silent track so the parts can be joined
with stream copy. Dance cut points come from the beat grid in configs/dance_*.json (first_beat + beats/bpm),
see CUTS.

Guarantees: deterministic; reads inputs, writes only videos/final/parts/ and --out; raises if an input is missing
or the result is outside 60-180 s (the course asks for 1-3 minutes).
"""
import argparse
import subprocess
from pathlib import Path

from video_cards import titled

PARTS = Path("videos/final/parts")
SRC = Path("videos/final/src")  # clips re-rendered without experiment IDs in their labels
# Dance cut points (s): Monkeys spin_l begins at 0.412 + 32 * 60/144 = 13.75 s and spin_r ends at
# 0.412 + 80 * 60/144 = 33.75 s; Fluffing march_shake begins at 0.473 + 64 * 60/122 = 31.95 s, the dance ends 67.4 s.
# Each STORY item: (input, title, explanation, cut) with cut = None or (start s, length s, audio fade s).
# No experiment IDs (E003...) in any text: the viewer does not know them.
STORY = [
    ("videos/final/card_title.mp4", None, None, None),
    (SRC / "reference.mp4", "まず、手本をまねて歩く",
     "左が手本の歩き、右は手本をまねるよう学習したロボット(基本の設定で 5,000 万ステップ)。同じ前進の指令でも、"
     "4 秒で手本は 0.62 m、学習結果は 0.21 m と、まだ届かない。", None),
    (SRC / "reward_x2.mp4", "前進の報酬を 2 倍にしたら、歩き始めが早くなった",
     "上が基本の設定、下が前進の報酬を 2 倍にしたもの。歩き始めは約 1,100 万ステップ早く、"
     "最終的な速さは 0.056 → 0.119 m/s と約 2 倍になった。", None),
    (SRC / "commands.mp4", "報酬 2 倍のまま長く学習すると、後ろ・回転・横歩きもできるようになった",
     "同じ指令で比べた。左は基本の 5,000 万ステップ、右は報酬 2 倍で 1.3 億ステップ(報酬と長さの両方が違う)。"
     "右は 4 秒で後退 0.27 m、左旋回 133°、横歩き 0.24 m。", None),
    ("videos/final/clip_navigator.mp4", "道案内役: 学習済みの歩きに「行き先の指令」を出す",
     "追加の学習はなし。目標点までの距離と向きのずれを毎回測り、ずれが大きければ旋回、合っていれば前進の指令を出す。"
     "着いたら次の点へ(4 倍速)。", None),
    (SRC / "soccer_goal.mp4", "サッカー: 道案内役を発展させて、ボールをゴールへ押す",
     "道案内役の仕組みの応用。ボールの後ろへ回り込み、ゴールの方向に向きを合わせて押す。追加の学習はなし。"
     "右上の地図の青い線が歩いた跡、赤い枠がゴール(2 倍速)。", None),
    (SRC / "dance_monkeys.mp4", "ダンス: 曲のテンポに合わせて、指令を切り替えるだけ",
     "曲の拍(144 BPM)ごとに「前進・後退・スピン・カニ歩き」の指令を切り替え、首は別に動かす。歩く方策はそのまま。"
     "エンターテインメントとして踊らせた。", (13.3, 20.5, 0.5)),
    ("videos/final/card_failures.mp4", None, None, None),
    (SRC / "skip.mp4", "できなかったこと: スキップ風の歩き",
     "足を 8 cm 高く上げる手本(左)を作り、まねさせて 1 億ステップ学習。学習結果(右)は足をほとんど上げず、すり足のままだった。", None),
    (SRC / "soccer_fall.mp4", "できなかったこと: サッカーでの転倒",
     "ボールがゴールポストに挟まって動かないのに、押し続けて後ろに倒れた(8 通りのうち 1 回)。", None),
    (SRC / "dance_fluffing.mp4", "ダンス(2 曲目): 曲を変えても、振り付けの設定を変えるだけ",
     "拍の数え方と指令の並べ方(振り付けの設定ファイル)を書き換えるだけで別の曲に対応でき、歩きの周期も曲のテンポに合わせている。",
     (31.65, 35.7, 0.3)),
    ("videos/final/card_end.mp4", None, None, None),
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
    for i, (src, title, text, cut) in enumerate(STORY):
        dst = PARTS / f"{i:02d}.mp4"
        if title is not None:
            framed = PARTS / f"{i:02d}_titled.mp4"
            titled(str(src), title, text, str(framed))
            src = framed
        if cut is None:
            normalize(src, dst)
        else:
            normalize(src, dst, *cut)
        parts.append(dst)
        print(f"{dst}: {float(probe(dst)):5.1f} s  <- {src}")

    listing = PARTS / "list.txt"
    listing.write_text("".join(f"file '{q.name}'\n" for q in parts))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", a.out])
    total = float(probe(a.out))
    print(f"wrote {a.out}: {total:.1f} s")
    if not 60 <= total <= 180:
        raise RuntimeError(f"length {total:.1f} s is outside the 60-180 s the course allows")


if __name__ == "__main__":
    main()

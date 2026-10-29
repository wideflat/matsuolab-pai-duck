"""Text cards and captions for the final video (1280x720, 30 fps, Japanese subtitles, no narration).

Subcommands (from the repo root):
  title|end|failures --out X.mp4    static card (6-8 s) with a short fade in/out, no audio track.
  caption --inp A.mp4 --text "..." [--head "..."] --out B.mp4
      Re-encodes a clip onto a 1280x720 canvas with a caption. A 720-high clip gets a dark caption bar over its
      bottom edge; a shorter clip is centred on a light canvas with --head above and --text below. Audio, if any,
      is copied through.
Also provides open_writer() for the render_*.py scripts.

Guarantees: deterministic; every card string is in this file (no external input); caption raises if the input is
not 1280 wide or is taller than 720.
"""
import argparse
import subprocess

from PIL import Image, ImageDraw, ImageFont

from render_compare import COLORS, FONT, FPS, INK, INK2, SURFACE

W, H = 1280, 720
FADE = 0.4  # s
SONGS = ["Fluffing a Duck", "Monkeys Spinning Monkeys"]
CC = "Licensed under Creative Commons: By Attribution 4.0 / https://creativecommons.org/licenses/by/4.0/"


def font(size):
    return ImageFont.truetype(FONT, size)


def wrap(draw, text, f, max_w):
    lines, cur = [], ""
    for ch in text:
        if ch == "\n" or draw.textlength(cur + ch, font=f) > max_w:
            lines.append(cur)
            cur = "" if ch == "\n" else ch
        else:
            cur += ch
    return lines + [cur]


def open_writer(out, w, h, audio_from=None):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-"]
    if audio_from:
        cmd += ["-i", audio_from, "-map", "0:v", "-map", "1:a?", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def blank():
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 18, H], fill=COLORS[0])
    return img, d


def card_title():
    img, d = blank()
    d.text((90, 150), "東大 松尾研「フィジカルAI応用編」最終課題", font=font(26), fill=INK2)
    d.text((90, 215), "Open Duck Mini を", font=font(80), fill=INK)
    d.text((90, 315), "強化学習で歩かせる", font=font(80), fill=INK)
    d.text((90, 450), "MuJoCo Playground + PPO で学習  /  学習済みの方策の上に制御を重ねて、", font=font(28), fill=INK2)
    d.text((90, 492), "点から点への移動、サッカー、ダンスまで", font=font(28), fill=INK2)
    d.text((90, 620), "すべてシミュレーション(実機は使っていません)", font=font(24), fill=INK2)
    return img


def card_end():
    img, d = blank()
    d.text((90, 70), "使ったもの", font=font(40), fill=INK)
    left = ["Open Duck Mini v2 (Open Duck project)",
            "github.com/apirrone/Open_Duck_Playground",
            "github.com/apirrone/Open_Duck_reference_motion_generator",
            "MuJoCo Playground / MuJoCo (Google DeepMind)",
            "Brax (PPO)"]
    for i, t in enumerate(left):
        d.text((90, 140 + i * 40), t, font=font(24), fill=INK2)
    d.text((90, 390), "音楽", font=font(40), fill=INK)
    y = 455
    for song in SONGS:
        d.text((90, y), f"{song}  Kevin MacLeod (incompetech.com)", font=font(24), fill=INK)
        for line in wrap(d, CC, font(19), W - 180):
            y += 28
            d.text((90, y + 14), line, font=font(19), fill=INK2)
        y += 60
    return img


def card_failures():
    img, d = blank()
    d.text((90, 60), "できなかったこと", font=font(56), fill=INK)
    rows = [("ジャンプ", "72 通りの「しゃがんで伸ばす」動きを試したが、両足が浮いたのは最大 0.02 秒・0.15 cm。バク転に要る高さの見積もりは約 11 cm"),
            ("スキップ風の歩き", "足を 8 cm 上げる手本で 1 億ステップ学習しても、足は 0.4 cm しか上がらず(E005 は 0.8 cm)。方策は手本の足の高さをまねしなかった"),
            ("サッカー", "ゴールポストにボールが挟まると、押し続けて転倒(8 通り中 1 回)")]
    y = 170
    for i, (head, body) in enumerate(rows):
        d.rectangle([90, y + 4, 98, y + 52], fill=COLORS[1])
        d.text((120, y), head, font=font(34), fill=INK)
        for j, line in enumerate(wrap(d, body, font(24), W - 240)):
            d.text((120, y + 50 + j * 34), line, font=font(24), fill=INK2)
        y += 160 + (34 if len(wrap(d, body, font(24), W - 240)) > 1 else 0)
    return img


def write_card(img, out, seconds):
    n = int(seconds * FPS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-vf", f"fade=t=in:st=0:d={FADE},fade=t=out:st={seconds - FADE}:d={FADE}",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out], stdin=subprocess.PIPE)
    raw = img.tobytes()
    for _ in range(n):
        ff.stdin.write(raw)
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def caption(inp, text, head, out):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=s=x:p=0", inp], capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(v) for v in probe.split("x"))
    if w != W or h > H:
        raise ValueError(f"{inp} is {w}x{h}; expected width {W} and height <= {H}")
    rd = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", inp, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                          stdout=subprocess.PIPE)
    wr = open_writer(out, W, H, audio_from=inp)
    f_cap, f_head = font(28), font(36)
    base = Image.new("RGB", (W, H), SURFACE)
    bd = ImageDraw.Draw(base)
    if h < H:
        y0 = 110
        bd.text((40, 30), head or "", font=f_head, fill=INK)
        for i, line in enumerate(wrap(bd, text, f_cap, W - 80)):
            bd.text((40, y0 + h + 40 + i * 40), line, font=f_cap, fill=INK)
    while True:
        buf = rd.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        frame = Image.frombytes("RGB", (w, h), buf)
        if h < H:
            canvas = base.copy()
            canvas.paste(frame, (0, y0))
        else:
            canvas = frame
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            lines = wrap(od, text, f_cap, W - 80)
            top = H - 30 - 40 * len(lines)
            od.rectangle([0, top - 14, W, H], fill=(0, 0, 0, 170))
            for i, line in enumerate(lines):
                od.text((40, top + i * 40), line, font=f_cap, fill="white")
            canvas = Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB")
        wr.stdin.write(canvas.tobytes())
    wr.stdin.close()
    if wr.wait() != 0 or rd.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["title", "end", "failures", "caption"])
    p.add_argument("--out", required=True)
    p.add_argument("--inp")
    p.add_argument("--text")
    p.add_argument("--head", default="")
    a = p.parse_args()
    if a.cmd == "caption":
        if not (a.inp and a.text):
            raise SystemExit("caption needs --inp and --text")
        caption(a.inp, a.text, a.head, a.out)
    else:
        card, secs = {"title": (card_title, 6), "end": (card_end, 7), "failures": (card_failures, 9)}[a.cmd]
        write_card(card(), a.out, secs)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

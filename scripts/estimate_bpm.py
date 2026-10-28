"""Estimate the tempo (BPM) and first-beat time of an audio file, numpy only (no librosa).

Method: decode to mono with ffmpeg, build an onset-strength envelope (positive spectral flux of the
log-magnitude STFT), pick the autocorrelation peak in each tempo band, refine it with a comb (sum of
onset strength on a beat grid), then choose the grid phase with the most onset strength.
One candidate is printed per band because a single global pick confuses 2:3 and 1:2 tempo ratios
(Fluffing a Duck came out 81 instead of 122). Choose the band by listening or a trusted listing.

Usage:
  .venv/bin/python scripts/estimate_bpm.py "videos/music/Fluffing a Duck.mp3"

Guarantees: deterministic; reads the file only through ffmpeg; prints, per band, BPM, first-beat time
and a confidence ratio (grid strength / mean strength; ~1 means no clear beat).
"""
import argparse
import subprocess

import numpy as np

SR, HOP, NFFT = 22050, 256, 1024
BANDS = [(60, 100), (100, 140), (140, 180)]


def decode(path, seconds):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-t", str(seconds), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, check=True, capture_output=True).stdout, dtype=np.float32)


def onset_envelope(x):
    n = 1 + (len(x) - NFFT) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP][:n] * np.hanning(NFFT)
    mag = np.log1p(100 * np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(0, np.diff(mag, axis=0)).sum(axis=1)
    return flux - np.convolve(flux, np.ones(16) / 16, mode="same")


def grid_strength(env, period, phase):
    idx = np.arange(phase, len(env) - 1, period)
    i0 = np.floor(idx).astype(int)
    return np.mean(env[i0] * (1 - (idx - i0)) + env[i0 + 1] * (idx - i0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--seconds", type=float, default=60)
    a = p.parse_args()

    env = onset_envelope(decode(a.audio, a.seconds))
    for lo, hi in BANDS:
        estimate(env, lo, hi)


def estimate(env, min_bpm, max_bpm):
    fps = SR / HOP
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lags = np.arange(int(fps * 60 / max_bpm), int(fps * 60 / min_bpm) + 1)
    coarse = 60 * fps / lags[np.argmax(ac[lags])]
    bpms = np.arange(coarse - 3, coarse + 3, 0.05)
    best = max(((max(grid_strength(env, 60 * fps / b, ph) for ph in np.arange(0, 60 * fps / b, 0.5)), b)
                for b in bpms))
    bpm = best[1]
    period = 60 * fps / bpm
    phase = max(np.arange(0, period, 0.25), key=lambda ph: grid_strength(env, period, ph))
    conf = grid_strength(env, period, phase) / np.mean(np.abs(env))
    print(f"[{min_bpm}-{max_bpm}] bpm={bpm:.2f} first_beat={phase / fps:.3f}s beat={60 / bpm:.4f}s confidence={conf:.2f}")


if __name__ == "__main__":
    main()

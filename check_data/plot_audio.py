#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

try:
    import soundfile as sf
    _USE_SF = True
except Exception:
    _USE_SF = False
    from scipy.io import wavfile


PATTERN = re.compile(r"^M_(\d+)_(\d+)\.wav$", re.IGNORECASE)


def read_wav(path: str):
    """
    返回: sr(int), audio(np.ndarray float32, shape [T])
    若是多通道 wav, 会转成单通道(取第1列)
    """
    if _USE_SF:
        audio, sr = sf.read(path, always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = audio[:, 0]
        audio = audio.astype(np.float32, copy=False)
        return sr, audio

    sr, audio = wavfile.read(path)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio[:, 0]

    if audio.dtype.kind in ("i", "u"):
        maxv = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / float(maxv)
    else:
        audio = audio.astype(np.float32, copy=False)

    return int(sr), audio


def group_files(folder: str):
    groups = defaultdict(dict)  # groups[chunk][channel] = filepath
    for name in os.listdir(folder):
        m = PATTERN.match(name)
        if not m:
            continue
        chunk = int(m.group(1))
        ch = int(m.group(2))
        groups[chunk][ch] = os.path.join(folder, name)
    return groups


def plot_chunk(chunk_id: int, ch2path: dict, out_path: str, title_prefix: str = "M"):
    channels = sorted(ch2path.keys())
    if not channels:
        return

    srs = []
    audios = []
    lengths = []
    for ch in channels:
        sr, a = read_wav(ch2path[ch])
        srs.append(sr)
        audios.append(a)
        lengths.append(len(a))

    sr0 = srs[0]
    for sr in srs[1:]:
        if sr != sr0:
            raise ValueError(f"Chunk {chunk_id} 采样率不一致: {srs}")

    min_len = min(lengths)
    if any(L != min_len for L in lengths):
        audios = [a[:min_len] for a in audios]

    t = np.arange(min_len, dtype=np.float32) / float(sr0)

    n = len(channels)
    fig, axes = plt.subplots(n, 1, figsize=(16, max(2.2 * n, 4)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ch, a in zip(axes, channels, audios):
        ax.plot(t, a, linewidth=0.8)
        ax.set_ylabel(f"ch {ch}")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{title_prefix}_{chunk_id} waveforms, sr={sr0}, T={min_len}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="包含 M_<chunk>_<channel>.wav 的文件夹")
    parser.add_argument("--outdir", type=str, default=None, help="输出图片目录，默认在 folder 下建 plots")
    parser.add_argument("--max_chunks", type=int, default=None, help="最多处理多少个 chunk，默认全部")
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    outdir = args.outdir
    if outdir is None:
        outdir = os.path.join(folder, "plots")
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    groups = group_files(folder)
    if not groups:
        raise FileNotFoundError("没找到符合 M_<chunk>_<channel>.wav 的文件")

    chunk_ids = sorted(groups.keys())
    if args.max_chunks is not None:
        chunk_ids = chunk_ids[: args.max_chunks]

    for chunk_id in chunk_ids:
        out_path = os.path.join(outdir, f"chunk_{chunk_id:04d}.png")
        plot_chunk(chunk_id, groups[chunk_id], out_path, title_prefix="M")
        print(f"saved: {out_path}")

    print(f"done. total chunks: {len(chunk_ids)}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import sys
import glob
import re

ROOT = "/uufs/sci.utah.edu/projects/smartair/Dataset/Video/MEB/12.03.22"
SUBFOLDERS = [
    "12.03.22_a",
    "12.03.22_b",
    "12.03.22_c",
    "12.03.22_d",
    "12.03.22_e",
    "12.03.22_f",
]

START_IDX = 64       # 从 00064 开始
GLOBAL_END_CAP = 2935  # 全局上限 02935
STEP = 25             # 每隔 4 帧

def numeric_tail(path, default=0):
    # 提取如 task_video_test_x_7 的最后数字用于排序
    m = re.search(r"_(\d+)$", os.path.basename(path))
    return int(m.group(1)) if m else default

def max_frame_index(rgb_dir):
    # 找到该 rgb images 目录里五位数字命名的 jpg 的最大索引
    files = glob.glob(os.path.join(rgb_dir, "[0-9][0-9][0-9][0-9][0-9].jpg"))
    if not files:
        return None
    nums = []
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        if name.isdigit():
            nums.append(int(name))
    return max(nums) if nums else None

def collect_all_jpg_paths():
    all_paths = []
    for sub in SUBFOLDERS:
        base_dir = os.path.join(ROOT, sub)
        if not os.path.isdir(base_dir):
            continue

        task_dirs = glob.glob(os.path.join(base_dir, "task_video_test_*"))
        task_dirs.sort(key=numeric_tail)

        for tdir in task_dirs:
            rgb_dir = os.path.join(tdir, "rgb-images")
            if not os.path.isdir(rgb_dir):
                continue

            max_idx = max_frame_index(rgb_dir)
            if max_idx is None:
                continue

            # 每个目录单独计算结束索引
            local_end = min(GLOBAL_END_CAP, max_idx - 64)
            if local_end < START_IDX:
                # 没有足够的帧可采样
                continue

            for idx in range(START_IDX, local_end + 1, STEP):
                fpath = os.path.join(rgb_dir, f"{idx:05d}.jpg")
                if os.path.isfile(fpath):
                    all_paths.append(os.path.abspath(fpath))
    return all_paths

def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "meb_rgb_paths.txt"
    jpg_paths = collect_all_jpg_paths()

    with open(out_path, "w") as f:
        for p in jpg_paths:
            f.write(p + "\n")

    print(f"共写入 {len(jpg_paths)} 行到 {out_path}")

if __name__ == "__main__":
    main()

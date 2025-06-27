#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from collections import defaultdict

def fill_missing_frames(input_file, output_file):
    """
    读取 input_file，每一行包含一张图像的路径。
    解析出 (prefix + division + midfix) 以及帧号 frame_id，
    按 (prefix + division) 分组，找出各自最小和最大帧号，并将中间缺失的帧号补齐。
    最后将补齐后的所有路径输出到 output_file。
    """
    # 该正则用于分解每一行的路径：
    # 分成四段：
    #   group(1): prefix (division 之前的那部分路径)
    #   group(2): division，例如 task_video_test_a_0
    #   group(3): midfix，一般是 /rgb-images/ 
    #   group(4): frame_str，形如 00064
    pattern = re.compile(r'^(.*?)(task_video_test_[^/]+)(/rgb-images/)(\d+)\.jpg$')

    # grouped_data 用于存放分组后的信息，结构：
    # {
    #   (prefix + division): {
    #       "prefix": ...,
    #       "division": ...,
    #       "midfix": ...,
    #       "frames": set([...]),
    #   },
    #   ...
    # }
    grouped_data = defaultdict(lambda: {
        "prefix": "",
        "division": "",
        "midfix": "",
        "frames": set()
    })

    # 1. 读取文件，按行匹配正则，提取信息
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 空行跳过

            match = pattern.match(line)
            if not match:
                # 若与预期格式不符，可视需要选择抛出异常或直接跳过
                # 这里我们选择跳过
                continue

            prefix_part = match.group(1)  # division 之前的路径
            division_name = match.group(2)  # 例如 task_video_test_a_0
            midfix_part = match.group(3)  # 例如 /rgb-images/
            frame_str = match.group(4)    # 例如 00064
            frame_id = int(frame_str)

            key = prefix_part + division_name  # 用于区分不同分组(前缀+division)

            # 初始化或更新 grouped_data
            grouped_data[key]["prefix"] = prefix_part
            grouped_data[key]["division"] = division_name
            grouped_data[key]["midfix"] = midfix_part
            grouped_data[key]["frames"].add(frame_id)

    # 2. 对每个分组补齐缺失帧，并写出到 output_file
    with open(output_file, "w", encoding="utf-8") as out_f:
        for key, info in grouped_data.items():
            frames_list = sorted(info["frames"])
            if not frames_list:
                continue  # 没有帧就跳过（理论上不会出现）

            min_id = frames_list[0]
            max_id = frames_list[-1]

            prefix_part = info["prefix"]      # 例: ../Dataset/Video/LDS/04.10.23/04.10.23_a/
            division_name = info["division"]  # 例: task_video_test_a_0
            midfix_part = info["midfix"]      # 例: /rgb-images/

            # range(min_id, max_id+1) 就涵盖了从最小帧到最大帧
            for fid in range(min_id, max_id + 1):
                # 通常帧号要补足零位，一般是 5 位或者 6 位，看原文件的格式
                # 观察到示例多为 5 位数字，如 00064，所以这里用 5 位宽度
                fid_str = f"{fid:05d}"
                # 拼接成完整路径
                full_path = f"{prefix_part}{division_name}{midfix_part}{fid_str}.jpg"
                out_f.write(full_path + "\n")


if __name__ == "__main__":
    # 你可根据需要修改下面这两个文件名
    input_txt = "trainlist_e2e_new_1.txt"
    output_txt = "trainlist_e2e_new_1_filled.txt"

    fill_missing_frames(input_txt, output_txt)
    print(f"已完成补帧并输出到: {output_txt}")

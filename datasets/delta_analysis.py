#!/usr/bin/env python3
# compute_coord_disp_by_label.py

import json
import argparse
from collections import defaultdict

def compute_coord_avg_disp(bboxes):
    """
    bboxes: list of length L, each either [x1,y1,x2,y2] or None
    返回：4 元组 (avg_dx1, avg_dy1, avg_dx2, avg_dy2)
    如果有任何 None 则返回 None（可根据需要改成忽略 None）
    """
    if any(bb is None for bb in bboxes):
        return None

    # 转成 list of tuples
    coords = [(bb[0], bb[1], bb[2], bb[3]) for bb in bboxes]
    # 计算差分并 pad 首位 0
    deltas = [(0.0, 0.0, 0.0, 0.0)]
    for i in range(1, len(coords)):
        x1, y1, x2, y2 = coords[i]
        px1, py1, px2, py2 = coords[i-1]
        deltas.append((
            x1 - px1,
            y1 - py1,
            x2 - px2,
            y2 - py2
        ))
    # 分别对 4 个通道求平均绝对值
    sum_dx1 = sum(abs(d[0]) for d in deltas)
    sum_dy1 = sum(abs(d[1]) for d in deltas)
    sum_dx2 = sum(abs(d[2]) for d in deltas)
    sum_dy2 = sum(abs(d[3]) for d in deltas)
    L = len(deltas)
    return (
        sum_dx1 / L,
        sum_dy1 / L,
        sum_dx2 / L,
        sum_dy2 / L
    )

def main(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # print(data)
    # label -> list of (dx1,dy1,dx2,dy2) per-track averages
    accum = defaultdict(list)

    for key, item in data.items():
        # print(key)
        for track_name, track_data in item.items():
            # print()
            bboxes = track_data.get('boxes')
            labels = track_data.get('labels')
            # print(track_data)
            if not bboxes or not labels or len(bboxes) < 2:
                continue

            coord_disp = compute_coord_avg_disp(bboxes)
            # print(coord_disp)
            if coord_disp is None:
                continue

            track_label = labels[-1]
            accum[track_label].append(coord_disp)

    # 打印每个 label 的平均值
    print(f"{'Label':>8}  {'Count':>5}  {'Avg_dx1':>8}  {'Avg_dy1':>8}  {'Avg_dx2':>8}  {'Avg_dy2':>8}")
    print("-" * 60)
    for label, vals in sorted(accum.items()):
        count = len(vals)
        # 按坐标通道累加再求平均
        sum_dx1 = sum(v[0] for v in vals)
        sum_dy1 = sum(v[1] for v in vals)
        sum_dx2 = sum(v[2] for v in vals)
        sum_dy2 = sum(v[3] for v in vals)
        print(f"{label:>8}  {count:5d}  "
              f"{(sum_dx1/count):8.3f}  "
              f"{(sum_dy1/count):8.3f}  "
              f"{(sum_dx2/count):8.3f}  "
              f"{(sum_dy2/count):8.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-track avg coord displacement and then avg-by-label"
    )
    parser.add_argument(
        "--json", "-j", required=True,
        help="Path to your data JSON file"
    )
    args = parser.parse_args()
    main(args.json)

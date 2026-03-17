#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Dict, List, Tuple


def load_txt_keys(txt_path: str, keep_basename: bool = True) -> List[str]:
    keys = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            print(s)
            keys.append(s)
    return keys


def load_json_dict(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{json_path} 不是 dict/json object，而是 {type(data)}")
    return data


def merge_two_json_dicts(a: Dict, b: Dict) -> Tuple[Dict, List[str]]:
    """
    合并两个 dict。若同 key 在 a 和 b 都存在且 value 不相等，记录冲突 key。
    冲突时优先使用 b 的值（可按需改成优先 a）。
    """
    merged = dict(a)
    conflicts = []
    for k, v in b.items():
        if k in merged and merged[k] != v:
            conflicts.append(k)
        merged[k] = v
    return merged, conflicts


def filter_json_by_keys(data: Dict, keys: List[str]) -> Tuple[Dict, List[str]]:
    out = {}
    missing = []
    for k in keys:
        if k in data:
            out[k] = data[k]
        else:
            missing.append(k)
    return out, missing


def save_json(data: Dict, out_path: str, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def main():
    parser = argparse.ArgumentParser(
        description="Merge train/test json then split by train/test txt keys."
    )
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--test_txt", required=True)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--test_json", required=True)

    parser.add_argument("--out_train_json", default="train.filtered.json")
    parser.add_argument("--out_test_json", default="test.filtered.json")

    parser.add_argument("--keep_basename", action="store_true")
    parser.add_argument("--no_keep_basename", dest="keep_basename", action="store_false")
    parser.set_defaults(keep_basename=True)

    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--save_missing", action="store_true")
    parser.add_argument("--save_conflicts", action="store_true",
                        help="If set, save conflicting keys to conflicts.txt")
    args = parser.parse_args()

    # load
    train_keys = load_txt_keys(args.train_txt, keep_basename=args.keep_basename)
    test_keys = load_txt_keys(args.test_txt, keep_basename=args.keep_basename)

    train_data = load_json_dict(args.train_json)
    test_data = load_json_dict(args.test_json)
    print(train_keys, test_keys)
    # merge
    all_data, conflicts = merge_two_json_dicts(train_data, test_data)

    # split
    train_filtered, train_missing = filter_json_by_keys(all_data, train_keys)
    test_filtered, test_missing = filter_json_by_keys(all_data, test_keys)

    save_json(train_filtered, args.out_train_json, indent=args.indent)
    save_json(test_filtered, args.out_test_json, indent=args.indent)

    # report
    print("=== Summary ===")
    print(f"train.txt keys: {len(train_keys)}")
    print(f"test.txt keys:  {len(test_keys)}")
    print(f"train.json keys: {len(train_data)}")
    print(f"test.json keys:  {len(test_data)}")
    print(f"merged keys:     {len(all_data)}")
    print(f"conflicts:       {len(conflicts)}")
    print("")
    print("=== Output ===")
    print(f"train kept: {len(train_filtered)}  missing: {len(train_missing)}  -> {args.out_train_json}")
    print(f"test  kept: {len(test_filtered)}  missing: {len(test_missing)}  -> {args.out_test_json}")

    if args.save_missing:
        train_miss_path = os.path.splitext(args.out_train_json)[0] + ".missing.txt"
        test_miss_path = os.path.splitext(args.out_test_json)[0] + ".missing.txt"
        with open(train_miss_path, "w", encoding="utf-8") as f:
            f.write("\n".join(train_missing))
        with open(test_miss_path, "w", encoding="utf-8") as f:
            f.write("\n".join(test_missing))
        print(f"Saved missing lists to: {train_miss_path}, {test_miss_path}")

    if args.save_conflicts and conflicts:
        conflict_path = "conflicts.txt"
        with open(conflict_path, "w", encoding="utf-8") as f:
            f.write("\n".join(conflicts))
        print(f"Saved conflicts to: {conflict_path}")


if __name__ == "__main__":
    main()
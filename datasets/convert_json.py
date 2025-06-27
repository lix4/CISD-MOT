#!/usr/bin/env python3
# save as convert_labels.py
import json
import sys
from pathlib import Path

label_map = {1: 0, 2: 1, 3: 2}

def convert(obj):
    """
    递归遍历 dict / list，遇到 'labels' 键就改值。
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "labels":
                if isinstance(v, list):
                    obj[k] = [label_map.get(x, x) for x in v]
                else:                       # 单个整数
                    obj[k] = label_map.get(v, v)
            else:
                convert(v)
    elif isinstance(obj, list):
        for item in obj:
            convert(item)

def main(src, dst):
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    convert(data)

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 转换完成，结果已保存到 {dst}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python convert_labels.py <input.json> [output.json]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path
    main(input_path, output_path)

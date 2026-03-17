import json
import os
import sys

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_list_from_txt(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(line)
    return items

def main():
    if len(sys.argv) < 5:
        print("用法 python split_json_by_txt.py json1.json json2.json train.txt test.txt")
        sys.exit(1)

    json1_path = sys.argv[1]
    json2_path = sys.argv[2]
    train_txt = sys.argv[3]
    test_txt = sys.argv[4]

    print("加载 JSON 文件...")
    dict1 = load_json(json1_path)
    dict2 = load_json(json2_path)

    print(f"dict1 size: {len(dict1)}")
    print(f"dict2 size: {len(dict2)}")

    train_keys = load_list_from_txt(train_txt)
    test_keys = load_list_from_txt(test_txt)

    new_train = {}
    new_test = {}

    def fetch_entry(key):
        """从两个 dict 里查找 key"""
        if key in dict1:
            return dict1[key]
        if key in dict2:
            return dict2[key]
        print(f"警告 未在两个 JSON 中找到键: {key}")
        return None

    print("\n构建 new_train.json ...")
    for key in train_keys:
        entry = fetch_entry(key)
        if entry is not None:
            new_train[key] = entry

    print("构建 new_test.json ...")
    for key in test_keys:
        entry = fetch_entry(key)
        if entry is not None:
            new_test[key] = entry

    # 保存
    out_train = "new_train.json"
    out_test = "new_test.json"

    with open(out_train, "w") as f:
        json.dump(new_train, f, indent=2)

    with open(out_test, "w") as f:
        json.dump(new_test, f, indent=2)

    print(f"\n完成！写入 {out_train} ({len(new_train)} entries)")
    print(f"写入 {out_test} ({len(new_test)} entries)")

if __name__ == "__main__":
    main()

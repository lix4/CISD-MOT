import os
import sys
from collections import Counter, defaultdict

def read_img_file_list(list_txt_path):
    """
    读取包含图片路径的 txt
    每一行是一个 jpg 图片路径
    """
    img_files = []
    with open(list_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_files.append(line)
    return img_files

def get_task_name_from_path(path):
    """
    从路径中提取类似 task_video_test_b_0 这样的目录名
    """
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        if p.startswith("task_video_"):
            return p
    return "UNKNOWN"

def count_labels_from_imgs(img_files):
    """
    读取图片列表 通过字符串替换得到 anno txt 路径
    然后统计标签分布 并按 task_video_xxx 分组

    标注文件每一行格式:
      <label> x1 y1 x2 y2 ...
    只使用第一个数字作为标签
    """
    global_counter = Counter()
    task_counters = defaultdict(Counter)

    for img_path in img_files:
        # 根据你的需求构造 anno 路径
        anno_path = img_path.replace('rgb-images', 'labels/av').replace('.jpg', '_g.txt')

        if not os.path.isfile(anno_path):
            print(f"警告 找不到标注文件: {anno_path}", file=sys.stderr)
            continue

        task_name = get_task_name_from_path(img_path)

        with open(anno_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    label = float(parts[0])
                except (ValueError, IndexError):
                    print(f"警告 行格式错误 跳过: {anno_path} -> {line}", file=sys.stderr)
                    continue

                global_counter[label] += 1
                task_counters[task_name][label] += 1

    return global_counter, task_counters

def main():
    if len(sys.argv) < 2:
        print("用法 python count_label_dist_by_task.py img_file_list.txt")
        sys.exit(1)

    list_txt_path = sys.argv[1]

    if not os.path.isfile(list_txt_path):
        print(f"找不到列表文件 {list_txt_path}")
        sys.exit(1)

    img_files = read_img_file_list(list_txt_path)
    print(f"共读取到 {len(img_files)} 个图片路径")

    global_counter, task_counters = count_labels_from_imgs(img_files)

    # 整体统计
    total = sum(global_counter.values())
    print("\n整体标签计数分布:")
    for label in sorted(global_counter.keys()):
        count = global_counter[label]
        ratio = count / total if total > 0 else 0
        print(f"  标签 {label}: {count} 个 占比 {ratio:.4f}")

    # 按 task_video_xxx 分组统计
    print("\n按 task_video_xxx 分组的标签情况:")
    for task_name in sorted(task_counters.keys()):
        counter = task_counters[task_name]
        labels_sorted = sorted(counter.keys())
        print(f"\n任务 {task_name}:")
        print(f"  出现的类别标签: {labels_sorted}")
        for label in labels_sorted:
            print(f"    标签 {label}: {counter[label]} 个")

if __name__ == "__main__":
    main()

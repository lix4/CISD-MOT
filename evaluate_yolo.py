import json
import cv2
import numpy as np
import os

json_path = './datasets/test_tracks_2_corrected_padded.json'
json_path_1 = './datasets/test_tracks_yolov11s_2_corrected_padded.json'
json_path_2 = './datasets/test_tracks_yolov11s_2_corrected_padded.json'
output_video_path = 'expanded_video_test_2.avi'       # 输出 AVI 文件
fps = 25

# 1. 读入 JSON
with open(json_path, 'r') as f:
    data = json.load(f)

with open(json_path_1, 'r') as f:
    data_1 = json.load(f)

with open(json_path_2, 'r') as f:
    data_2 = json.load(f)

# 2. 建立 frame_id -> path 的映射（并提取 int 帧号）
id_to_path = {}
for path in data.keys():
    base = os.path.basename(path)                   # e.g. "00012.jpg" 或 "14414.jpg"
    num_str, _ = os.path.splitext(base)             # e.g. "00012" 或 "14414"
    frame_id = int(num_str)                         # 转成 int：12 或 14414
    id_to_path[frame_id] = path
# print(id_to_path)
# 3. 获取所有帧号并升序排序
sorted_ids = sorted(id_to_path.keys())
# min_id, max_id = sorted_ids[0], sorted_ids[-1]

# 4. 初始化视频写入器（取最小帧号路径确定尺寸）
sample_path = id_to_path[sorted_ids[0]]
sample_img = cv2.imread(sample_path)
if sample_img is None:
    raise FileNotFoundError(f"无法读取示例图像：{sample_path}")
h, w = sample_img.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w*3, h))

# 5. 针对每个原始帧 frame_id，采样并绘制
for frame_id in sorted_ids:
    # 计算向前每隔4帧的 16 个帧号
    sample_ids = [frame_id - 4*i for i in range(16)]
    # 如果不足 16 帧或有越界／缺帧就跳过
    # if len(sample_ids) < 16:
    #     continue
    # 取其中实际存在的，并升序
    sample_ids = sorted(sample_ids)
    # if len(sample_ids) != 16:
    assert len(sample_ids) == 16
    #     continue
    # print(sample_ids)
    # 取 f0（当前 frame_id）对应的 track 数据
    track_datas = data[id_to_path[frame_id]]    # 这是个 dict: {tid_str: {'boxes':[...]}}
    track_datas_1 = data_1[id_to_path[frame_id]]    # 这是个 dict: {tid_str: {'boxes':[...]}}
    track_datas_2 = data_2[id_to_path[frame_id]]    # 这是个 dict: {tid_str: {'boxes':[...]}}

    # 在这 16 帧上，依次画出 boxes[k]
    for k, sid in enumerate(sample_ids):
        # print(sid)
        img_path = id_to_path[frame_id][:-9] + str(sid).zfill(5) +  '.jpg'
        img = cv2.imread(img_path)
        
        if img is None:
            raise FileNotFoundError(f"无法读取图像：{img_path}")
        img_1 = img.copy()
        img_2 = img.copy()
        
        # 对 f0 下的每个 track，画出第 k 个 box
        for tid_str, track_data in track_datas.items():
            boxes = track_data.get('boxes') or []
            if k < len(boxes):
                box = boxes[k]
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f'ID {tid_str}', (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 对 f0 下的每个 track，画出第 k 个 box
        for tid_str_1, track_data_1 in track_datas_1.items():
            boxes = track_data_1.get('boxes') or []
            if k < len(boxes):
                box = boxes[k]
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img_1, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img_1, f'ID {tid_str_1}', (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 对 f0 下的每个 track，画出第 k 个 box
        for tid_str_2, track_data_2 in track_datas_2.items():
            boxes = track_data_2.get('boxes') or []
            if k < len(boxes):
                box = boxes[k]
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img_2, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img_2, f'ID {tid_str_2}', (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 可选：标注采样序号
        cv2.putText(img, f'Step {k+1}/16', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        out.write(np.hstack([img, img_2, img_1]))

# 6. 完成并释放
out.release()
print("✅ Expanded AVI 视频生成完成：", output_video_path)

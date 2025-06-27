import os
# import cv2
import numpy as np
from deep_sort_pytorch.deep_sort.deep_sort import DeepSort

# 初始化 DeepSORT
deepsort = DeepSort("./deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")

# 假设你有一系列图片和检测框结果
# detections_per_frame = List of shape [ [ [x1,y1,x2,y2,conf], ... ], ... ]
detections_per_frame = load_detections_somehow()  # replace with your logic
image_dir = "frames/"
frame_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])

# 用于保存所有 track
all_tracks = []

for frame_idx, (frame_path, dets) in enumerate(zip(frame_paths, detections_per_frame)):
    image = cv2.imread(frame_path)

    if len(dets) == 0:
        dets = np.empty((0, 5))

    bbox_xywh = []
    confs = []

    # convert to [x_center, y_center, w, h]
    for *bbox, conf in dets:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        xc = x1 + w / 2
        yc = y1 + h / 2
        bbox_xywh.append([xc, yc, w, h])
        confs.append(conf)

    bbox_xywh = np.array(bbox_xywh)
    confs = np.array(confs)

    outputs = deepsort.update(bbox_xywh, confs, image)

    # outputs: [x1, y1, x2, y2, track_id]
    for output in outputs:
        x1, y1, x2, y2, track_id = output
        all_tracks.append([frame_idx, x1, y1, x2, y2, track_id])

        # optional draw
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'ID:{track_id}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv2.imwrite(f"results/{frame_idx:05d}.jpg", image)

# 保存 track 数据
np.savetxt("all_tracks.txt", np.array(all_tracks), fmt="%.2f")
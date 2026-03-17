import json
import cv2
import numpy as np
import os

# json_path = './datasets/valid_tracks_yolov11s_af_03_05_corrected_padded.json'
json_path = './datasets/test_tracks_2_corrected_padded.json'
output_video_path = 'video_lf_test_2_gt.avi'       # 输出 AVI 文件
fps = 25

# 1. 读入 JSON
with open(json_path, 'r') as f:
    data = json.load(f)


# 3. 获取所有帧号并升序排序
sorted_ids = sorted(data.keys())


# min_id, max_id = sorted_ids[0], sorted_ids[-1]

# 4. 初始化视频写入器（取最小帧号路径确定尺寸）
sample_path = sorted_ids[0]
sample_img = cv2.imread(sample_path)
if sample_img is None:
    raise FileNotFoundError(f"无法读取示例图像：{sample_path}")
h, w = sample_img.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w*2, h))

# 5. 针对每个原始帧 frame_id，采样并绘制
for sorted_id in sorted_ids:

    track_datas = data[sorted_id]    # 这是个 dict: {tid_str: {'boxes':[...]}}

    # print(sid)
    img_path = sorted_id
    gt_label_path = img_path.replace("rgb-images", "labels/av").replace(".jpg", "_g.txt")
    img = cv2.imread(img_path)
    
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{img_path}")
    img_1 = img.copy()
    
    all_boxes_per_frame = []
    # 对 f0 下的每个 track，画出第 k 个 box
    for tid_str, track_data in track_datas.items():
        boxes = track_data.get('boxes') or []
        box = boxes[-1]
        if box is not None:
            all_boxes_per_frame.append(box)
        if isinstance(box, (list, tuple)) and len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f'ID {tid_str}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # print(all_boxes_per_frame)
    all_boxes_per_frame = np.array(all_boxes_per_frame)
                
    gt_boxes = np.loadtxt(gt_label_path)
    if len(gt_boxes) == 0:
        continue
    if gt_boxes.ndim == 1:
        gt_boxes = np.expand_dims(gt_boxes, 0)

    gt_lbls = []
    for gt_box in gt_boxes:
        lbl, x1, y1, x2, y2 = map(int, gt_box)
        cv2.rectangle(img_1, (x1, y1), (x2, y2), (0, 0, 255), 2)
        gt_lbls.append(lbl)
        # cv2.putText(img_1, f'ID {tid_str}', (x1, y1 - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # print(gt_label_path)
    # print(gt_boxes.shape, all_boxes_per_frame.shape)
    if gt_boxes.shape[0] != all_boxes_per_frame.shape[0]:
        # 打印出出错的图像路径
        print(f"[GT/PRED COUNT MISMATCH] {img_path}", gt_lbls)
        # 把左右两幅图拼起来（左：预测，右：GT）
        error_img = np.hstack([img, img_1])
        # 用原图文件名作为保存名
        fn = os.path.basename(img_path)
        tmp = img_path.split('/')
        # print(tmp)
        save_path = os.path.join('evaluate_gt_lf', tmp[9] + '-' + tmp[10]  + '-' + fn)
        # 写出到 evaluate_gt_lf 文件夹
        cv2.imwrite(save_path, error_img)
        # 跳过本帧，不写入到视频
        continue

    # 可选：标注采样序号
    # cv2.putText(img,  (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    out.write(np.hstack([img, img_1]))

# 6. 完成并释放
out.release()
print("✅ Expanded AVI 视频生成完成：", output_video_path)

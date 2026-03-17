import json
import cv2
import numpy as np
import os
from models.MOT_IVD_v1_9 import MOT_IVD_v1_9
import torch
from datasets.dataset_mot import listDataset, avivd_collate_fn
import torchvision.transforms as T
from os.path import join

# ---------- Config ----------
# json_path = './datasets/valid_tracks_yolov11s_af_03_05_corrected_padded.json'
json_path = './datasets/test_tracks_yolov11s_2_corrected_padded.json'
ckpt_path = "./runs/MOT_classification_cl_v1_9_yolo11s_af/best_model_epoch_12_mAP_0.9099.pth"
# txt_list = './meta-files/validlist_e2e_new_1.txt'
txt_list = './meta-files/testlist_e2e_new_2.txt'
base_path = '/uu/sci.utah.edu/projects/smartair/Dataset'
output_video_path = 'video_lf_cls_test_2.avi'
fps = 25
classes = ["Moving", "Idling", "EngineOff"]  # <-- verify your order

# ---------- Utils ----------
def put_text(img, text, org, font_scale=0.5, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, cv2.LINE_AA)

def draw_box(img, box, color=(0, 180, 255), thick=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

# ---------- 1) Read JSON just to get a sample frame size (optional) ----------
with open(json_path, 'r') as f:
    data_json = json.load(f)
sorted_ids = sorted(data_json.keys())
sample_path = sorted_ids[0]
sample_img = cv2.imread(sample_path)
if sample_img is None:
    raise FileNotFoundError(f"无法读取示例图像：{sample_path}")
H, W = sample_img.shape[:2]

# ---------- 2) Load model ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MOT_IVD_v1_9(num_classes=3).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt if "model_state_dict" not in ckpt else ckpt["model_state_dict"]
model.load_state_dict(state_dict, strict=True)
model.eval()

# ---------- 3) Dataloader ----------
kwargs = {'num_workers': 8, 'pin_memory': True, 'prefetch_factor': 2}
valid_dataset = listDataset(
    base_path=base_path,
    txt_list=txt_list,
    json_path=json_path,
    load_frames=False,              # images will be read on the fly via frame paths
    transform=T.ToTensor()
)
dataloader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=avivd_collate_fn,
    **kwargs
)

# ---------- 4) Video writer (single annotated frame per sample) ----------
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (W*2, H))

# ---------- 5) Inference + draw ----------
with torch.no_grad():
    for batch in dataloader:
        # Expecting: clips_unused, bboxes, mask, audio, labels, video_ids, _, yolo_confs
        _, bboxes, mask, audio, labels, video_ids, _, yolo_confs = batch

        # Read the corresponding frame image
        # If video_ids is a list of file paths for the current sample, pick the last frame path,
        # or simply the unique frame path used with your -1 index.
        # Commonly, for tracklets, you may have per-vehicle, per-frame paths; adapt as needed:
        if isinstance(video_ids, (list, tuple)) and len(video_ids) > 0:
            # Many collate_fns give a list[str], one per sample; if it's nested, adjust indexing:
            frame_path = video_ids[0] if isinstance(video_ids[0], str) else video_ids[0][-1]
        else:
            # fallback – use sorted_ids (not ideal, but prevents crash)
            frame_path = sorted_ids[0]

        tmp = frame_path.split("-")
        frame_path = join('/uu/sci.utah.edu/projects/smartair/Dataset/Video/LDS', tmp[0][:-2], tmp[0], tmp[1], 'rgb-images', tmp[2])
        # print(frame_path)
        img = cv2.imread(frame_path)
        if img is None:
            # If the path doesn’t directly point to an image, you may need to reconstruct
            # from base_path + relative path; adapt as your dataset stores it.
            print(f"[WARN] 读取失败: {frame_path}")
            img = np.zeros((H, W, 3), dtype=np.uint8)
        
        img_gt = img.copy()

        # ------ Draw GT (safe even if no preds) ------
        try:
            gt_path = frame_path.replace('rgb-images', 'labels/av').replace('.jpg', '_g.txt')
            gt_boxes = np.loadtxt(gt_path)
            # 统一成 (K,5) 或 (0,5)
            if isinstance(gt_boxes, float) or (isinstance(gt_boxes, np.ndarray) and gt_boxes.ndim == 0):
                gt_boxes = np.empty((0, 5))
            elif isinstance(gt_boxes, np.ndarray) and gt_boxes.ndim == 1:
                if gt_boxes.size == 0:
                    gt_boxes = np.empty((0, 5))
                else:
                    gt_boxes = gt_boxes[None, :]

            for i in range(gt_boxes.shape[0]):
                cls_id = int(gt_boxes[i, 0]) - 1
                box = gt_boxes[i, 1:]
                color = [(0, 200, 0), (0, 165, 255), (255, 0, 0)][cls_id % 3]
                draw_box(img_gt, box, color=color, thick=2)
                label = f"{classes[cls_id]}"
                x1, y1 = int(box[0]), int(box[1]) - 6
                put_text(img_gt, label, (max(0, x1), max(0, y1)))
        except Exception as e:
            # 没有GT文件也不要中断
            pass

        # ------ Predictions (only if there are YOLO boxes) ------
        N = bboxes.shape[0] if isinstance(bboxes, torch.Tensor) else len(bboxes)

        if N > 0:
            # Move tensors you actually need
            bboxes_dev = bboxes.to(device)
            audio_dev  = audio.to(device)

            # Forward: 你的分类头是对最后一帧 [N, C]
            logits, joint_emb_n = model(bboxes_dev, audio_dev)
            probs = torch.softmax(logits, dim=-1)
            cls_scores, cls_ids = probs.max(dim=-1)

            # yolo_confs 对齐
            if isinstance(yolo_confs, torch.Tensor):
                yolo_confs = yolo_confs.squeeze(-1).detach().cpu()
            else:
                yolo_confs = torch.tensor(yolo_confs).view(-1)

            # 兜底：长度不一致时截断/补1
            if yolo_confs.numel() < N:
                pad = torch.ones(N - yolo_confs.numel())
                yolo_confs = torch.cat([yolo_confs, pad], dim=0)
            elif yolo_confs.numel() > N:
                yolo_confs = yolo_confs[:N]

            final_scores = (yolo_confs * cls_scores.cpu()).numpy()
            cls_ids_np   = cls_ids.cpu().numpy()

            # 取最后一帧的框
            if bboxes_dev.ndim == 3:
                boxes_last = bboxes_dev[:, -1, :].detach().cpu().numpy()
            else:
                boxes_last = bboxes_dev.detach().cpu().numpy()

            for i in range(boxes_last.shape[0]):
                box = boxes_last[i]
                cls_id = int(cls_ids_np[i])
                score = float(final_scores[i]) if i < len(final_scores) else 0.0
                color = [(0, 200, 0), (0, 165, 255), (255, 0, 0)][cls_id % 3]
                draw_box(img, box, color=color, thick=2)
                x1, y1 = int(box[0]), int(box[1]) - 6
                put_text(img, f"{classes[cls_id]} {score:.2f}", (max(0, x1), max(0, y1)))
        else:
            # 没有YOLO检测：仍写帧，并做个小提示
            put_text(img, "No detection", (10, 25), font_scale=0.8, thickness=2)

        
        # ------ Write one frame per sample ------
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))
        if img_gt.shape[1] != W or img_gt.shape[0] != H:
            img_gt = cv2.resize(img_gt, (W, H))

        out.write(np.hstack([img, img_gt]))

# ---------- 6) Release ----------
out.release()
print("✅ AVI 生成完成：", output_video_path)

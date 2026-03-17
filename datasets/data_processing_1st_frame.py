import json
import os
from scipy.optimize import linear_sum_assignment
import torch
from types import SimpleNamespace
from pathlib import Path
from typing import Union, Optional
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.ops import non_max_suppression
import numpy as np
import cv2
import torchvision.transforms as T

def compute_iou(boxA, boxB):
    """Compute IoU of two boxes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

def load_model(nc:int=1, ckpt:Optional[Union[str, Path]] = None, device="cuda"):
    model = DetectionModel(cfg='yolo11s.yaml', ch=3, nc=nc).to(device)
    if ckpt:
        ckpt = Path(ckpt).expanduser()
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"✅  已加载权重: {ckpt}")
    # 伪参数供 v8DetectionLoss 使用
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5,
        fl_gamma=0.0, label_smoothing=0.0,
        nbs=64, classes=None
    )
    return model

def pad_first_frame_with_iou(json_path, frame_offset=15, frame_step=4, iou_thresh=0.3):
    """
    For each JSON key (frame), compute the first frame index = current_index - frame_offset * frame_step,
    load GT boxes from its txt, then IoU match each track's second frame box to a GT box,
    and fill the first slot in track['boxes'] if IoU >= threshold.
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    model = load_model(nc=1, ckpt='../runs/MOT_detection_yolo11s_MEB_ft/best_model.pt').cuda()
    model.eval()

    # Process each entry
    for key, entry in data.items():
        # Parse current frame number from filename
        base = os.path.basename(key)
        name, _ = os.path.splitext(base)
        try:
            current_idx = int(name)
        except ValueError:
            continue  # skip if filename not numeric

        # Compute first frame index
        first_idx = current_idx - frame_offset * frame_step
        # Construct GT txt path (same dir, same prefix)
        dirpath = os.path.dirname(key)
        gt_txt = os.path.join(dirpath, f"{first_idx:05d}.jpg")
        # --- (1) 读取原 BGR、准备网络输入 ---
        orig_bgr = cv2.imread(gt_txt)                      # H×W×3, BGR
        inp_rgb  = cv2.resize(orig_bgr[..., ::-1], (224,224))
        inp      = T.ToTensor()(inp_rgb).unsqueeze(0).cuda()

        with torch.no_grad():
            raw  = model(inp)
        preds = non_max_suppression(raw, 0.7, 0.7)[0]
        if preds is None or len(preds) == 0:
            continue

        scale   = np.array([320/224, 240/224, 320/224, 240/224])
        # boxes_xywh = preds[:, :4].cpu().numpy()
        boxes_xyxy   = (preds[:, :4].cpu().numpy() * scale)

        print(boxes_xyxy)
        # # Read GT boxes from txt
        # gt_boxes = []
        # with open(gt_txt, 'r') as gf:
        #     for line in gf:
        #         parts = line.strip().split()
        #         if len(parts) >= 4:
        #             x1, y1, x2, y2 = map(float, parts[1:5])
        #             gt_boxes.append([x1, y1, x2, y2])

        # print(gt_boxes)
        # For each track in this entry
        for track_id, track_data in entry.items():
            # Get the second frame box (index 1)
            boxes = track_data.get('boxes', [])
            if len(boxes) < 2 or boxes[1] is None:
                continue
            box1 = boxes[1]

            # Find best GT match
            best_iou = 0.0
            best_box = None
            for gb in boxes_xyxy:
                # print(gb, box1)
                iou = compute_iou(gb, box1)
                if iou > best_iou:
                    best_iou = iou
                    best_box = gb

            # If above threshold, fill first slot
            if best_box is not None and best_iou >= iou_thresh:
                # print("here")
                track_data['boxes'][0] = best_box.tolist()

    # Save modified JSON
    out_path = json_path.replace('.json', '_padded.json')
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Padded JSON saved to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pad first frame via IoU match')
    parser.add_argument('--json', type=str, required=True, help='Path to JSON file')
    parser.add_argument('--offset', type=int, default=15, help='Frame offset count')
    parser.add_argument('--step', type=int, default=4, help='Frame step interval')
    parser.add_argument('--iou_thresh', type=float, default=0.3, help='IoU threshold')
    args = parser.parse_args()
    pad_first_frame_with_iou(args.json, frame_offset=args.offset, frame_step=args.step, iou_thresh=args.iou_thresh)

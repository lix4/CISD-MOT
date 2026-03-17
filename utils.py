import torch
import torch.nn as nn
from ultralytics.nn.modules.head import DFL
from ultralytics.utils.metrics import ConfusionMatrix, box_iou, ap_per_class
from ultralytics.utils.ops import xywh2xyxy
import numpy as np
from ultralytics.utils.ops import non_max_suppression
from ultralytics.nn.modules.head import Detect
import yaml
from ultralytics.nn.tasks import DetectionModel
import cv2
from torchmetrics.detection.mean_ap import MeanAveragePrecision
# from utils.eval import compute_map_iou_range  # 你自己的mAP计算
from typing import List, Any, Dict
import os
import random
import matplotlib.pyplot as plt
from torchmetrics.classification import PrecisionRecallCurve
###### Plot PR Curve ######

def collect_detection_scores(preds, targets, iou_thresh=0.5):
    """
    preds: list of dict, 每张图的预测
      dict keys: "boxes" (Tensor[N,4]), "scores" (Tensor[N]), "labels" (Tensor[N])
    targets: list of dict, 每张图的 GT
      dict keys: "boxes" (Tensor[M,4]), "labels" (Tensor[M])
    """
    all_scores = {}  # cls -> [scores]
    all_labels = {}  # cls -> [0/1]

    for pred, tgt in zip(preds, targets):
        if pred["boxes"].numel() == 0:
            continue
        # 计算 IoU 矩阵
        ious = box_iou(pred["boxes"], tgt["boxes"])  # [N_pred, N_gt]
        # 每个 GT 只匹配一次
        gt_matched = torch.zeros(tgt["boxes"].size(0), dtype=torch.bool)

        for idx in torch.argsort(pred["scores"], descending=True):
            cls = int(pred["labels"][idx])
            conf = float(pred["scores"][idx])
            # 找最优 GT
            max_iou, max_j = torch.max(ious[idx], dim=0)
            is_tp = 0
            if max_iou >= iou_thresh and (not gt_matched[max_j]) and cls == int(tgt["labels"][max_j]):
                is_tp = 1
                gt_matched[max_j] = True

            all_scores.setdefault(cls, []).append(conf)
            all_labels.setdefault(cls, []).append(is_tp)

    return all_scores, all_labels

def plot_detection_pr_curve(all_scores, all_labels):
    """
    all_scores: dict cls -> [scores]
    all_labels: dict cls -> [0/1]
    """
    plt.figure(figsize=(8,6))
    for cls in sorted(all_scores.keys()):
        y_score = torch.tensor(all_scores[cls])
        y_true  = torch.tensor(all_labels[cls])

        pr_curve = PrecisionRecallCurve(pos_label=1)
        precision, recall, _ = pr_curve(y_score, y_true)

        plt.plot(recall.numpy(), precision.numpy(), label=f"Class {cls}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Detection PR Curves per Class")
    plt.legend()
    plt.grid(True)
    plt.show()

###### Plot PR Curve ######

def jitter_bbox(box, delta=0.1, img_wh=(320, 240)):
    """
    box  : [x1, y1, x2, y2]（像素坐标）
    delta: 最大抖动幅度（像素）；可改成 0.1*W/H 做相对抖动
    img_wh: (W, H) 图像尺寸，用于裁剪
    """
    if box is None:
        return None

    x1, y1, x2, y2 = map(float, box)
    # 每个坐标独立 ±delta
    x1 += random.gauss(0, 2)
    y1 += random.gauss(0, 2)
    x2 += random.gauss(0, 2)
    y2 += random.gauss(0, 2)
    # print(x1,y1,x2,y2, random.uniform(-delta, delta))
    # 裁剪到图像范围并保证 x1<x2, y1<y2
    W, H = img_wh
    x1 = max(0, min(x1, x2-1))
    y1 = max(0, min(y1, y2-1))
    x2 = min(W-1, max(x2, x1+1))
    y2 = min(H-1, max(y2, y1+1))

    # iou = compute_iou([x1, y1, x2, y2], box)

    return [x1, y1, x2, y2]

def build_predictions(video_ids: List[Any],
                      bboxes: torch.Tensor,
                      scores: torch.Tensor,
                      labels: torch.Tensor,
                      yolo_confs: torch.Tensor,
                      cls_scores: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
    """
    Group raw prediction arrays into a list of dicts per image for torchmetrics.

    Args:
        video_ids: List of identifiers for each bbox (len N)
        bboxes: Tensor[N, 4] of predicted boxes
        labels: Tensor[N] of predicted labels
    Returns:
        List of dicts [{'boxes', 'scores', 'labels'}] sorted by video_id.
    """
    preds_per_vid: Dict[Any, Dict[str, List]] = {}
    # print(len(video_ids), bboxes[:, -1, :].shape, labels.shape)
    # print(len(video_ids), bboxes[:, -1, :].shape[0], labels.shape[0], scores.shape[0], yolo_confs.shape[0], cls_scores.shape[0])
    assert len(video_ids) == bboxes[:, -1, :].shape[0] == labels.shape[0] == scores.shape[0] == yolo_confs.shape[0] == cls_scores.shape[0]
    for vid, box, lbl, score, yolo_conf, cls_score in zip(video_ids, bboxes[:, -1, :], labels, scores, yolo_confs, cls_scores):
        if vid not in preds_per_vid:
            preds_per_vid[vid] = {'boxes': [], 'scores': [], 'labels': [], 'yolo_confs': [], "cls_scores": []}
        if int(lbl) < 0:
            continue
        if lbl == -1:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
        preds_per_vid[vid]['boxes'].append(box.tolist())
        # preds_per_vid[vid]['scores'].append(1.0)
        preds_per_vid[vid]['labels'].append(int(lbl))
        preds_per_vid[vid]['scores'].append(score)
        preds_per_vid[vid]['yolo_confs'].append(yolo_conf)
        preds_per_vid[vid]['cls_scores'].append(cls_score)

    predictions: List[Dict[str, torch.Tensor]] = []
    for vid in sorted(preds_per_vid.keys()):
        data = preds_per_vid[vid]
        predictions.append({
            'image_id': vid,
            'boxes':  torch.tensor(data['boxes'], dtype=torch.float32),
            'scores': torch.tensor(data['scores'], dtype=torch.float32),
            'labels': torch.tensor(data['labels'], dtype=torch.int64),
            'yolo_confs': torch.tensor(data['yolo_confs'], dtype=torch.float32),
            'cls_scores': torch.tensor(data['cls_scores'], dtype=torch.float32)
        })
    return predictions

def load_ground_truths(video_ids: List[str], data_location = 'LDS') -> List[Dict[str, torch.Tensor]]:
    """
    Load ground truth annotations for each image (video_id) in video_ids.
    Constructs path for each video_id:
      /uu/sci.utah.edu/projects/smartair/Dataset/Video/<c0>/<c1>/<c2>/<video_id>.txt
    Returns a list of dicts [{'boxes', 'labels'}] in the same order as sorted unique video_ids.
    """
    gts: List[Dict[str, torch.Tensor]] = []
    for vid in sorted(set(video_ids)):
        comps = vid.split('-')
        gt_path = os.path.join(
            '/uufs/sci.utah.edu/projects/smartair/Dataset/Video/' + data_location,
            comps[0][:-2], comps[0], comps[1], 'labels/av', comps[2][:-4] + "_g.txt"
        )
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"GT file not found: {gt_path}")
        boxes, labels = [], []
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0])) - 1
                x1, y1, x2, y2 = map(float, parts[1:5])
                if data_location == 'MEB':
                    # print(data_location)
                    x1=x1/2
                    y1=y1/2
                    x2=x2/2
                    y2=y2/2
                    if cls == 0:
                        cls = 1
                    elif cls == 1:
                        cls = 2
                    else:
                        cls = 0
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)
        if boxes:
            gts.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            })
        else:
            gts.append({
                'boxes': torch.zeros((0, 4),  dtype=torch.float32),
                'labels': torch.zeros((0,   ),  dtype=torch.int64)
            })
    return gts

def box_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    print(inter / (area1 + area2 - inter))
    return inter / (area1 + area2 - inter)

def compute_map_iou_range(stats, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    """
    Compute mAP@0.5:0.95 from detection stats.

    stats: List of (correct, conf, pred_cls, target_cls) for each image
    """
    if len(stats) == 0:
        return 0.0, 0.0

    correct, conf, pred_cls, target_cls = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    iouv = np.array(iou_thresholds)
    niou = len(iouv)

    ap = np.zeros((len(np.unique(target_cls)), niou))
    for ci, c in enumerate(np.unique(target_cls)):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 or n_gt == 0:
            continue

        # Sort predictions by descending confidence
        idx = np.argsort(-conf[i])
        matches = correct[i][idx]

        for j, iou_thresh in enumerate(iouv):
            # print(matches)
            tp_cumsum = np.cumsum(matches >= iou_thresh)
            fp_cumsum = np.cumsum(matches < iou_thresh)
            recall = tp_cumsum / (n_gt + 1e-16)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
            ap[ci, j] = compute_ap(recall, precision)

    map_50 = ap[:, 0].mean()
    map_5095 = ap.mean()
    return map_50, map_5095


def compute_ap(recall, precision):
    """Compute the average precision, given precision and recall."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

# ---------------------------------- ① 复刻 Detect 推理分支 ----------------------------------
def decode_yolo_raw(raw_list, model):
    """
    raw_list: list(len=nl) of Tensor(bs, na*(5+nc), ny, nx)
    return  : Tensor(bs, n_all, 6) with [x1 y1 x2 y2 conf cls]
    """
    detect = next(m for m in model.modules() if m.__class__.__name__ == 'Detect')
    stride, anchor_grid, grid = detect.stride, detect.anchor_grid, detect.grid
    nl, na, no = detect.nl, detect.na, detect.no      # no = 5+nc
    bs = raw_list[0].shape[0]
    preds = []

    for i in range(nl):
        x = raw_list[i].sigmoid()                     # 1) sigmoid
        bs, _, ny, nx = x.shape
        x = (x.view(bs, na, no, ny, nx)
               .permute(0,1,3,4,2).contiguous())      # bs,na,ny,nx,no

        if grid[i].shape[2:4] != x.shape[2:4]:        # 生成新网格
            grid[i] = ops._make_grid(nx, ny).to(x.device)

        # 2) decode centre-x,y
        xy = (x[..., 0:2] * 2. - 0.5 + grid[i]) * stride[i]
        # 3) decode w,h
        wh = (x[..., 2:4] * 2) ** 2 * anchor_grid[i]
        # 4) conf + cls
        conf_cls = x[..., 4:]

        y = torch.cat((xy, wh, conf_cls), -1).view(bs, -1, no)
        preds.append(y)

    return torch.cat(preds, 1)                        # (bs, n_all, 5+nc)
# --------------------------------------------------------------------------------------------


def evaluate_map(model, val_loader, device, iou_thres=0.5, conf_thres=0.3, save_dir=None, max_vis=20):
    model.eval()
    stats = []

    # patch DFL（模型创建后要设置 dfl 否则报错）
    if hasattr(model.model[-1], 'dfl') and isinstance(model.model[-1].dfl, nn.Module):
        model.model[-1].dfl = DFL(16).to(device)

    map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=None, class_metrics=True, average='macro')
    
    # 如果需要可视化，先建目录
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)


    # 用于存所有候选样本的信息
    viz_samples = []

    with torch.no_grad():
        for batch_idx, (pic_paths, images, targets) in enumerate(val_loader):
            images = images.to(device)
            # forward: 得到 raw outputs
            # print(images.shape)
            raw_outputs = model.predict(images)
            # decode + DFL + NMS（YOLOv8的处理逻辑）
            preds = non_max_suppression(raw_outputs, conf_thres=conf_thres, iou_thres=iou_thres)
            # print(preds)
            for i, pred in enumerate(preds):
                if pred is None or len(pred) == 0:
                    continue
                # print("A", pred)

                pred = pred.detach().cpu()
                pred_boxes = pred[:, :4]
                pred_scores = pred[:, 4]
                # print('pred_scores', pred_scores)
                pred_cls = pred[:, 5]

                # print("pred", pred_boxes, pred_scores, pred_cls)

                # GT
                gt = targets['bboxes'][targets['batch_idx'] == i]
                gt_cls = targets['cls'][targets['batch_idx'] == i]
                gt_boxes = xywh2xyxy(gt).to(device)
                for gt_box in gt_boxes:
                    gt_box[0]*=224
                    gt_box[1]*=224
                    gt_box[2]*=224
                    gt_box[3]*=224
                # print("B", gt_boxes)
                if len(gt_boxes) == 0:
                    # print("here")
                    continue
                # print(pred_boxes, gt_boxes)
                pred_dic = {'boxes': torch.tensor(pred_boxes), 'labels': torch.tensor(pred_cls).int(), 'scores': torch.tensor(pred_scores)}
                gt_dic = {'boxes': torch.tensor(gt_boxes), 'labels': torch.tensor(gt_cls).int()}
                map_metric.update([pred_dic], [gt_dic])

                # ---------- 可视化部分 ----------
                # 收集候选用于后面随机采样
                if save_dir is not None:
                    sample = {
                        "img_tensor": images[i].detach().cpu(),         # [C,H,W], 0~1
                        "pred_boxes": pred_boxes.clone(),               # xyxy 像素
                        "pred_scores": pred_scores.clone(),
                        "pred_cls": pred_cls.clone(),
                        "gt_boxes": gt_boxes.detach().cpu().clone(),    # xyxy 像素
                        "gt_cls": gt_cls.detach().cpu().clone(),
                        "base_name": os.path.basename(pic_paths[i]),
                        "batch_idx": batch_idx,
                        "img_idx": i,
                    }
                    viz_samples.append(sample)
                # ---------- 可视化部分结束 ----------

    # if len(stats):
    # correct, scores, pred_cls, gt_cls = [torch.cat(x, 0).numpy() for x in zip(*stats)]
    # print(len(stats))
    # map50, map5095 = compute_map_iou_range(stats)
    map_value = map_metric.compute()
    print(f"mAP: {map_value['map']:.4f}")
    print(f"mAP (IoU=0.5): {map_value['map_50']:.4f}")
    # print(f"mAP (IoU=0.50.95): {map_value['map']:.4f}")
    # for i, ap in enumerate(map_value['map_per_class']):
    #     print(f"Class {i} AP: {ap:.4f}")
    # print(map_value)
    map_metric.reset()
    mAP50 =map_value['map_50']
    mAP75 =map_value['map_75']
    mAP5095=map_value['map']
    
    # ---------------- 随机可视化部分 ----------------
    if save_dir is not None and len(viz_samples) > 0:
        os.makedirs(save_dir, exist_ok=True)
        k = min(max_vis, len(viz_samples))
        chosen = random.sample(viz_samples, k)

        for idx, sample in enumerate(chosen):
            img_tensor = sample["img_tensor"]      # [C,H,W], 0~1
            pred_boxes = sample["pred_boxes"]
            pred_scores = sample["pred_scores"]
            gt_boxes   = sample["gt_boxes"]
            base_name  = sample["base_name"]
            b_idx      = sample["batch_idx"]
            i_idx      = sample["img_idx"]

            # tensor -> numpy, RGB -> BGR
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 画预测框 红色
            for j, pb in enumerate(pred_boxes):
                x1, y1, x2, y2 = pb.int().tolist()
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 可选 写分数
                # score_text = f"{float(pred_scores[j]):.2f}"
                # cv2.putText(img_vis, score_text, (x1, y1 - 2),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # 画 GT 框 绿色
            for gb in gt_boxes:
                x1, y1, x2, y2 = gb.int().tolist()
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # resize 成 320x240 再保存
            img_vis = cv2.resize(img_vis, (320, 240), interpolation=cv2.INTER_LINEAR)

            out_name = f"val_{b_idx:03d}_{i_idx:02d}_{base_name}"
            out_path = os.path.join(save_dir, out_name)
            cv2.imwrite(out_path, img_vis)

    # ---------------- 随机可视化结束 ----------------

    return mAP50, mAP75, mAP5095
    # else:
    #     return 0.0, 0.0
    
def draw_bbox_with_label(img, box, label, color, gt=None):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    cv2.putText(img, f'Pred: {label}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if gt is not None:
        cv2.putText(img, f'GT: {gt}', (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    return img

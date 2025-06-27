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

# def load_model_from_yaml(yaml_path, nc=1, ch_list=[256, 512, 1024], in_channels=3):
#     with open(yaml_path, "r") as f:
#         cfg_dict = yaml.safe_load(f)

#     # ❌ 把 Detect 从 head 移除（只保留结构）
#     cfg_dict['head'] = cfg_dict['head'][:-1]

#     # ✅ 构建模型骨架（没有 Detect）
#     model = DetectionModel(cfg=cfg_dict, ch=in_channels)

#     # ✅ 手动添加 Detect 模块（绕过 parse_model 解析）
#     detect_layer = Detect(ch=[256, 512, 1024], nc=1)
#     model.model.append(detect_layer)  # 相当于 model.model[-1] = Detect(...)
#     print(detect_layer)
#     print("✅ nc =", detect_layer.nc)
#     print("✅ 输出通道 =", [m[-1].out_channels for m in detect_layer.cv2])
#     return model

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


def evaluate_map(model, val_loader, device, iou_thres=0.5, conf_thres=0.25):
    model.eval()
    stats = []

    # patch DFL（模型创建后要设置 dfl 否则报错）
    if hasattr(model.model[-1], 'dfl') and isinstance(model.model[-1].dfl, nn.Module):
        model.model[-1].dfl = DFL(16).to(device)

    map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=None, class_metrics=True, average='macro')
    
    with torch.no_grad():
        for _, images, targets in val_loader:
            images = images.to(device)
            # forward: 得到 raw outputs
            # print(images.shape)
            raw_outputs = model.predict(images)
            # print("###############")
            # print(raw_outputs)
            # decoded     = decode_yolo_raw(raw_outputs, model)
            # print(len(raw_outputs))
            # print(raw_outputs[0][0,:,:])
            # print(raw_outputs[0].shape, raw_outputs[1][0].shape, raw_outputs[1][1].shape, raw_outputs[1][2].shape)
            # decode + DFL + NMS（YOLOv8的处理逻辑）
            preds = non_max_suppression(raw_outputs, conf_thres=conf_thres, iou_thres=0.7)
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

                # correct = torch.zeros(len(pred_boxes), dtype=torch.bool, device=device)
                # ious = box_iou(pred_boxes, gt_boxes)
                # iou_max, iou_argmax = ious.max(1)
                # for j in range(len(pred_boxes)):
                #     if iou_max[j] > iou_thres and pred_cls[j] == gt_cls[iou_argmax[j]]:
                #         correct[j] = True

                # stats.append((correct.cpu(), pred_scores.cpu(), pred_cls.cpu(), gt_cls.cpu()))

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
    mAP5095=map_value['map']
    # print("here")
    return mAP50, mAP5095
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

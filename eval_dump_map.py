import argparse
import json

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute detection mAP from eval_dump.json"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="eval_dump.json",
        help="Path to dumped eval json"
    )
    parser.add_argument(
        "--fuse_mode",
        type=str,
        default="stored",
        choices=[
            "stored",
            "product",
            "weighted_sum",
            "geometric",
            "rank",
            "cls_only",
            "yolo_only",
            "weighted_time"
        ],
        help="How to fuse yolo_confs and cls_scores"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight used by weighted_sum/geometric fusion"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for MeanAveragePrecision"
    )
    return parser.parse_args()


def to_tensor(x, dtype):
    return torch.tensor(x, dtype=dtype)


def load_dump(path):
    with open(path, "r") as f:
        payload = json.load(f)

    predictions = []
    for pred in payload["predictions"]:
        item = {
            "boxes": to_tensor(pred["boxes"], torch.float32),
            "scores": to_tensor(pred["scores"], torch.float32),
            "labels": to_tensor(pred["labels"], torch.int64),
        }
        if "image_id" in pred:
            item["image_id"] = pred["image_id"]
        if "yolo_confs" in pred:
            item["yolo_confs"] = to_tensor(pred["yolo_confs"], torch.float32)
        if "cls_scores" in pred:
            item["cls_scores"] = to_tensor(pred["cls_scores"], torch.float32)
        predictions.append(item)

    groundtruth = []
    for gt in payload["groundtruth"]:
        item = {
            "boxes": to_tensor(gt["boxes"], torch.float32),
            "labels": to_tensor(gt["labels"], torch.int64),
        }
        if "image_id" in gt:
            item["image_id"] = gt["image_id"]
        groundtruth.append(item)

    return predictions, groundtruth


def normalize_minmax(x, eps=1e-12):
    if x.numel() == 0:
        return x
    return (x - x.min()) / (x.max() - x.min() + eps)


def fuse_scores(predictions, mode, alpha):
    if mode == "stored":
        return predictions

    boxes_all = []
    conf_all = []
    prob_all = []
    offsets = []
    start = 0

    for pred in predictions:
        n = pred["boxes"].shape[0]
        if n == 0:
            continue
        if "yolo_confs" not in pred or "cls_scores" not in pred:
            raise ValueError(
                f"Prediction dump is missing yolo_confs/cls_scores, cannot use fuse_mode={mode}"
            )
        boxes_all.append(pred["boxes"])
        conf_all.append(pred["yolo_confs"])
        prob_all.append(pred["cls_scores"])
        offsets.append((pred, start, start + n))
        start += n

    if not offsets:
        return predictions

    conf = torch.cat(conf_all)
    prob = torch.cat(prob_all)
    conf_norm = normalize_minmax(conf)

    if mode == "product":
        new_score = conf * prob
    elif mode == "weighted_sum":
        new_score = alpha * prob + (1.0 - alpha) * conf_norm
    elif mode == "geometric":
        new_score = (prob.clamp_min(1e-12) ** alpha) * (
            conf_norm.clamp_min(1e-12) ** (1.0 - alpha)
        )
    elif mode == "rank":
        n = prob.numel()
        rank_prob = prob.argsort().argsort().float()
        rank_conf = conf.argsort().argsort().float()
        new_score = rank_prob * (n + 1) + rank_conf
        new_score = normalize_minmax(new_score)
    elif mode == "cls_only":
        new_score = prob
    elif mode == "yolo_only":
        new_score = conf
    elif mode == "weighted_time":
        new_score = (prob ** alpha) * (conf ** alpha)
    else:
        raise ValueError(f"Unsupported fuse_mode: {mode}")

    fused_predictions = []
    for pred in predictions:
        copied = dict(pred)
        fused_predictions.append(copied)

    for pred, s, e in offsets:
        pred["scores"] = new_score[s:e]

    return predictions


def main():
    args = parse_args()
    predictions, groundtruth = load_dump(args.input)
    predictions = fuse_scores(predictions, args.fuse_mode, args.alpha)

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[args.iou],
        class_metrics=True,
        average="macro",
    )
    metric.update(predictions, groundtruth)
    results = metric.compute()

    print(f"Input: {args.input}")
    print(f"Fuse mode: {args.fuse_mode}")
    if args.fuse_mode in {"weighted_sum", "geometric"}:
        print(f"Alpha: {args.alpha:.4f}")
    print("Evaluation Metrics:")
    for name, value in results.items():
        if hasattr(value, "numel") and value.numel() == 1:
            print(f"{name}: {value.item():.6f}")
        else:
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()

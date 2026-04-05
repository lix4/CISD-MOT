import json

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


SCORE_KEY = "yolo_confs"
IOU_THRESH = 0.5
PATH = "/home/xiwenli/CISD-MOT/eval_dump.json"


def main():
    with open(PATH, "r") as f:
        data = json.load(f)

    predictions = []
    groundtruth = []
    pred_count_by_class = {}
    gt_count_by_class = {}

    for pred, gt in zip(data["predictions"], data["groundtruth"]):
        pred_boxes = torch.tensor(pred["boxes"], dtype=torch.float32)
        pred_labels = torch.tensor(pred["labels"], dtype=torch.int64)
        pred_scores = torch.tensor(pred[SCORE_KEY], dtype=torch.float32)

        gt_boxes = torch.tensor(gt["boxes"], dtype=torch.float32)
        gt_labels = torch.tensor(gt["labels"], dtype=torch.int64)

        predictions.append(
            {
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels,
            }
        )
        groundtruth.append(
            {
                "boxes": gt_boxes,
                "labels": gt_labels,
            }
        )

        for cls in pred_labels.tolist():
            pred_count_by_class[cls] = pred_count_by_class.get(cls, 0) + 1
        for cls in gt_labels.tolist():
            gt_count_by_class[cls] = gt_count_by_class.get(cls, 0) + 1

    try:
        metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[IOU_THRESH],
            class_metrics=True,
            average="macro",
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "torchmetrics detection AP requires `pycocotools` or `faster-coco-eval`.\n"
            "Install one of them first, then rerun this script."
        ) from exc

    metric.update(predictions, groundtruth)
    results = metric.compute()

    print(f"score_key={SCORE_KEY}")
    print(f"iou_thresh={IOU_THRESH}")
    print(f"num_images={len(predictions)}")

    all_classes = sorted(set(pred_count_by_class) | set(gt_count_by_class))
    for cls in all_classes:
        print(
            f"class_{cls}_num_gt={gt_count_by_class.get(cls, 0)} "
            f"class_{cls}_num_pred={pred_count_by_class.get(cls, 0)}"
        )

    print(f"overall_bbox_map={results['map'].item()}")
    print(f"overall_bbox_map_50={results['map_50'].item()}")

    classes = results.get("classes")
    map_per_class = results.get("map_per_class")
    if classes is not None and map_per_class is not None:
        for cls, ap in zip(classes.tolist(), map_per_class.tolist()):
            print(f"class_{cls}_bbox_map={ap}")


if __name__ == "__main__":
    main()

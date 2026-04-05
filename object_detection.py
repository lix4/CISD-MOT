import numpy as np
import torch
from datasets.CustomYOLODataset import CustomYOLODataset
# from ultralytics import YOLO
from types import SimpleNamespace
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
import torchvision
from torch.utils.data import DataLoader
from utils import *
import argparse
from pathlib import Path
from typing import Union, Optional
import albumentations as A
# import torchvision.transforms as ToTensor
from albumentations.pytorch import ToTensorV2
import cv2
import os
# from torchmetrics.detection.mean_ap import MeanAveragePrecision

def custom_collate(batch):
    pic_paths, images, targets = zip(*batch)          # tuple 长度 = B
    images = torch.stack(images, 0)                  # B,C,H,W

    all_labels = []                                  # 汇总所有 bbox
    for img_idx, boxes in enumerate(targets):
        # boxes: Tensor (Ni, 4)  or  shape = (0, 4) if no object
        # print(boxes.shape)
        # boxes = boxes[0]
        if boxes.dim() == 3 and boxes.size(0) == 1:    # 1×N×4 → squeeze
            boxes = boxes.squeeze(0)
        # print(boxes.shape)
        if boxes.numel() == 0:
            continue                                 # 跳过空图片

        img_idx_col = torch.full(
            (boxes.shape[0], 1), img_idx, dtype=torch.float32
        )                                            # (Ni,1)
        cls_col = torch.zeros(
            (boxes.shape[0], 1), dtype=torch.float32
        )                                            # (Ni,1) 全 0
        # print(img_idx_col.shape, cls_col.shape, boxes.shape)
        labels = torch.cat([img_idx_col, cls_col, boxes], dim=1)  # (Ni,6)
        all_labels.append(labels)

    if all_labels:
        labels = torch.cat(all_labels, dim=0)        # (N_total, 6)
    else:
        labels = torch.zeros((0, 6), dtype=torch.float32)

    return pic_paths, images, {
        "batch_idx": labels[:, 0],
        "cls":       labels[:, 1],
        "bboxes":    labels[:, 2:],                  # (N_total,4)
    }


def get_yolo11_transforms_224(
        hsv_h=0.1, hsv_s=0.9, hsv_v=0.9,
        deg=45, translate=0.1, scale_low=0.9, scale_high=1.1, shear=10,
        mosaic_prob=0.0
):
    """
    Albumentations Compose 覆盖 YOLOv11 训练阶段常用增强。
    输入要求：
        • image  : BGR→RGB 后的 numpy array, shape = (224,224,3)
        • bboxes : 归一化 xywh, 范围 0-1
    返回 (image, bboxes, labels)；image 已转成 FloatTensor, [0-1]。
    """
    # ----- 单张图增强 -----
    single_img_aug = [
        # 几何：仿射 + 透视
        A.Affine(
            rotate=(-deg, deg),
            translate_percent=(-translate, translate),
            scale=(scale_low, scale_high),
            shear=(-shear, shear),
            fit_output=False,   # 输出仍是 224×224
            p=0.9
        ),
        A.Perspective(scale=(0.0, 0.001), keep_size=True, p=0.3),
        # 颜色：HSV 抖动
        A.HueSaturationValue(
            hue_shift_limit=int(hsv_h * 180),       # OpenCV-HSV: 0-179
            sat_shift_limit=int(hsv_s * 255),
            val_shift_limit=int(hsv_v * 255),
            p=0.8
        ),
        # 翻转
        A.HorizontalFlip(p=0.5),
    ]

    # # ----- 组合增强（可选）-----
    # combo_aug = A.OneOf([
    #     # A.MixUp(p=1.0),
    #     A.CutMix(num_holes=1, max_h_size=112, max_w_size=112, p=1.0)
    # ], p=mixup_prob + cutmix_prob)

    # ----- Mosaic（需 4 张图，示例给接口占位）-----
    # 若想用 Mosaic，可先把 4 张 224×224 拼成 448×448、增强后再中心裁回 224×224，
    # 实现较繁琐，通常会直接把 dataset 输出尺寸调大到 640。
    # mosaic_stub = A.NoOp(p=1.0 - mosaic_prob)  # 保留占位

    return A.Compose(
        single_img_aug + [A.ToFloat(max_value=255.0), ToTensorV2()],
        bbox_params=A.BboxParams(
            format="yolo",       # 归一化 xywh
            min_visibility=0.0
        )
    )

def get_val_transforms_224():
    return A.Compose(
        [A.ToFloat(max_value=255.0), ToTensorV2()],
        bbox_params=A.BboxParams(
            format="yolo",           # 你已是归一化 xywh
        )
    )

def build_dataloaders(batch_size=8):
    train_ds = CustomYOLODataset(
        "./meta-files/MEB/DAMEBlist_train_e2e_new_1.txt",
        transform = get_yolo11_transforms_224()
        # transform = torchvision.transforms.ToTensor()
    )
    val_ds = CustomYOLODataset(
        # "./meta-files/MEB/DAMEBlist_test_e2e_new_1.txt",
        "./meta-files/validlist_e2e_new_1.txt",
        # "./meta-files/testlist_e2e_new_2.txt",
        transform = get_val_transforms_224()
        # transform=torchvision.transforms.ToTensor()
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=custom_collate)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    return train_dl, val_dl

def load_model(nc:int=1, od_model: str = None, ckpt:Optional[Union[str, Path]] = None, device="cuda"):
    # model = DetectionModel(cfg='yolo11l.yaml', ch=3, nc=nc).to(device)
    print("Loading Detection Model %s" % od_model)
    model = DetectionModel(cfg=od_model, ch=3, nc=nc).to(device)
    if ckpt:
        ckpt = Path(ckpt).expanduser()
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        print(f"✅  已加载权重: {ckpt}")
    # 伪参数供 v8DetectionLoss 使用
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5,
        fl_gamma=0.0, label_smoothing=0.0,
        nbs=64, classes=None
    )
    return model


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch_idx, (pic_path, images, targets) in enumerate(train_loader):
        images = images.to(device)
        # targets = targets.to(device)
        optimizer.zero_grad()
        # print(images.shape)
        # 前向传播：outputs 形状、结构依赖于具体模型
        # print(images.dtype)
        results = model(images)
        # print(results[0].shape, results[1].shape, results[2].shape)
        # pred=results[0]
        # print(pred.shape)
        # print(targets)
        # train output 3 torch.Size([8, 65, 28, 28]) torch.Size([8, 65, 14, 14]) torch.Size([8, 65, 7, 7])
        # print("train output", len(results), results[0].shape, results[1].shape, results[2].shape)
        total_loss, loss_items = criterion(results, targets)
        loss = total_loss.sum()     # 这是一个 Tensor
        loss.backward()
        optimizer.step()

        box, cls, dfl = loss_items
        box_total, cls_total, dfl_total = total_loss

        print(f"Loss: {loss.item():.4f} | Box: {box:.4f} ({box_total:.2f}) | "
            f"Cls: {cls:.4f} ({cls_total:.2f}) | DFL: {dfl:.4f} ({dfl_total:.2f})")

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 训练 / 测试脚本")
    sub = parser.add_subparsers(dest="mode", required=True)

    # ▶ train 子命令
    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--epochs",   type=int,   default=100)
    train_cmd.add_argument("--patience", type=int,   default=100)
    train_cmd.add_argument("--lr",       type=float, default=1e-4)
    train_cmd.add_argument("--job_name",       type=str)
    train_cmd.add_argument("--weights", type=str, help="yolo pretrained .pt 权重文件路径")
    train_cmd.add_argument("--od_model", type=str, required=True, help="YOLO 配置文件路径 (cfg)")
    
    # ▶ test 子命令
    test_cmd = sub.add_parser("test")
    test_cmd.add_argument("--weights", type=str, required=True, help="要评估的 .pt 权重文件路径")
    test_cmd.add_argument("--od_model", type=str, required=True, help="YOLO 配置文件路径 (cfg)")

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------ train ------------------------------------
    if args.mode == "train":
        train_loader, val_loader = build_dataloaders()
        model     = load_model(nc=1, od_model=args.od_model, ckpt=args.weights, device=device)
        criterion = v8DetectionLoss(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_map, es_counter = -1, 0
        for epoch in range(args.epochs):
            # --- 使用原来的 train_one_epoch ---
            print(f"[Train] Epoch {epoch+1}")
            train_one_epoch(model, train_loader, criterion, optimizer)
            
            # --- 验证 ---
            with torch.no_grad():
                map50, map75, map5095 = evaluate_map(model.eval(), val_loader, device)
            print(f"[Val] Epoch {epoch+1:03d} | mAP@50={map50:.4f} | mAP@75={map75:.4f} | mAP@50-95={map5095:.4f}")
            
            # 保存最新 & 最佳
            save_dir = os.path.join("runs", args.job_name)
            torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch.pt"))
            if map5095 > best_map:
                best_map = map5095
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                print(f"✨  The best weights have been saved (mAP@50-95={best_map:.4f})")
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= args.patience:
                    print("⏹️  Early Stopped")
                    break

    # ------------------------------------ test ------------------------------------
    elif args.mode == "test":
        _, val_loader = build_dataloaders()
        model = load_model(nc=1, od_model=args.od_model, ckpt=args.weights, device=device).eval()

        with torch.no_grad():
            map50, map75, map5095 = evaluate_map(model, val_loader, device, save_dir=os.path.join("runs", args.job_name if hasattr(args, "job_name") else "test_vis"), max_vis=50)
        print(f"[Test] mAP@50={map50:.4f} | mAP@75={map75:.4f} | mAP@50-95={map5095:.4f}")

import os
import cv2
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
# from deep_sort_realtime.deepsort_tracker import DeepSort

def custom_collate(batch):
    pic_paths, images, targets = zip(*batch)
    images = torch.stack(images, 0)

    all_labels = []
    for image_idx, boxes in enumerate(targets):
        boxes=boxes[0]
        # print(boxes.shape)
        if boxes.numel() == 0:
            continue
        # 假设你已经得到 [N, 4] 的 box（x, y, w, h）格式：
        boxes = boxes[:, 1:]  # 去掉 frame_idx
        image_idx_col = torch.full((boxes.shape[0], 1), image_idx, dtype=torch.float32)
        class_col = torch.zeros((boxes.shape[0], 1), dtype=torch.float32)  # 全部设为 0
        # print(image_idx_col.shape, class_col.shape, boxes.shape)
        labels = torch.cat([image_idx_col, class_col, boxes], dim=1)  # → [N, 6]
        all_labels.append(labels)

    if all_labels:
        labels = torch.cat(all_labels, dim=0)  # [N_total, 6]
    else:
        labels = torch.zeros((0, 6), dtype=torch.float32)

    return pic_paths, images, {'batch_idx': labels[:,0], 
                               'cls': labels[:,1],
                               'bboxes': labels[:, 2:]
                               }


    
# 示例：构造 DataLoader
# 构建训练和验证的 DataLoader，假设分别有 "data/train_list.txt" 和 "data/val_list.txt"
train_dataset = CustomYOLODataset("./meta-files/trainlist_e2e_new_1.txt", transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
val_dataset = CustomYOLODataset("./meta-files/testlist_e2e_new_1.txt", transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)

# 加载预训练模型（例如 yolov8n）
model = DetectionModel(cfg='yolov8n.yaml', ch=3, nc=1)  # channels=3, classes=1
model = model.cuda()
# print("model.yaml['nc']:", model.yaml['nc'])
# print("nc:", model.model[-1].nc)
# print("per-scale output channels:", [m[-1].out_channels for m in model.model.model[-1].cv2])  # should be [18, 18, 18]
# exit()
# 2. 构造 fake args
model.args = SimpleNamespace(
    box=7.5,
    cls=0.5,
    dfl=1.5,
    fl_gamma=0.0,
    label_smoothing=0.0,
    nbs=64,       # nominal batch size
    classes=None  # for class filtering
)

criterion = v8DetectionLoss(model)  # loss函数和model有关联
# 初始化 DeepSORT 跟踪器
# deepsort = DeepSort()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# early stopping & save best model
best_val_map = -1
early_stop_counter = 0
early_stop_patience = 100
save_path = "best_model.pt"
num_epochs = 100

def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch_idx, (pic_path, images, targets) in enumerate(train_loader):
        images = images.to(device)
        # targets = targets.to(device)
        optimizer.zero_grad()
        
        # 前向传播：outputs 形状、结构依赖于具体模型
        results = model(images)
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

# 训练和验证的 for 循环逻辑
for epoch in range(num_epochs):
    
    ########## train epoch ##########
    train_one_epoch(model, train_loader, criterion, optimizer)
    ########## train epoch ##########
    
    # 验证阶段
    # ============ 验证阶段 =============
    # ========= 验证阶段（使用 mAP） =========
    map50, map5095 = evaluate_map(model, val_loader, device)
    print(f"[Validation] Epoch {epoch+1} | mAP@50: {map50:.4f} | mAP@50-95: {map5095:.4f}")
    
    torch.save(model.state_dict(), 'yolo_chpt.pt')
    if map5095 > best_val_map:
        best_val_map = map5095
        early_stop_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved new best model at epoch {epoch+1} with mAP@0.5:0.95 {map5095:.4f}")
    else:
        early_stop_counter += 1
        print(f"⚠️ No improvement. Early stop counter: {early_stop_counter}/{early_stop_patience}")

    if early_stop_counter >= early_stop_patience:
        print("⏹️ Early stopping triggered.")
        break
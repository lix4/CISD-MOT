import os
import torch
import cv2
from collections import defaultdict
from torch.utils.data import DataLoader
from models.MOT_IVD_v2_1 import MOT_IVD_v2_1
from datasets.dataset_mot import listDataset, avivd_collate_fn
import torchvision.transforms as T

# ========= 配置 =========
CHECKPOINT_PATH = './runs/MOT_classification_v2_1/best_model_epoch_18_mAP_0.8477.pth'
DATA_ROOT = '/uu/sci.utah.edu/projects/smartair/Dataset'
TXT_LIST = './meta-files/testlist_e2e_new_1.txt'
JSON_PATH = './datasets/valid_tracks.json'
SAVE_DIR = 'eval_outputs/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLS_NAMES = ['Moving', 'Idling', 'Off']
COLOR_MAP = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}  # Green, Red, Blue
L = 16
TARGET_FRAME = L - 1

os.makedirs(SAVE_DIR, exist_ok=True)

# ========= 模型加载 =========
model = MOT_IVD_v2_1(num_classes=3)
model.load_state_dict(torch.load(CHECKPOINT_PATH)['model_state_dict'])
model.to(DEVICE)
model.eval()

# ========= 数据加载 =========
transform = T.Compose([T.ToTensor()])
dataset = listDataset(DATA_ROOT, TXT_LIST, json_path=JSON_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=1, collate_fn=avivd_collate_fn, shuffle=False, num_workers=8, pin_memory=True)

# 定义视频写入器
OUTPUT_VIDEO = 'output_all.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 25, (320, 240))
# ========= 推理与保存 =========
with torch.no_grad():
    for batch_idx, (bboxes, mask, audio, labels, video_ids, bn_indices) in enumerate(loader):
        all_keys = set()
        frame_groups = defaultdict(list)

        if bboxes.shape[0] > 0:
            bboxes, audio = bboxes.to(DEVICE), audio.to(DEVICE)
            logits = model(bboxes, audio)  # [BN, L, C]
            preds = logits.argmax(dim=-1).cpu()
            bboxes = bboxes.cpu()
            labels = labels.cpu()
            mask = mask.cpu()

            for idx_bn, (vid, (b, l, m, p), (bi, ni)) in enumerate(zip(video_ids, zip(bboxes, labels, mask, preds), bn_indices)):
                frame_id = int(vid.split('-')[-1][:5]) - (15 - TARGET_FRAME) * 4
                frame_key = (vid, frame_id)
                all_keys.add(frame_key)

                if m[TARGET_FRAME]:
                    frame_groups[frame_key].append((b[TARGET_FRAME], l[TARGET_FRAME].item(), p[TARGET_FRAME].item()))

        # 如果完全没有 bbox 的情况也要记录 video_id
        for vid in video_ids:
            frame_id = int(vid.split('-')[-1][:5]) - (15 - TARGET_FRAME) * 4
            frame_key = (vid, frame_id)
            all_keys.add(frame_key)

        # 遍历所有帧，画图
        for (vid, frame_id) in all_keys:
            img_path = os.path.join(
                DATA_ROOT, 'Video/LDS',
                vid.split('-')[-3][:-2], vid.split('-')[-3], vid.split('-')[-2],
                'rgb-images', f'{frame_id:05d}.jpg'
            )
            if not os.path.exists(img_path):
                print(f"[Missing] {img_path}")
                continue

            img = cv2.imread(img_path)
            print(img.shape)
            entries = frame_groups.get((vid, frame_id), [])

            if len(entries) == 0:
                # 没有车
                cv2.putText(img, 'No vehicle detected', (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
                suffix = '_empty'
            else:
                # 有车
                for (box, gt, pred_cls) in entries:
                    x1, y1, x2, y2 = map(int, box)
                    color = COLOR_MAP[pred_cls]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f'Pred: {CLS_NAMES[pred_cls]}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if gt != -1:
                        cv2.putText(img, f'GT: {CLS_NAMES[gt]}', (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                suffix = ''

            save_name = f'{vid.split("-")[-3]}-{vid.split("-")[-2]}-{frame_id:05d}{suffix}.jpg'
            save_path = os.path.join(SAVE_DIR, save_name)
            # cv2.imwrite(save_path, img)
            writer.write(img)

writer.release()

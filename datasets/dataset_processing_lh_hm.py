import os
import sys
import json
from tqdm import tqdm
from dataset_mot_backup import listDataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torch
import cv2
from pathlib import Path
from typing import Union, Optional
from ultralytics.nn.tasks import DetectionModel
from types import SimpleNamespace
from ultralytics.utils.ops import non_max_suppression
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from CustomYOLODataset import CustomYOLODataset, custom_collate

# 获取当前文件的上上级目录（即项目根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# from utils import box_iou

def load_model(nc:int=1, ckpt:Optional[Union[str, Path]] = None, device="cuda"):
    model = DetectionModel(cfg='yolov8n.yaml', ch=3, nc=nc).to(device)
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

def box_iou_xyxy(box1, box2):          # 单框 IoU，与你前面的实现等价
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter)

def match_preds_with_gt(pred_boxes, gt_boxes, gt_labels,
                        iou_thr: float = 0.5, scores=None):
    Mp, Mg = len(pred_boxes), len(gt_boxes)
    if Mp == 0:
        return []

    order = np.argsort(-(scores if scores is not None else np.zeros(Mp)))
    assigned = [-1] * Mp          # 默认 -1
    gt_used  = np.zeros(Mg, dtype=bool)

    for pi in order:
        best_iou, best_g = 0.0, -1
        for gi in range(Mg):
            if gt_used[gi]:
                continue
            iou = box_iou_xyxy(pred_boxes[pi], gt_boxes[gi])
            if iou > best_iou:
                best_iou, best_g = iou, gi

        # **只有同时满足两件事才写入标签**
        if best_g != -1 and best_iou >= iou_thr:
            assigned[pi] = int(gt_labels[best_g])
            gt_used[best_g] = True

    return assigned



def reset_deepsort(ds: DeepSort):
    """
    手动清空 deep-sort-realtime 1.3.2 的内部状态
    """
    t = ds.tracker                      # <-- deep_sort.tracker.Tracker 实例
    t.tracks.clear()                    # 清空已有 Track 对象
    t._next_id = 1                      # ID 计数器归零
    # 外观特征度量（metric）里也有缓存，用 dict() / list 清一下
    if hasattr(t.metric, 'samples'):
        t.metric.samples.clear()


def generate_tracks_json(dataset, kwargs, gt=True, save_path='output_tracks_stream.json'):
    # dataset.train = False
    # loader = DataLoader(dataset , batch_size=1, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,  collate_fn=custom_collate, **kwargs)

    model = load_model(nc=1, ckpt='../runs/MOT_detection_1/best_model.pt').cuda()
    model.eval()
    # tracker  = DeepSort(
    #     max_age=3,           # 连续丢 4 帧就结束轨迹
    #     n_init=1,            # 起轨确认阈值
    #     nms_max_overlap=1.0, # 不在 tracker 内做 NMS
    #     max_cosine_distance=0.2,  # ReID 阈值
    #     embedder="mobilenet",     # 轻量特征提取器
    # )
    tracker = DeepSort(
        max_age = 3,  # 半秒容忍
        n_init  = 1,
        max_iou_distance = 0.95
    )
    
    # print(len(loader))
    # exit(0)
    L = 16
    d = 4
    with open(save_path, 'w') as f:
        f.write('{\n')

        for idx, item in enumerate(tqdm(loader, desc="Extracting tracks", disable=True)):
            # sample = item[0] if isinstance(item, list) else item

            video_id = item[0][0]
            

            print("########## ", video_id, " ##########")

            if not gt:
                imgpath = video_id
                im_split = imgpath.replace("\\", "/").split('/')
                im_ind = int(im_split[-1][:5])
                img_folder = imgpath[:-10]  # remove /xxxxx.jpg

                with torch.no_grad():
                    trajectories = defaultdict(lambda: {
                        "boxes" : [None]*L,
                        "labels": [-1]*L,       # 如果你要存类别
                    })

    
                    # === Load video clip ===
                    # DeepSORT tracker 只实例化一次即可
                    for j, offset in enumerate(reversed(range(L))):
                        i_temp   = im_ind - offset * d
                        img_fp   = os.path.join(img_folder, f"{i_temp:05d}.jpg")

                        # --- (1) 读取原 BGR、准备网络输入 ---
                        orig_bgr = cv2.imread(img_fp)                      # H×W×3, BGR
                        inp_rgb  = cv2.resize(orig_bgr[..., ::-1], (224,224))
                        inp      = T.ToTensor()(inp_rgb).unsqueeze(0).cuda()

                        # exit()
                        # gt_labels = 
                        anno_pth = img_fp.replace('rgb-images', 'labels/av')[:-4] + '_g.txt'
                        anno = np.loadtxt(anno_pth)
                        # print(anno.shape)
                        if anno.shape[0] == 0:
                            gt_xyxy = torch.zeros((0, 5))
                            gt_labels = torch.zeros((0))
                        elif anno.ndim == 1:
                            anno = np.expand_dims(anno, 0)
                            gt_xyxy = torch.from_numpy(anno[:,1:])
                            gt_labels = torch.from_numpy(anno[:,0])
                        else:
                            gt_xyxy = torch.from_numpy(anno[:,1:])
                            gt_labels = torch.from_numpy(anno[:,0])
                        # print(anno.shape)

                        # --- (2) YOLO 推理 + NMS ---
                        with torch.no_grad():
                            raw  = model(inp)
                        preds = non_max_suppression(raw, 0.7, 0.7)[0]
                        if preds is None or len(preds) == 0:
                            tracker.update_tracks(np.empty((0,4)), frame=orig_bgr)   # 喂空帧
                            continue
                        

                        # --- (3) 坐标反缩放到 320×240 ---
                        scale   = np.array([320/224, 240/224, 320/224, 240/224])
                        # boxes_xywh = preds[:, :4].cpu().numpy()
                        boxes_xyxy   = (preds[:, :4].cpu().numpy() * scale)
                        confs   = preds[:, 4].cpu().numpy()
                        clses   = preds[:, 5].cpu().numpy().astype(int)
                        # print(clses)
                        # print(sample['labels'].shape)
                        ############ TODO: load directly from  ############
                        # gt_xyxy = sample['bboxes'][0, :, -1, :]
                        # gt_labels = sample['labels'][0, :, -1]

                        # ↓② 转 raw_detections 格式
                        raw_dets = [([x1, y1, x2-x1, y2-y1], float(conf), int(clas))
                                    for (x1,y1,x2,y2), conf, clas in zip(boxes_xyxy, confs, clses)]
                        
                        # print(raw_dets)

                        # (可选) IoU 贴 GT 类别 → labels_inherited
                        labels_inherited = [-1] * len(boxes_xyxy)
                        # print('before match_preds_with_gt', boxes_xyxy, gt_xyxy)
                        labels_inherited = match_preds_with_gt(boxes_xyxy, gt_xyxy, gt_labels, scores=confs)
                        print("check labels_inherited", gt_labels, labels_inherited)
                        assert len(labels_inherited) == len(raw_dets)
                        # print(labels_inherited)
                        # --- (4) DeepSORT 更新 ---
                        tracks = tracker.update_tracks(raw_dets, frame=orig_bgr)
                        print("A", len(labels_inherited), len(tracks), len(boxes_xyxy))
                        # for m, t in enumerate(tracks):
                        #     if t.is_confirmed() and t.time_since_update == 0:
                        #         # continue
                        #         tid = t.track_id
                        #         trajectories[tid]["boxes"][j]  = t.to_ltrb()
                        #         print("t box", t.to_ltrb())
                        #         try:
                        #             trajectories[tid]["labels"][j] = int(labels_inherited[m])
                        #         except Exception as e:
                        #             print(video_id)
                        #             print(raw_dets)
                        #             print(boxes_xyxy)
                        #             # exit()
                        #             raise ValueError("out of range")
                        confirmed_tracks = [t for t in tracks if t.is_confirmed() and t.time_since_update == 0]

                        for t in confirmed_tracks:
                            tid = t.track_id
                            trajectories[tid]["boxes"][j] = t.to_ltrb()

                            # 找到这个 track 的 bbox 与哪个 raw_dets 匹配最好（通过 IoU）
                            best_iou, best_det = 0.0, -1
                            for m, (x1, y1, w, h) in enumerate([det[0] for det in raw_dets]):
                                box_det = [x1, y1, x1 + w, y1 + h]
                                iou = box_iou_xyxy(box_det, t.to_ltrb())
                                if iou > best_iou:
                                    best_iou = iou
                                    best_det = m

                            if best_det != -1:
                                trajectories[tid]["labels"][j] = int(labels_inherited[best_det])
                            else:
                                trajectories[tid]["labels"][j] = -1  # 找不到匹配就填 -1

                    # clip 结束后
                    reset_deepsort(tracker)            # clip 结束后调用
                    # —— ② 把 dict 转成张量格式  [N, L, 4] / [N, L]
                    valid_tracks = {tid:traj for tid,traj in trajectories.items()
                                    if sum(b is not None for b in traj["boxes"]) >= 2}  # 起码命中2帧

                    N = len(valid_tracks)
                    bboxes = torch.zeros((N, L, 4), dtype=torch.float32)
                    mask   = torch.zeros((N, L),     dtype=torch.bool)
                    labels = torch.full((N, L), -1,  dtype=torch.long)

                    for n,(tid,traj) in enumerate(valid_tracks.items()):
                        for k, box in enumerate(traj["boxes"]):
                            if box is not None and traj['labels'][k] != -1:
                                bboxes[n, k] = torch.tensor(box, dtype=torch.float32)
                                mask[n, k]   = True                 # 只有 box+合法标签 才置 True
                                labels[n, k] = traj['labels'][k]
                    ##### DeepSORT tracker #####

            else:
                bboxes = sample['bboxes'].squeeze(0).tolist()
                labels = sample['labels'].squeeze(0).tolist()
                mask = sample['mask'].squeeze(0).tolist()

            ################# write into json #################

            video_dict = {}
            for i in range(len(bboxes)):
                track_id = f"track_{i}"
                video_dict[track_id] = {
                    'boxes': [[float(x) for x in box] if m else None for box, m in zip(bboxes[i], mask[i])],
                    'labels': [int(lbl) if m else -1 for lbl, m in zip(labels[i], mask[i])],
                    'mask': [bool(m) for m in mask[i]]
                }

            json_str = json.dumps({video_id: video_dict}, indent=2)
            if idx > 0:
                f.write(',\n')  # ← 正确拼接 JSON 对象
            f.write(json_str[1:-1])  # 去掉外层大括号，避免嵌套

        f.write('\n}\n')  # 最终关闭 JSON

    print(f"✅ Incremental JSON saved to {save_path}")



if __name__ == '__main__':

    basepath='/uu/sci.utah.edu/projects/smartair/Dataset'
    trainlist='../meta-files/testlist_e2e_new_1.txt'
    dataset_use='car'
    init_width=224
    init_height=224
    clip_duration=16
    num_workers=8
    kwargs = {'num_workers': num_workers, 'pin_memory': True, 'prefetch_factor': 2}
    # dataset=listDataset(basepath, trainlist, shape=(init_width, init_height),
    #                    transform=torchvision.transforms.Compose([
    #                        torchvision.transforms.ToTensor(),
    #                    ]), 
    #                    train=False, 
    #                    clip_duration=clip_duration)
    train_ds = CustomYOLODataset(
        "../meta-files/trainlist_e2e_new_1.txt",
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    )
    

    # generate_tracks_json(dataset, gt=False, save_path='train_tracks_yolo.json')
    generate_tracks_json(train_ds, kwargs, gt=False, save_path='train_tracks_yolo_1.json')

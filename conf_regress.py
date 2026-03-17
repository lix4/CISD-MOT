import torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as T
from datasets.dataset_mot import listDataset, avivd_collate_fn
from models.MOT_IVD_v1_9_audio_cl import MOT_IVD_v1_9
from ultralytics.nn.tasks import DetectionModel
from types import SimpleNamespace
from ultralytics.utils.ops import non_max_suppression
from os.path import join
import cv2
import torchvision

class LRCalibrator(nn.Module):
    """ 2-D logistic regression:  σ(a*conf + b*cls + c) """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):              # x: [N,2]
        return torch.sigmoid(self.linear(x)).squeeze(1)   # [N]

def fit_calibrator(det_model, cls_model, dataloader, device):
    det_model.eval()
    cls_model.eval()
    feats, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            _, bboxes, mask, audio, lab, vid_ids, _, yolo_conf = batch
            if bboxes.shape[0] == 0:     # 跳过空帧
                continue
            
            # print(vid_ids)
            tmp = vid_ids[0].split('-')
            # print(tmp)
            img_path = join('/uu/sci.utah.edu/projects/smartair/Dataset/Video/LDS', tmp[0][:-2], tmp[0], tmp[1], 'rgb-images', tmp[2])
            print(img_path)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224)) 
            image = torchvision.transforms.ToTensor()(image).to(device)
            image = image.unsqueeze(0)   

            with torch.no_grad():
                raw_outputs = det_model(image)
            
            batch_preds = raw_outputs[0]
            # batch_preds = batch_preds.permute(0, 2, 1)
            print(len(raw_outputs), raw_outputs[0].shape, raw_outputs[1][0].shape)
            # take any box whose conf >= 0.0
            # output_list = non_max_suppression(batch_preds, 0.0, 0.5)[0]
            pred1 = batch_preds[0:1]      # [1, C, N]
            assert pred1.shape == batch_preds.shape
            # 1. “打标签”记录原始 box 索引
            idxs = torch.arange(pred1.shape[-1], device=pred1.device).reshape(1, 1, -1)  # [1,1,N]
            pred1_idx = torch.cat([pred1, idxs], dim=1)  # [1, C+1, N]

            # 2. 调用 NMS，conf_thres=0 保留所有，iou_thres=0.5 划分
            #    输出是 List[Tensor]，这里只有一张图，所以取 [0]
            out = non_max_suppression(pred1_idx, conf_thres=0.0, iou_thres=0.5)[0]
            # out.shape == (num_kept, 6 + nmasks + 1)
            # 最后一列 out[:, -1] 就是 kept 的原始索引

            kept_idx = out[:, -1].long().unique()
            all_idx  = torch.arange(pred1.shape[-1], device=pred1.device)
            
            conf_all = batch_preds[0, 4, :]                # [N]

            # 3. 分离 kept / suppressed
            kept_boxes       = out[:, :6]  # (x1,y1,x2,y2,conf,cls)
            suppressed_idx   = all_idx[~all_idx.unsqueeze(1).eq(kept_idx).any(1)]
            suppressed_boxes = pred1[0, :6, suppressed_idx].T  # 同样取前6列并转成 (n,6)

            print((conf_all >= 0.6).any())
            kept_confs       = conf_all[kept_idx]       # 保留框的 conf
            suppressed_confs = conf_all[suppressed_idx] # 抑制框的 conf
            print(f"Kept {len(kept_idx)} boxes, suppressed {len(suppressed_idx)} boxes")
            print(kept_confs, suppressed_confs)

            bboxes = bboxes.to(device)
            exit()
            # mask   = mask.to(device)[:, -1:]
            audio  = audio.to(device)

            logits, _ = cls_model(bboxes, audio)      # [N,C]
            probs     = torch.softmax(logits, -1) # …
            # print(yolo_conf.shape, probs.shape)
            cls_prob  = probs.max(-1).values.cpu()   # [N]

            #========= IoU label =========
            # 假设 batch 已经带 GT bboxes → 计算 IoU
            # 这里只给示例占位:
            # iou = batch[-2]               # [N] 你自己算 IoU>=0.5 ⇒1 否则0
            feats.append(torch.stack([yolo_conf, cls_prob], 1))
            labels.append((iou >= 0.5).float())

    X = torch.cat(feats, 0)
    y = torch.cat(labels, 0)

    calib = LRCalibrator().to(X.device)
    opt   = optim.Adam(calib.parameters(), lr=1e-2)
    lossf = nn.BCELoss()

    for _ in range(300):
        pred = calib(X)
        loss = lossf(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()

    # 把 a,b,c 存下来（只 3 个数字，也可以写入 yaml）
    a, b = calib.linear.weight.data[0].tolist()
    c    = calib.linear.bias.item()
    torch.save({'a':a, 'b':b, 'c':c}, 'calibrator.pt')
    return {'a':a, 'b':b, 'c':c}

# ---------- 在脚本最开头 ----------
# calib_params = torch.load('calibrator.pt')   # {'a':…, 'b':…, 'c':…}

def fused_score(conf, cls_prob):
    """conf, cls_prob: Tensor[N]"""
    z = calib_params
    return torch.sigmoid(z['a']*conf + z['b']*cls_prob + z['c'])


########### load det model ###########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

det_model = DetectionModel(cfg='yolo11s.yaml', ch=3, nc=1).to(device)
ckpt = './runs/MOT_detection_yolo11s_aug_af/best_model.pt'
det_model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
print(f"✅  已加载权重: {ckpt}")
# 伪参数供 v8DetectionLoss 使用
det_model.args = SimpleNamespace(
    box=7.5, cls=0.5, dfl=1.5,
    fl_gamma=0.0, label_smoothing=0.0,
    nbs=64, classes=None
)

########### load cls model ###########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cls_model = MOT_IVD_v1_9(num_classes=3).cuda()
ckpt = torch.load("./runs/MOT_classification_cl_v1_9_yolo11s_af/best_model_epoch_12_mAP_0.9099.pth", map_location=device)
state_dict = ckpt if "model_state_dict" not in ckpt else ckpt["model_state_dict"]
cls_model.load_state_dict(state_dict, strict=True)

########### dataset ###########
kwargs = {'num_workers': 8, 'pin_memory': True, 'prefetch_factor': 2}
# 拟合校准器
valid_dataset = listDataset(
        base_path='/uu/sci.utah.edu/projects/smartair/Dataset',
        txt_list='./meta-files/validlist_e2e_new_1.txt',
        json_path='./datasets/valid_tracks_yolov11s_af_03_05_corrected_padded.json',
        load_frames=False,
        transform=T.ToTensor()
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=avivd_collate_fn,
    **kwargs
)
calib_params = fit_calibrator(det_model, cls_model, valid_loader, device)
print("Calibrator params:", calib_params)


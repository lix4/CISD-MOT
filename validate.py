import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAveragePrecision
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from collections import defaultdict
# from models.MOT_IVD_v2_1 import MOT_IVD_v2_1
# from models.MOT_IVD_v2_2 import MOT_IVD_v2_2
from models.MOT_IVD_v1_4 import MOT_IVD_v1_4
import torchvision.transforms as T
from datasets.dataset_mot import listDataset, avivd_collate_fn

def validate_one_epoch(model, dataloader, device, num_classes=3):
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    ap_metric = MulticlassAveragePrecision(num_classes=num_classes, average=None).to(device)  # 每类单独输出
    # counter = defaultdict(int)
    
    with torch.no_grad():
        for batch in dataloader:
            # print("aere")
            clips, bboxes, mask, audio, labels, _, _ = batch
            # print("here")
            if clips.shape[0] == 0:
                continue
            # clips = clips.to(device)
            bboxes = bboxes.to(device)
            mask = mask.to(device)
            audio = audio.to(device)
            labels = labels.to(device)

            logits = model(bboxes, audio)  # [BN, L, C]
            preds = torch.argmax(logits, dim=-1)  # [BN, L]

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())
            # print(labels)
            # for cls in range(3):
            #     counter[cls]+=torch.sum(labels==cls)
            # print(labels.cpu().shape, mask.cpu().shape)
            all_masks.append(mask.cpu())

    preds = torch.cat(all_preds, dim=0).view(-1)
    targets = torch.cat(all_targets, dim=0).view(-1)
    masks = torch.cat(all_masks, dim=0).view(-1)

    preds = preds[masks]
    targets = targets[masks]

    # Compute F1
    f1 = f1_score(targets.numpy(), preds.numpy(), average='macro')

    # Compute mAP using torchmetrics
    preds_for_ap = F.one_hot(preds, num_classes=num_classes).float()
    # targets = targets.float()
    # targets_for_ap = F.one_hot(targets, num_classes=num_classes).float()

    # print(preds_for_ap.shape, targets.shape)
    per_class_ap = ap_metric(preds_for_ap.to(device), targets.to(device))  # [num_classes]
    
    # print(counter)
    print(f1)
    mAP = 0.
    for c, ap in enumerate(per_class_ap):
        print(f"Class {c} AP: {ap:.4f}")
        mAP+=ap
    mAP = mAP / 3
    return f1, mAP

# ========= 配置 =========
CHECKPOINT_PATH = './runs/MOT_classification_v1_4/best_model_epoch_3_mAP_0.8570.pth'
DATA_ROOT = '/uu/sci.utah.edu/projects/smartair/Dataset'
TXT_LIST = './meta-files/testlist_e2e_new_1.txt'
JSON_PATH = './datasets/valid_tracks.json'
SAVE_DIR = 'eval_outputs/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========= 模型加载 =========
model = MOT_IVD_v1_4(num_classes=3)
model.load_state_dict(torch.load(CHECKPOINT_PATH)['model_state_dict'])
model.to(DEVICE)
model.eval()

# ========= 数据加载 =========
transform = T.Compose([T.ToTensor()])
dataset = listDataset(DATA_ROOT, TXT_LIST, json_path=JSON_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=1, collate_fn=avivd_collate_fn, shuffle=False, num_workers=8, pin_memory=True)

f1, mAP = validate_one_epoch(model, loader, DEVICE, num_classes=3)

# print(f"F1: {f1:.4f}")
# for c, ap in enumerate(per_class_ap):
#     print(f"Class {c} AP: {ap:.4f}")
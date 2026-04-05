import torch
from losses.mask_ce import masked_cross_entropy_loss
from losses.SupConLoss import SupConLoss
from models.MOT_IVD_v1_7 import MOT_IVD_v1_7
from models.MOT_IVD_v1_9 import MOT_IVD_v1_9
# 注意这里加上 models. 前缀，因为文件在 models 目录下面
from models.MOT_IVD_v1_9_ab_audio_bbox import MOT_IVD_v1_9 as MOT_IVD_v1_9_audio_bbox
from models.MOT_IVD_v1_9_ab_audio_dis import MOT_IVD_v1_9 as MOT_IVD_v1_9_audio_dis
from models.MOT_IVD_v1_9_ab_audio_bbox_dis import MOT_IVD_v1_9 as MOT_IVD_v1_9_audio_bbox_dis
from models.MOT_IVD_v1_9_3c import MOT_IVD_v1_9_3c
# from models.MOT_IVD_v1_9_audio_cl import MOT_IVD_v1_9
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score
from datasets.dataset_mot import listDataset, avivd_collate_fn
import torchvision
import os
import torchvision.transforms as T
from torchmetrics.classification import MulticlassAveragePrecision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
# from torchmetrics.detection.precision_recall_curve import PrecisionRecallCurve
import argparse
from utils import *
import matplotlib.pyplot as plt
import json

# def avivd_collate_fn(batch):
#     bboxes_list = []
#     mask_list = []
#     audio_list = []
#     video_ids = []
#     label_list = []
#     bn_indices = []

#     for i, sample in enumerate(batch):
#         bboxes = sample['bboxes']
#         mask = sample['mask']
#         audio = sample['audio']
#         video_id = sample['video_id']
#         labels = sample['labels']
#         N = bboxes.shape[0]

#         label_list.append(labels)
#         bboxes_list.append(bboxes)
#         mask_list.append(mask)

#         audio_repeated = audio.unsqueeze(0).expand(N, -1, -1, -1)  # [N, 6, 128, 469]
#         audio_list.append(audio_repeated)

#         video_ids.extend([video_id] * N)
#         bn_indices.extend([(i, n) for n in range(N)])

#     bboxes_flat = torch.cat(bboxes_list, dim=0)  # [BN, L, 4]
#     mask_flat = torch.cat(mask_list, dim=0)      # [BN, L]
#     labels_flat = torch.cat(label_list, dim=0)   # [BN, L]
#     audio_flat = torch.cat(audio_list, dim=0)    # [BN, 6, 128, 469]
#     return bboxes_flat, mask_flat, audio_flat, labels_flat, video_ids, bn_indices
# 用 key 映射到不同文件里的 MOT_IVD_v1_9
MODEL_REGISTRY = {
    "audio_bbox_dis_JL": MOT_IVD_v1_9,
    "audio_bbox": MOT_IVD_v1_9_audio_bbox,
    "audio_bbox_dis": MOT_IVD_v1_9_audio_bbox_dis,
    "audio_dis": MOT_IVD_v1_9_audio_dis,
    "3c": MOT_IVD_v1_9_3c
}

def build_model(model_key, *args, **kwargs):
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key {model_key}, available {list(MODEL_REGISTRY.keys())}")
    ModelCls = MODEL_REGISTRY[model_key]
    return ModelCls(*args, **kwargs)

def validate_one_epoch(model, dataloader, loss_n, device, num_classes=3, data_location = 'LDS'):
    # print(data_location)
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    all_predictions, all_groundtruth = [], []

    ap_metric = MulticlassAveragePrecision(num_classes=num_classes, average=None).to(device)  # 每类单独输出
    # Initialize metric
    det_cls_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=[0.5], class_metrics=True, average='macro')
    # pr_metric = PrecisionRecallCurve(num_classes=3)
    count_invalid = 0
    with torch.no_grad():
        for batch in dataloader:
            _, bboxes, mask, audio, labels, video_ids, _, yolo_confs = batch
            # print(bboxes.shape, yolo_confs.shape)
            if bboxes.shape[0] == 0:
                continue
            # clips = clips.to(device)
            # clips = clips.permute(0, 2, 1, 3, 4)
            bboxes = bboxes.to(device)
            mask = mask.to(device)[:,-1:]
            audio = audio.to(device)
            ###########
            # audio = audio[:, [0, 2, 5], :, :]
            ###########

            labels = labels.to(device)[:,-1:]
            # print(bboxes.shape, audio.shape)
            logits, joint_emb_n = model(bboxes, audio)  # [BN, L, C]
            ce_loss = masked_cross_entropy_loss(logits, labels, mask)
            if joint_emb_n is not None:
                scl_loss = loss_n(joint_emb_n.unsqueeze(1), labels[:,-1])
                loss = ce_loss + scl_loss
            else:
                scl_loss = None
                loss = ce_loss


            # print(mask, labels)
            assert (mask==False).sum() == (labels==-1).sum()
            count_invalid+=(labels==-1).sum()


            # logits shape: BN x C
            # print(logits.shape)
            preds = torch.argmax(logits, dim=-1)  # [BN, L]
            # print(preds.shape)
            probs = torch.softmax(logits, dim=-1)        # 或 sigmoid
            cls_scores, cls_ids = probs.max(dim=-1)       # single-label 情形
            # print("B", yolo_confs.shape, cls_scores.shape)
            # print("yolo_confs")

            # final_scores = 0.4 * pct_conf + 0.6 * pct_prob
            final_scores = cls_scores.cpu()
            # final_scores = (yolo_confs * cls_scores.cpu()) / 2          # 若模型也有 objectness
            ground_truths = load_ground_truths(video_ids, data_location)
            # print(bboxes.shape, preds.shape, labels.shape)
            predictions = build_predictions(video_ids, bboxes, final_scores, cls_ids, yolo_confs, cls_scores.cpu())
            # print(predictions)
            # print(ground_truths)

            all_predictions.extend(predictions)
            all_groundtruth.extend(ground_truths)

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())
            all_masks.append(mask.cpu())

    print(count_invalid)

    dump_payload = {
        "predictions": [
            {
                k: (v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v)
                for k, v in pred.items()
            }
            for pred in all_predictions
        ],
        "groundtruth": [
            {
                k: (v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v)
                for k, v in gt.items()
            }
            for gt in all_groundtruth
        ],
    }

    with open("eval_dump_03_05.json", "w") as f:
        json.dump(dump_payload, f)


    ###################### TANet CLS ######################
    preds = torch.cat(all_preds, dim=0).view(-1)
    targets = torch.cat(all_targets, dim=0).view(-1)
    masks = torch.cat(all_masks, dim=0).view(-1)
    preds = preds[masks]
    targets = targets[masks]

    f1 = f1_score(targets, preds, average='macro')
    # print(f"F1: {f1:.4f}")
    one_hot_target = F.one_hot(targets, num_classes=num_classes)
    one_hot_pred = F.one_hot(preds, num_classes=num_classes)
    # print(one_hot_pred.shape, one_hot_target.shape)
    per_class_ap = ap_metric(one_hot_pred.float().to(device), one_hot_target.argmax(dim=1).to(device))  # [num_classes]
    mAP = 0.
    for c, ap in enumerate(per_class_ap):
        print(f"Class {c} AP: {ap:.4f}")
        mAP+=ap
    mAP = mAP / 3
    ###################### TANet CLS ######################

    ###################### overall detection ######################
    # 1) 拼接到全局向量
    # img_ids   = []
    # boxes_all = []; labels_all = []; conf_all = []; prob_all = []
    # offsets = []
    # start = 0

    # for p in all_predictions:
    #     n = p['boxes'].shape[0]
    #     if n == 0:
    #         continue
    #     # img_ids.append(torch.full((p['boxes'].shape[0],), p['image_id'], dtype=torch.long))
    #     boxes_all.append(p['boxes'])
    #     labels_all.append(p['labels'])
    #     conf_all.append(p['yolo_confs'])
    #     prob_all.append(p['cls_scores'])
    #     offsets.append((p, start, start + n))
    #     start += n

    # # img_ids = torch.cat(img_ids)
    # boxes   = torch.cat(boxes_all)
    # labels  = torch.cat(labels_all)
    # conf    = torch.cat(conf_all)    # ∈[0.3,1]（你之前conf_thres>=0.3）
    # prob    = torch.cat(prob_all)    # ∈[0,1]

    # # 2.1 词典式排序分数（先按prob，prob相近时再看conf）——强制重排
    # N = prob.numel()
    # rank_prob = prob.argsort().argsort().float()
    # rank_conf = conf.argsort().argsort().float()
    # new_score = rank_prob * (N + 1) + rank_conf
    # new_score = (new_score - new_score.min()) / (new_score.max() - new_score.min() + 1e-12)

    # （或）2.2 线性缩放后乘（温和重排）
    # conf2 = (conf - 0.30) / 0.70
    # new_score = conf2 * prob

    # 3) 按 offsets 把 new_score 写回各自的字典
    # for p, s, e in offsets:
    #     p['scores'] = new_score[s:e]

    print("########### det/cls mAP ###########")
    det_cls_metric.update(all_predictions, all_groundtruth)
    results = det_cls_metric.compute()
    # Print all metrics
    print("Evaluation Metrics:")
    for name, value in results.items():
        # Handle tensor values
        if hasattr(value, 'item'):
            try:
                val = value.item()
                print(f"{name}: {val:.4f}")
                continue
            except Exception:
                pass
        print(f"{name}: {value}")
    det_cls_metric.reset()
    print("########### det/cls mAP ###########")
    ###################### overall detection ######################
    exit(0)
    return f1, mAP

def train_one_step(model, batch, loss_n, optimizer, device):
    """
    One training step.
    Inputs:
        - batch: output of DataLoader + collate_fn
        - model: your model (BBoxAudioClassifier)
        - optimizer: torch optimizer
    Returns:
        - loss value (float)
    """
    # print(len(batch))
    _, bboxes, mask, audio, labels, _, _, _ = batch  # [BN, L, 4], [BN, L], [BN, 128, 469], [BN, L]
    
    # clips = clips.to(device)
    # clips = clips.permute(0, 2, 1, 3, 4)
    bboxes = bboxes.to(device)
    mask = mask.to(device)
    audio = audio.to(device)
    ###########
    # audio = audio[:, [0, 2, 5], :, :]
    ###########
    labels = labels.to(device)
    # print(labels)
    # print(bboxes, labels.unique())

    model.train()
    optimizer.zero_grad()
    # print(bboxes.shape, audio.shape)
    # print(labels[:,-1], labels.shape)
    logits, joint_emb_n = model(bboxes, audio)  # [BN, L, C]
    # print(labels[:,-1], mask[:,-1])

    ce_loss = masked_cross_entropy_loss(logits, labels, mask)
    if joint_emb_n is not None:
        scl_loss = loss_n(joint_emb_n.unsqueeze(1), labels[:,-1])
        loss = ce_loss + scl_loss
    else:
        scl_loss = None
        loss = ce_loss
    loss.backward()
    optimizer.step()

    return loss.item(), ce_loss.item(), scl_loss.item() if scl_loss is not None else 0.0

def train_loop(model, train_loader, val_loader, loss_n, optimizer, device, num_epochs, checkpoint_dir):
    # batch = next(iter(iter(train_loader)))
    # print(batch[0], batch[3])
    best_map = -float('inf')

    for epoch in range(1, num_epochs+1):
        total_loss = 0    
        for step, batch in enumerate(train_loader):
            loss, ce_loss, scl_loss = train_one_step(model, batch, loss_n, optimizer, device)
            total_loss += loss
            if step % 10 == 0:
                print(f"Epoch {epoch} Step {step} | Loss: {loss:.4f}, ce_loss: {ce_loss:.4f}, scl_loss: {scl_loss:.4f}", flush=True)
        avg_loss = total_loss / len(train_loader)
        f1, mAP = validate_one_epoch(model, val_loader, loss_n, device)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f} | F1: {f1:.4f} | mAP: {mAP:.4f}\n", flush=True)

        # Save if mAP improves or drops
        if mAP > best_map:
            best_map = mAP
            ckpt_path = os.path.join('runs', checkpoint_dir, f"best_model_epoch_{epoch}_mAP_{mAP:.4f}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_map': best_map
            }, ckpt_path)
            print(f"Best checkpoint saved to {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train / Validate MOTG_IVD classifier")
    parser.add_argument(
        "--model_key",
        type=str,
        default="audio_bbox_dis_JL",
        choices=list(MODEL_REGISTRY.keys()),
        help="Which MOT_IVD_v1_9 variant to use",
    )
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to .pth checkpoint for evaluation or finetune")
    parser.add_argument("--val_only", action="store_true",
                        help="Only run validation once and exit")
    parser.add_argument("--job_name", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    checkpoint_dir = args.job_name
    use_cuda = True
    num_workers = 8
    batch_size = 16
    basepath =  '/uu/sci.utah.edu/projects/smartair/Dataset'
    # trainlist = './meta-files/trainlist_e2e_new_1.txt' 
    # testlist = './meta-files/validlist_e2e_new_1.txt' 
    init_width, init_height = 224, 224
    clip_duration = 16
    sup_con_l = SupConLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = build_model(args.model_key)
    model = build_model(args.model_key, num_classes=3).cuda()
    # model = MOT_IVD_v1_9(num_classes=3).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    kwargs = {'num_workers': num_workers, 'pin_memory': True, 'prefetch_factor': 2} if use_cuda else {}
    num_epochs=100
    # train_dataset=listDataset(basepath, trainlist, shape=(init_width, init_height),
    #                    transform=torchvision.transforms.Compose([
    #                        torchvision.transforms.ToTensor(),
    #                    ]), 
    #                    train=True, 
    #                    clip_duration=clip_duration)
    train_dataset = listDataset(
        base_path='/uufs/sci.utah.edu/projects/smartair/Dataset',
        txt_list='./meta-files/trainlist_e2e_new_1.txt',
        # txt_list='./meta-files/MEB/test_split_1/DAMEBlist_train_e2e_new_1.txt',
        json_path='./datasets/train_tracks_yolov11s_corrected_padded.json',
        # json_path='./datasets/new_train.json',
        # bbox_jitter = True,
        load_frames=False,
        transform=T.ToTensor()
    )
    valid_dataset = listDataset(
        base_path='/uufs/sci.utah.edu/projects/smartair/Dataset',
        txt_list='./meta-files/validlist_e2e_new_1.txt',
        # txt_list='./meta-files/MEB/test_split_1/DAMEBlist_test_e2e_new_1_copy.txt',
        json_path='./datasets/valid_tracks_yolov11s_af_03_05_corrected_padded.json',
        # json_path='./datasets/valid_tracks_yolov11s_af_corrected_padded.json',
        # json_path='./datasets/valid_tracks_yolov11s_corrected_padded.json',
        # json_path='./datasets/valid_tracks.json',
        # json_path='./datasets/new_test.json',
        load_frames=False,
        transform=T.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=avivd_collate_fn,
        **kwargs
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=avivd_collate_fn,
        **kwargs
    )

    # ---------- 载入预训练 ----------
    if args.pretrained is not None:
        ckpt = torch.load(args.pretrained, map_location=device)
        state_dict = ckpt if "model_state_dict" not in ckpt else ckpt["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        print(f"=> Loaded weights from {args.pretrained}")
        # 如果继续训练则恢复优化器
        if not args.val_only and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("=> Optimizer state restored")

    # ---------- 只验证 ----------
    if args.val_only:
        f1, mAP = validate_one_epoch(model, valid_loader, sup_con_l, device, data_location = 'LDS')
        print(f"[Validate-Only] F1: {f1:.4f} | mAP: {mAP:.4f}")

        print("##################################################")
        test_dataset = listDataset(
            base_path='/uu/sci.utah.edu/projects/smartair/Dataset',
            txt_list='./meta-files/testlist_e2e_new_2.txt',
            json_path='./datasets/test_tracks_yolov11s_2_corrected_padded.json',
            # json_path='./datasets/test_tracks_2_corrected_padded.json',
            load_frames=False,
            transform=T.ToTensor()
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=avivd_collate_fn,
            **kwargs
        )
        f1, mAP = validate_one_epoch(model, test_loader, sup_con_l, device, data_location = 'LDS')
        print(f"[Test-Only] F1: {f1:.4f} | mAP: {mAP:.4f}")
        exit(0)

    train_loop(model, train_loader, valid_loader, sup_con_l, optimizer, device, num_epochs, checkpoint_dir)

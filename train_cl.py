import torch
from losses.mask_ce import masked_cross_entropy_loss
from losses.SupConLoss import SupConLoss
from models.MOT_IVD_v1_7 import MOT_IVD_v1_7
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score
from datasets.dataset_mot import listDataset, avivd_collate_fn
import torchvision
import os
import torchvision.transforms as T
from torchmetrics.classification import MulticlassAveragePrecision

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


def validate_one_epoch(model, dataloader, loss_n, device, num_classes=3):

    model.eval()
    all_preds, all_targets, all_masks = [], [], []

    ap_metric = MulticlassAveragePrecision(num_classes=num_classes, average=None).to(device)  # 每类单独输出

    with torch.no_grad():
        for batch in dataloader:
            _, bboxes, mask, audio, labels, _, _ = batch
            if bboxes.shape[0] == 0:
                continue
            # clips = clips.to(device)
            # clips = clips.permute(0, 2, 1, 3, 4)
            bboxes = bboxes.to(device)
            mask = mask.to(device)[:,-1:]
            audio = audio.to(device)
            labels = labels.to(device)[:,-1:]
            # print(clips.shape, audio.shape)
            logits, joint_emb_n = model(bboxes, audio)  # [BN, L, C]
            ce_loss = masked_cross_entropy_loss(logits, labels, mask)
            scl_loss = loss_n(joint_emb_n.unsqueeze(1), labels[:,-1])
            loss = ce_loss + scl_loss
            # print(logits.shape)
            preds = torch.argmax(logits, dim=-1)  # [BN, L]
            # print(labels.shape, mask.shape)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())
            all_masks.append(mask.cpu())

    preds = torch.cat(all_preds, dim=0).view(-1)
    targets = torch.cat(all_targets, dim=0).view(-1)
    masks = torch.cat(all_masks, dim=0).view(-1)
    # print(preds.shape, masks.shape)
    preds = preds[masks]
    targets = targets[masks]

    f1 = f1_score(targets, preds, average='macro')
    # print(f"F1: {f1:.4f}")
    one_hot_target = F.one_hot(targets, num_classes=num_classes)
    one_hot_pred = F.one_hot(preds, num_classes=num_classes)
    # print(one_hot_pred.shape, one_hot_target.shape)
    # map_score = average_precision_score(one_hot_target.numpy(), one_hot_pred.numpy(), average='macro')
    per_class_ap = ap_metric(one_hot_pred.float().to(device), one_hot_target.argmax(dim=1).to(device))  # [num_classes]
    mAP = 0.
    for c, ap in enumerate(per_class_ap):
        print(f"Class {c} AP: {ap:.4f}")
        mAP+=ap
    mAP = mAP / 3
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
    _, bboxes, mask, audio, labels, _, _ = batch  # [BN, L, 4], [BN, L], [BN, 128, 469], [BN, L]
    
    # clips = clips.to(device)
    # clips = clips.permute(0, 2, 1, 3, 4)
    bboxes = bboxes.to(device)
    mask = mask.to(device)
    audio = audio.to(device)
    labels = labels.to(device)
    # print(audio.shape)
    # print(bboxes, labels.unique())

    model.train()
    optimizer.zero_grad()

    # print(labels[:,-1], labels.shape)
    logits, joint_emb_n = model(bboxes, audio)  # [BN, L, C]
    # print(labels[:,-1], mask[:,-1])

    ce_loss = masked_cross_entropy_loss(logits, labels, mask)
    scl_loss = loss_n(joint_emb_n.unsqueeze(1), labels[:,-1])
    loss = ce_loss + scl_loss
    loss.backward()
    optimizer.step()

    return loss.item(), ce_loss.item(), scl_loss.item()

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


if __name__ == "__main__":
    checkpoint_dir = 'MOT_classification_cl_v1_7_yolo'
    use_cuda = True
    num_workers = 8
    batch_size = 16
    basepath =  '/uu/sci.utah.edu/projects/smartair/Dataset'
    trainlist = './meta-files/trainlist_e2e_new_1.txt' 
    testlist = './meta-files/testlist_e2e_new_1.txt' 
    init_width, init_height = 224, 224
    clip_duration = 16
    sup_con_l = SupConLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MOT_IVD_v1_7(num_classes=3).cuda()
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
        base_path='/uu/sci.utah.edu/projects/smartair/Dataset',
        txt_list='./meta-files/trainlist_e2e_new_1.txt',
        json_path='./datasets/train_tracks_yolo_1_corrected.json',
        load_frames=False,
        transform=T.ToTensor()
    )
    valid_dataset = listDataset(
        base_path='/uu/sci.utah.edu/projects/smartair/Dataset',
        txt_list='./meta-files/testlist_e2e_new_1.txt',
        json_path='./datasets/valid_tracks_yolo_1_corrected.json',
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
        batch_size=batch_size,
        shuffle=False,
        collate_fn=avivd_collate_fn,
        **kwargs
    )
    train_loop(model, train_loader, valid_loader, sup_con_l, optimizer, device, num_epochs, checkpoint_dir)

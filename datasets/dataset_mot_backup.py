import torch
import random
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import time

class listDataset(Dataset):
    def __init__(self, base_path, txt_list, transform=None, shape=(224, 224), clip_duration=16, train=False):
        
        with open(txt_list, 'r') as f:
            self.lines = f.readlines()

        self.base_path = base_path
        self.transform = transform
        self.shape = shape
        self.clip_duration = clip_duration
        self.train = train

    def __len__(self):
        return len(self.lines)
    
    # === IoU-based tracking ===
    def box_iou(self, box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = boxAArea + boxBArea - interArea
        return interArea / union if union > 0 else 0

    def __getitem__(self, index):
        imgpath = self.lines[index].strip()
        im_split = imgpath.replace("\\", "/").split('/')
        im_ind = int(im_split[-1][:5])
        img_folder = imgpath[:-10]  # remove /xxxxx.jpg
        label_base = os.path.join('/'.join(im_split[:-2]), 'labels/av')
        audio_path = os.path.join('/'.join(im_split[:-2]), 'audio', f"{im_ind:05d}.npy")

        # === Load video clip ===
        clip = []
        frame_indices = []
        d = 4
        for i in reversed(range(self.clip_duration)):
            i_temp = im_ind - i * d
            max_num = len(os.listdir(img_folder))
            i_temp = (i_temp + max_num) % max_num
            frame_indices.append(i_temp)
            img_fp = os.path.join(img_folder, f"{i_temp:05d}.jpg")
            clip.append(Image.open(img_fp).convert('RGB'))

        if self.transform:
            clip = [self.transform(img.resize(self.shape)) for img in clip]

        v_clip = torch.cat(clip, dim=0).view((self.clip_duration, -1) + self.shape).permute(1, 0, 2, 3)

        # === Load audio ===
        audio = torch.from_numpy(np.load(audio_path)).float()  # [6, 128, 469]

        # === Load bbox + label ===
        L = self.clip_duration
        frame_bboxes = [[] for _ in range(L)]
        frame_labels = [[] for _ in range(L)]
        for i, frame_id in enumerate(frame_indices):
            label_path = os.path.join(label_base, f"{frame_id:05d}_g.txt")
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                raw = np.loadtxt(label_path).reshape(-1, 5)
                for box in raw:
                    label = int(box[0])
                    coords = box[1:].tolist()
                    frame_bboxes[i].append(coords)
                    frame_labels[i].append(label - 1)

        tracks = []
        iou_threshold = 0.5
        for t in range(L):
            boxes = frame_bboxes[t]
            labels = frame_labels[t]
            assigned = [False] * len(boxes)
            for track in tracks:
                prev_box = track['boxes'][t-1] if t > 0 else None
                best_iou, best_idx = 0, -1
                if prev_box is not None:
                    for i, box in enumerate(boxes):
                        if not assigned[i]:
                            iou = self.box_iou(prev_box, box)
                            if iou > best_iou:
                                best_iou, best_idx = iou, i
                if best_iou > iou_threshold:
                    track['boxes'].append(boxes[best_idx])
                    track['labels'].append(labels[best_idx])
                    assigned[best_idx] = True
                else:
                    track['boxes'].append(None)
                    track['labels'].append(-1)
            for i, box in enumerate(boxes):
                if not assigned[i]:
                    new_track = {'boxes': [None]*t + [box], 'labels': [-1]*t + [labels[i]]}
                    tracks.append(new_track)

        N = len(tracks)
        bboxes = torch.zeros((N, L, 4), dtype=torch.float32)
        mask = torch.zeros((N, L), dtype=torch.bool)
        labels_out = torch.full((N, L), -1, dtype=torch.long)
        for n, track in enumerate(tracks):
            for l in range(L):
                if track['boxes'][l] is not None:
                    bboxes[n, l] = torch.tensor(track['boxes'][l], dtype=torch.float32)
                    mask[n, l] = True
                    labels_out[n, l] = track['labels'][l]

        return {
            'bboxes': bboxes,     # [N, L, 4]
            'mask': mask,         # [N, L]
            'audio': audio,       # [6, 128, 469]
            'video_id': imgpath,
            'labels': labels_out  # [N, L] with values in {0,1,2} or -1
        }


def avivd_collate_fn(batch):
    """
    batch: list of dicts with keys: 'bboxes': [N, L, 4], 'mask': [N, L], 'audio': [6, 128, 469], 'video_id': str
    Returns:
        - bboxes_flat: [BN, L, 4]
        - mask_flat:   [BN, L]
        - audio_flat:  [BN, 128, 469]
        - meta: dict with video_id list and original B,N sizes if needed
    """
    bboxes_list = []
    mask_list = []
    audio_list = []
    video_ids = []
    label_list = []
    bn_indices = []  # optional

    for i, sample in enumerate(batch):
        bboxes = sample['bboxes']     # [N, L, 4]
        mask = sample['mask']         # [N, L]
        audio = sample['audio']       # [6, 128, 469]
        # print("audio", audio.shape)
        video_id = sample['video_id']
        labels = sample['labels']
        N = bboxes.shape[0]

        label_list.append(labels)
        bboxes_list.append(bboxes)    # [N, L, 4]
        mask_list.append(mask)        # [N, L]

        audio_repeated = audio.unsqueeze(0).expand(N, -1, -1, -1)  # [N, 6, 128, 469]
        # audio_mean = audio_repeated.mean(dim=1)                   # [N, 128, 469] → average 6 channels
        audio_list.append(audio_repeated)

        video_ids.extend([video_id] * N)
        bn_indices.extend([(i, n) for n in range(N)])  # optional

    bboxes_flat = torch.cat(bboxes_list, dim=0)  # [BN, L, 4]
    mask_flat = torch.cat(mask_list, dim=0)      # [BN, L]
    labels_flat = torch.cat(label_list, dim=0)   # [BN, L]
    audio_flat = torch.cat(audio_list, dim=0)    # [BN, 128, 469]
    # print(bboxes_flat.shape, mask_flat.shape, audio_flat.shape)
    return bboxes_flat, mask_flat, audio_flat, labels_flat, video_ids, bn_indices


        
if __name__ == "__main__":
    basepath='../Dataset'
    trainlist='./trainlist_e2e_new_1.txt'
    dataset_use='car'
    init_width=224
    init_height=224
    batch_size=16
    clip_duration=16
    num_workers=8
    root = 'tmp'
    use_cuda=True
    kwargs = {'num_workers': num_workers, 'pin_memory': True, 'prefetch_factor': 2} if use_cuda else {}
    e2e_dataset=listDataset(basepath, trainlist, shape=(init_width, init_height),
                       transform=torchvision.transforms.Compose([
                        #    torchvision.transforms.Resize((224,224)),
                           torchvision.transforms.ToTensor(),
                       ]), 
                       train=False, 
                    #    seen=cur_model.seen,
                    #    batch_size=batch_size,
                       clip_duration=clip_duration)
    # train_loader = torch.utils.data.DataLoader(e2e_dataset,
    #     batch_size=batch_size, shuffle=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(
        e2e_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=avivd_collate_fn,
        **kwargs
    )
    print(len(train_loader.dataset))
    t1=time.time()
    for batch_idx, batch in enumerate(train_loader):
        bboxes_flat, mask_flat, audio_flat, labels_list, _, _ = batch
        print(bboxes_flat.shape, mask_flat.shape, audio_flat.shape, labels_list.shape)
        t2=time.time()
        print("one batch duration", t2 - t1)
        t1=t2
    # print((batch_idx+1)*batch_size/(t2-t1), t2-t1)
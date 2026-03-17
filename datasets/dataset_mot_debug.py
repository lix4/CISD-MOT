import os, json
import torch, numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# ========= IoU 工具 =========
def iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor):
    """
    boxes_a: [Na,4] boxes_b: [Nb,4] → IoU [Na,Nb]
    坐标格式 (x1,y1,x2,y2)
    """
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.empty((boxes_a.size(0), boxes_b.size(0)), device=boxes_a.device)

    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])      # 左上
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])      # 右下
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) *
              (boxes_a[:, 3] - boxes_a[:, 1]))[:, None]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) *
              (boxes_b[:, 3] - boxes_b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-6)


def greedy_pair(iou: torch.Tensor, thr: float = 0.5):
    """
    简单贪心一对一匹配；返回 (gt_idx, yolo_idx) 列表，只保留 IoU ≥ thr
    """
    pairs = []
    flat = torch.argsort(iou.reshape(-1), descending=True)
    n_gt, n_y = iou.shape
    used_gt = torch.zeros(n_gt, dtype=torch.bool)
    used_y = torch.zeros(n_y, dtype=torch.bool)
    for f in flat:
        g, y = divmod(f.item(), n_y)
        if used_gt[g] or used_y[y] or iou[g, y] < thr:
            continue
        pairs.append((g, y))
        used_gt[g] = True
        used_y[y] = True
    return pairs


# ========= 数据集 =========
class listDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        txt_list: str,
        json_path_gt: str,
        json_path_yolo: str,
        load_frames: bool = True,
        transform=None,
        shape=(224, 224),
        clip_duration: int = 16,
        train: bool = False,
    ):
        with open(txt_list, "r") as f:
            self.lines = f.readlines()

        with open(json_path_gt, "r") as f:
            self.track_json_gt = json.load(f)
        with open(json_path_yolo, "r") as f:
            self.track_json_yolo = json.load(f)

        self.base_path = base_path
        self.transform = transform
        self.shape = shape
        self.clip_duration = clip_duration
        self.train = train
        self.load_frames = load_frames

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        # ---------- 基础路径解析 ----------
        imgpath = self.lines[index].strip()
        im_split = imgpath.replace("\\", "/").split("/")
        im_ind = int(im_split[-1][:5])                          # 帧号
        img_folder = imgpath[:-10]                              # 去掉 '/xxxxx.jpg'
        audio_path = os.path.join("/".join(im_split[:-2]), "audio", f"{im_ind:05d}.npy")
        video_id = f"{im_split[-4]}-{im_split[-3]}-{im_split[-1]}"

        track_clips = None  
        # ---------- 载入 16 帧剪辑 ----------
        if self.load_frames:
            clip, d = [], 4                                     # 步长 d = 4
            max_num = len(os.listdir(img_folder))
            for i in reversed(range(self.clip_duration)):
                i_temp = (im_ind - i * d + max_num) % max_num
                fp = os.path.join(img_folder, f"{i_temp:05d}.jpg")
                img = Image.open(fp).convert("RGB")
                clip.append(self.transform(img) if self.transform else TF.to_tensor(img))
            v_clip = torch.cat(clip, 0).view((self.clip_duration, -1, 240, 320)).permute(1, 0, 2, 3)

        # ---------- 载入音频 ----------
        audio = torch.from_numpy(np.load(audio_path)).float()   # [6,128,469]

        # ---------- 读取两份 JSON ----------
        data_gt = self.track_json_gt.get(imgpath, {})
        data_yolo = self.track_json_yolo.get(imgpath, {})

        boxes_gt, boxes_yolo, labels, masks, clips = [], [], [], [], []

        tids = sorted(set(data_gt.keys()) | set(data_yolo.keys()))
        for tid in tids:
            gt = data_gt.get(
                tid,
                {"boxes": [[0, 0, 0, 0]] * self.clip_duration,
                 "labels": [-1] * self.clip_duration,
                 "mask": [0] * self.clip_duration},
            )
            yl = data_yolo.get(
                tid,
                {"boxes": [[0, 0, 0, 0]] * self.clip_duration},
            )

            b_gt = [[0, 0, 0, 0] if v is None else v for v in gt["boxes"]]
            b_yl = [[0, 0, 0, 0] if v is None else v for v in yl["boxes"]]

            boxes_gt.append(b_gt)
            boxes_yolo.append(b_yl)
            labels.append([-1 if m == 0 else v for v, m in zip(gt["labels"], gt["mask"])])
            masks.append(gt["mask"])

            if self.load_frames:
                track = []
                for t, (x1, y1, x2, y2) in enumerate(b_gt):      # 用 GT 框裁剪
                    if x2 <= x1 or y2 <= y1:
                        crop = torch.zeros((3, *self.shape), dtype=torch.float32)
                    else:
                        crop = v_clip[:, t, y1:y2, x1:x2]
                        crop = TF.resize(crop, self.shape, antialias=True)
                    track.append(crop)
                clips.append(torch.stack(track, 0))

        # ---------- 列表 → 张量 ----------
        bboxes_gt = torch.tensor(boxes_gt, dtype=torch.float32)      # [N,L,4]
        bboxes_yolo = torch.tensor(boxes_yolo, dtype=torch.float32)  # [N,L,4]
        labels_out = torch.tensor(labels, dtype=torch.long)          # [N,L]
        mask_out = torch.tensor(masks, dtype=torch.bool)             # [N,L]
        if self.load_frames:
            track_clips = torch.stack(clips, 0) if clips else torch.zeros((0, self.clip_duration, 3, *self.shape))

        # ---------- IoU 匹配+对齐 ----------
        last_gt = bboxes_gt[:, -1, :] if len(bboxes_gt) else torch.zeros((0, 4))
        last_yolo = bboxes_yolo[:, -1, :] if len(bboxes_yolo) else torch.zeros((0, 4))
        pairs = greedy_pair(iou_matrix(last_gt, last_yolo), thr=0.5)

        if pairs:
            idx_gt, idx_y = zip(*pairs)
            idx_gt = torch.tensor(idx_gt)
            idx_y = torch.tensor(idx_y)

            bboxes_gt = bboxes_gt[idx_gt]
            bboxes_yolo = bboxes_yolo[idx_y]
            labels_out = labels_out[idx_gt]
            mask_out = mask_out[idx_gt]
            if self.load_frames:
                track_clips = track_clips[idx_gt]
        else:  # 没匹配到，返回空张量
            Nm = 0
            bboxes_gt = torch.zeros((0, self.clip_duration, 4), dtype=torch.float32)
            bboxes_yolo = torch.zeros_like(bboxes_gt)
            labels_out = torch.full((0, self.clip_duration), -1, dtype=torch.long)
            mask_out = torch.zeros((0, self.clip_duration), dtype=torch.bool)
            if self.load_frames:
                track_clips = torch.zeros((0, self.clip_duration, 3, *self.shape), dtype=torch.float32)

        # ---------- 返回 ----------
        return {
            "clips": track_clips,          # [Nm,L,3,H,W] 或 None
            "bboxes_gt": bboxes_gt,        # [Nm,L,4]
            "bboxes_yolo": bboxes_yolo,    # [Nm,L,4]
            "labels": labels_out,          # [Nm,L]
            "mask": mask_out,              # [Nm,L]
            "audio": audio,                # [6,128,469]
            "video_id": video_id,
        }

def avivd_collate_fn(batch):
    track_list, b_gt_list, b_yolo_list = [], [], []
    lab_list, msk_list, aud_list = [], [], []
    vids, bn_idx = [], []

    for i, s in enumerate(batch):
        N = s['bboxes_gt'].shape[0]

        track_list.append(s['clips'])
        b_gt_list  .append(s['bboxes_gt'])
        b_yolo_list.append(s['bboxes_yolo'])
        lab_list  .append(s['labels'])
        msk_list  .append(s['mask'])

        aud_list.append(s['audio'].unsqueeze(0).expand(N, -1, -1, -1))
        vids.extend([s['video_id']]*max(N,1))
        bn_idx.extend([(i,n) for n in range(N)])

    clips_flat  = torch.cat(track_list, 0) if track_list[0] is not None else None
    b_gt_flat   = torch.cat(b_gt_list, 0)
    b_y_flat    = torch.cat(b_yolo_list,0)
    lab_flat    = torch.cat(lab_list, 0)
    msk_flat    = torch.cat(msk_list, 0)
    aud_flat    = torch.cat(aud_list, 0)

    return clips_flat, b_gt_flat, b_y_flat, msk_flat, aud_flat, lab_flat, vids, bn_idx




if __name__ == '__main__':
    # dataset = listDataset(
    #     base_path='/uu/sci.utah.edu/projects/smartair/Dataset',
    #     txt_list='../meta-files/trainlist_e2e_new_1.txt',
    #     json_path='train_tracks.json',
    #     transform=T.ToTensor()
    # )
    valid_dataset = listDataset(
        base_path='/uu/sci.utah.edu/projects/smartair/Dataset',
        txt_list='./meta-files/trainlist_e2e_new_1.txt',
        json_path='./datasets/train_tracks.json',
        transform=T.ToTensor()
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=avivd_collate_fn,
        # **kwargs
    )
    # loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=avivd_collate_fn)

    for batch in valid_loader:
        clips, bboxes, mask, audio, labels, video_ids, bn_indices = batch
        print((labels==-1).any())
        print(video_ids, bboxes.shape)
        # print(clips.permute(0, 2, 1, 3, 4).shape, bboxes.shape, labels.shape, audio.shape)
        # break

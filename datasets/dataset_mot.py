import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import torchvision.transforms.functional as TF

class listDataset(Dataset):
    def __init__(self, base_path, txt_list, json_path, load_frames=True, transform=None, shape=(224, 224), clip_duration=16, train=False):
        with open(txt_list, 'r') as f:
            self.lines = f.readlines()

        with open(json_path, 'r') as f:
            self.track_json = json.load(f)

        self.base_path = base_path
        self.transform = transform
        self.shape = shape
        self.clip_duration = clip_duration
        self.train = train
        self.load_frames = load_frames

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        imgpath = self.lines[index].strip()
        im_split = imgpath.replace("\\", "/").split('/')
        im_ind = int(im_split[-1][:5])
        img_folder = imgpath[:-10]  # remove /xxxxx.jpg
        audio_path = os.path.join('/'.join(im_split[:-2]), 'audio', f"{im_ind:05d}.npy")

        track_data = self.track_json.get(imgpath, {})
        # print(im_split)
        video_id = f"{im_split[-4]}-{im_split[-3]}-{im_split[-1]}"

        # === Load video clip ===
        clip = []
        d = 4
        frame_indices = []
        max_num = len(os.listdir(img_folder))
        for i in reversed(range(self.clip_duration)):
            i_temp = (im_ind - i * d + max_num) % max_num
            frame_indices.append(i_temp)
            img_fp = os.path.join(img_folder, f"{i_temp:05d}.jpg")
            if self.load_frames:
                clip.append(Image.open(img_fp).convert('RGB'))

        if self.load_frames:
            if self.transform:
                clip = [self.transform(img) for img in clip]

        if self.load_frames:
            v_clip = torch.cat(clip, dim=0).view((self.clip_duration, -1) + (240, 320)).permute(1, 0, 2, 3)
        # print(v_clip.shape)
        # === Load audio ===
        audio = torch.from_numpy(np.load(audio_path)).float()  # [6, 128, 469]

        # === Load labels from JSON ===
        boxes = []
        labels = []
        mask = []
        track_clips = []
        # print(track_data.keys())
        for tid in sorted(track_data.keys()):
            track = track_data[tid]
            b = [[0, 0, 0, 0] if v is None else v for v in track['boxes']]
            l = [-1 if m == 0 else v for v, m in zip(track['labels'], track['mask'])]
            m = track['mask']
            
            track_clip = []
            if self.load_frames:
                for t in range(self.clip_duration):
                    x1, y1, x2, y2 = map(int, b[t])
                    # print(x1,y1,x2,y2)
                    if x2 <= x1 or y2 <= y1:
                        # Empty box fallback
                        crop = torch.zeros((3, self.shape[0], self.shape[1]), dtype=torch.float32)
                    else:
                        crop = v_clip[:, t, y1:y2, x1:x2]  # [C, h, w]
                        crop = TF.resize(crop, (self.shape[0], self.shape[1]), antialias=True)  # Resize to [3,112,112]
                    # print(crop.shape)
                    track_clip.append(crop)

                track_clip = torch.stack(track_clip, dim=0)
                track_clips.append(track_clip)
            boxes.append(b)
            labels.append(l)
            mask.append(m)

        # print(len(track_clips))
        bboxes = torch.tensor(boxes, dtype=torch.float32)   # [N, L, 4]
        # track_clips = torch.stack(track_clips, dim=0)
        # print(track_clips.shape)
        labels_out = torch.tensor(labels, dtype=torch.long) # [N, L]
        mask = torch.tensor(mask, dtype=torch.bool)         # [N, L]

        if len(boxes) == 0:
            if self.load_frames:
                track_clips = torch.zeros((0, self.clip_duration, 3, self.shape[0], self.shape[1]), dtype=torch.float32)
            bboxes = torch.zeros((0, self.clip_duration, 4), dtype=torch.float32)
            labels_out = torch.full((0, self.clip_duration), -1, dtype=torch.long)
            mask = torch.zeros((0, self.clip_duration), dtype=torch.bool)
        else:
            if self.load_frames:
                track_clips = torch.stack(track_clips, dim=0)
            bboxes = torch.tensor(boxes, dtype=torch.float32)   # [N, L, 4]
            labels_out = torch.tensor(labels, dtype=torch.long) # [N, L]
            mask = mask.to(torch.bool)         # [N, L]
        # print('bbox', bboxes.shape)
        # print(video_id)
        return {
            'clips': track_clips, 
            'bboxes': bboxes,
            'labels': labels_out,
            'mask': mask,
            'audio': audio,
            'video_id': video_id
        }


def avivd_collate_fn(batch):
    bboxes_list = []
    track_clips_list = []
    mask_list = []
    audio_list = []
    video_ids = []
    label_list = []
    bn_indices = []

    for i, sample in enumerate(batch):
        bboxes = sample['bboxes']
        track_clips = sample['clips']
        mask = sample['mask']
        audio = sample['audio']
        video_id = sample['video_id']
        labels = sample['labels']
        N = bboxes.shape[0]
        # print(bboxes.shape[0])

        label_list.append(labels)
        bboxes_list.append(bboxes)
        mask_list.append(mask)
        track_clips_list.append(track_clips)

        audio_repeated = audio.unsqueeze(0).expand(N, -1, -1, -1)  # [N, 6, 128, 469]
        audio_list.append(audio_repeated)

        if N == 0:
            video_ids.extend([video_id] * 1)
        else:
            video_ids.extend([video_id] * N)
        bn_indices.extend([(i, n) for n in range(N)])

    # print(len(track_clips_list))
    track_clips_flat = None
    if not isinstance(track_clips_list, list):
        track_clips_flat = torch.cat(track_clips_list, dim=0)
    bboxes_flat = torch.cat(bboxes_list, dim=0)  # [BN, L, 4]
    mask_flat = torch.cat(mask_list, dim=0)      # [BN, L]
    labels_flat = torch.cat(label_list, dim=0)   # [BN, L]
    audio_flat = torch.cat(audio_list, dim=0)    # [BN, 6, 128, 469]
    return track_clips_flat, bboxes_flat, mask_flat, audio_flat, labels_flat, video_ids, bn_indices


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

import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets.dataset_mot import listDataset, avivd_collate_fn

def save_samples_with_audio_video(track_clips_flat, labels_flat, mask_flat, audio_flat, video_ids, output_dir="debug_videos", fps=5):
    os.makedirs(output_dir, exist_ok=True)
    BN, L, C, H, W = track_clips_flat.shape  # [BN, L, 3, 112, 112]
    _, _, FREQ, TIME = audio_flat.shape     # [BN, 6, 128, 469]

    for i in range(BN):
        frames = track_clips_flat[i]         # [L, 3, 112, 112]
        labels = labels_flat[i]              # [L]
        mask = mask_flat[i]                  # [L]
        audio = audio_flat[i]                # [6, 128, 469]
        video_id = video_ids[i].replace('/', '_')
        out_path = os.path.join(output_dir, f"{i:03d}_{video_id}.avi")

        # Use OpenCV VideoWriter (video will be H_total x W_total)
        canvas_h = 400  # total canvas height (video + spectrograms)
        canvas_w = max(112, 469)  # match audio width
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (canvas_w, canvas_h))

        # === Pre-render audio spectrograms ===
        fig, axes = plt.subplots(6, 1, figsize=(canvas_w/100, 6), dpi=100)
        for ch in range(6):
            axes[ch].imshow(audio[ch].cpu().numpy(), aspect='auto', origin='lower', cmap='magma')
            axes[ch].axis('off')
        plt.tight_layout(pad=0.1)
        canvas = FigureCanvas(fig)
        canvas.draw()
        audio_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        audio_img = audio_img.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        audio_img = cv2.resize(audio_img, (canvas_w, canvas_h - 112))  # Resize to fit under video

        for t in range(L):
            frame = frames[t]  # [3, H, W]
            img_np = TF.to_pil_image(frame.cpu()).convert("RGB")
            img_np = cv2.cvtColor(np.array(img_np), cv2.COLOR_RGB2BGR)
            img_np = cv2.resize(img_np, (canvas_w, 112))

            label = labels[t].item()
            msk = int(mask[t].item())
            text = f"L: {label}, M: {msk}"
            cv2.putText(img_np, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

            # Concatenate vertically
            combined = np.vstack([img_np, audio_img])
            writer.write(combined)

        writer.release()
        print(f"Saved: {out_path}")


dataset = listDataset(
    base_path='/uu/sci.utah.edu/projects/smartair/Dataset',
    txt_list='./meta-files/trainlist_e2e_new_1.txt',
    json_path='./datasets/train_tracks.json',
    transform=T.ToTensor()
)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=avivd_collate_fn)

for batch in loader:
    clips, bboxes, mask, audio, labels, video_ids, bn_indices = batch
    save_samples_with_audio_video(clips, labels, mask, audio, video_ids)
    # break  # 仅保存一批调试看效果


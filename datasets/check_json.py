import json
import numpy as np

with open('train_tracks_yolo_1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

all_track_num = 0
short_track_num = 0        # “single-frame” or very short tracks

for video_tracks in data.values():
    for track_info in video_tracks.values():
        all_track_num += 1

        mask = np.asarray(track_info["mask"], dtype=bool)
        present_frames = np.sum(mask)         # frames where the box exists

        if present_frames < 5:                # fewer than 5 visible frames
            short_track_num += 1

ratio = short_track_num / all_track_num if all_track_num else 0
print(f"{short_track_num = } / {all_track_num = }  →  ratio = {ratio:.4f}")

import torchvision.transforms as T
import torch
import json
import os

json_path = './datasets/train_tracks.json'

with open(json_path, 'r') as f:
    data = json.load(f)

existing_set = set()
for key, _ in data.items():
    base = "/".join(key.split("/")[:-1])
    index = int(key.split("/")[-1][:-4])
    print(base, key, index)
    sample_ids = [index - 4*i for i in range(16)]
    sample_ids = sorted(sample_ids)

    for sample_id in sample_ids:
        img_path = os.path.join(base, str(sample_id).zfill(5)+'.jpg')
        if img_path not in existing_set:
            existing_set.add(img_path)

output_txt = './od_af_train.txt'

with open(output_txt, 'w') as f:
    for path in sorted(existing_set):
        f.write(path + '\n')
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import torchvision
import torch
from torch.utils.data import DataLoader

def custom_collate(batch):
    pic_paths, images, targets = zip(*batch)
    images = torch.stack(images, 0)

    all_labels = []
    for image_idx, boxes in enumerate(targets):
        boxes=boxes[0]
        # print(boxes.shape)
        if boxes.numel() == 0:
            continue
        # 假设你已经得到 [N, 4] 的 box（x, y, w, h）格式：
        boxes = boxes[:, 1:]  # 去掉 frame_idx
        image_idx_col = torch.full((boxes.shape[0], 1), image_idx, dtype=torch.float32)
        class_col = torch.zeros((boxes.shape[0], 1), dtype=torch.float32)  # 全部设为 0
        # print(image_idx_col.shape, class_col.shape, boxes.shape)
        labels = torch.cat([image_idx_col, class_col, boxes], dim=1)  # → [N, 6]
        all_labels.append(labels)

    if all_labels:
        labels = torch.cat(all_labels, dim=0)  # [N_total, 6]
    else:
        labels = torch.zeros((0, 6), dtype=torch.float32)

    return pic_paths, images, {'batch_idx': labels[:,0], 
                               'cls': labels[:,1],
                               'bboxes': labels[:, 2:]
                               }

class CustomYOLODataset(Dataset):
    def __init__(self, list_file, transform=None):
        # 从txt文件中读取每行图片的路径
        # with open(list_file, 'r') as f:
        #     self.image_paths = [line.strip() for line in f if line.strip()]
        with open(list_file, 'r') as file:
            self.image_paths = file.readlines()
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index].strip()
        # 构造标注文件路径，假设图片路径中包含 "images"，对应标注文件在 "labels" 文件夹中
        ann_path = image_path.replace("rgb-images", "labels").rsplit('.', 1)[0] + ".txt"
        # print(image_path)
        # 加载图片并获取尺寸（cv2读取为BGR，需转换为RGB）
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        boxes = []
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # 如果标注为 x1, y1, x2, y2 格式
                    if len(parts) == 4:
                        x1, y1, x2, y2 = map(float, parts)
                        # 计算中心点和宽高（归一化）
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        box_width = (x2 - x1) / w
                        box_height = (y2 - y1) / h
                        # 如果数据中没有类别信息，默认设置类别为0（单类别任务），多类别时请根据实际情况调整
                        boxes.append([x_center, y_center, box_width, box_height])
                    # 如果标注文件中已有类别信息（如：class x1 y1 x2 y2），则可以做如下处理
                    elif len(parts) == 5:
                        clss, x1, y1, x2, y2 = parts
                        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        box_width = (x2 - x1) / w
                        box_height = (y2 - y1) / h
                        boxes.append([x_center, y_center, box_width, box_height])
        boxes = np.array(boxes) if boxes else np.zeros((0, 4))
        # print(boxes)
        # resize after formatting bbox
        image = cv2.resize(image, (224, 224)) 
        if self.transform:
            # sample = self.transform(
            #     image=image,
            #     bboxes=boxes
            # )
            # image_path = self.transform(image_path)
            image = self.transform(image)
            boxes = self.transform(boxes)
        # print("boxes", boxes.shape)
        # image = sample['image']
        # boxes = sample['bboxes']
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)  # (Ni,4)
        # print(boxes.shape)
        # print(image.shape, boxes.shape)
        # print(boxes.shape)
        return image_path, image, boxes

if __name__ == '__main__':
    train_ds = CustomYOLODataset(
        "./meta-files/trainlist_e2e_new_1.txt",
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    )
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=custom_collate)
    all_frames = set()
    print(len(train_dl))
    for batch_idx, (pic_path, images, targets) in enumerate(train_dl):
        print(pic_path)
        # pic_path = pic_path[0]
        # im_split = pic_path.replace("\\", "/").split('/')
        # im_ind = int(im_split[-1][:5])
        # img_folder = pic_path[:-10]  # remove /xxxxx.jpg
        # print(img_folder)
        # clip = []
        # d = 4
        # frame_indices = []
        # max_num = len(os.listdir(img_folder))
        # for i in reversed(range(16)):
        #     i_temp = (im_ind - i * d + max_num) % max_num
        #     img_fp = os.path.join(img_folder, f"{i_temp:05d}.jpg")
        #     all_frames.add(img_fp)
        #     frame_indices.append(img_fp)
            
        # print(pic_path)
        # print(frame_indices)
    
    # with open("od_af_valid.txt", "w") as f:
    #     for item in all_frames:
    #         f.write(str(item) + "\n")
#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.MOT_IVD_v1_9 import MOT_IVD_v1_9
import torch
import torchvision.transforms as T
from datasets.dataset_mot import listDataset, avivd_collate_fn
import matplotlib as mpl
from matplotlib.lines import Line2D

def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 t-SNE 可视化高维 latent space"
    )
    # parser.add_argument(
    #     "--latent", type=str, required=True,
    #     help="路径到 latent features 的 .npy 文件，形状 (N, D)"
    # )
    # parser.add_argument(
    #     "--labels", type=str, required=True,
    #     help="路径到 labels 的 .npy 文件，形状 (N,), 值为 1,2,3"
    # )
    parser.add_argument(
        "--perplexity", type=float, default=30.0,
        help="t-SNE 的 perplexity 参数 (默认: 30.0)"
    )
    parser.add_argument(
        "--n_iter", type=int, default=1000,
        help="t-SNE 的最大迭代次数 (默认: 1000)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=200.0,
        help="t-SNE 的学习率 (默认: 200.0)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MOT_IVD_v1_9(num_classes=3).cuda()
    model.load_state_dict(torch.load("./runs/MOT_classification_cl_v1_9_yolo11s_af/best_model_epoch_12_mAP_0.9099.pth")["model_state_dict"])
    model.to(device)
    model.eval()
    kwargs = {'num_workers': 8, 'pin_memory': True, 'prefetch_factor': 2}

    # # 加载数据
    # latent = np.load(args.latent)      # shape: (N, D)
    # labels = np.load(args.labels)      # shape: (N,), values in {1,2,3}
    test_dataset = listDataset(
            base_path='/uu/sci.utah.edu/projects/smartair/Dataset',
            txt_list='./meta-files/testlist_e2e_new_2.txt',
            json_path='./datasets/test_tracks_yolov11s_2_corrected_padded.json',
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

    all_latent = np.empty((0, 256))
    all_labels = np.empty((0, 1))
    all_masks = np.empty((0, 1),  dtype=bool)

    with torch.no_grad():
        for batch in test_loader:
            _, bboxes, mask, audio, labels, video_ids, _, yolo_confs = batch
            # print(bboxes.shape, yolo_confs.shape)
            if bboxes.shape[0] == 0:
                continue
            # clips = clips.to(device)
            # clips = clips.permute(0, 2, 1, 3, 4)
            bboxes = bboxes.to(device)
            mask = mask.to(device)[:,-1:]
            audio = audio.to(device)
            labels = labels.to(device)[:,-1:]
            # print(bboxes.shape, audio.shape)
            logits, joint_emb_n = model(bboxes, audio)  # [BN, L, C]
            all_latent = np.concatenate((all_latent, joint_emb_n.detach().cpu().numpy()), axis=0)
            all_labels = np.concatenate((all_labels, labels.cpu().numpy()), axis=0)
            all_masks = np.concatenate((all_masks, mask.cpu().numpy()), axis=0)

    # all_latent = np.array(all_latent)
    # print(all_latent.shape, all_masks[:, 0])
    all_latent = all_latent[all_masks[:, 0], :]
    all_labels = all_labels[all_masks[:, 0], :]
    print(np.unique(all_labels))
    # 运行 t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        init="random",
        random_state=42
    )
    latent_2d = tsne.fit_transform(all_latent)

    cmap = plt.get_cmap("tab10")
    # 用边界归一化，确保 1/2/3 各自映射到一个固定颜色
    norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    # 绘图
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        latent_2d[:, 0],
        latent_2d[:, 1],
        c=all_labels.flatten(),
        alpha=0.7,
        cmap=cmap,
        norm=norm
    )
    plt.title("t-SNE Visualization of Latent Space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # 添加图例
    # handles, _ = scatter.legend_elements(num=3)
    # 按照同一个 cmap/norm 取出 1/2/3 对应的颜色，做图例
    legend_labels = ["M", "I", "Eoff"]
    class_vals = [0, 1, 2]
    legend_colors = cmap(norm(class_vals))

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="",
            markerfacecolor=col, markeredgecolor="none",
            alpha=0.7, label=lab)
        for col, lab in zip(legend_colors, legend_labels)
    ]
    plt.legend(handles=legend_handles, title="Classes")

    plt.tight_layout()
    plt.savefig("tsne.png")

if __name__ == "__main__":
    main()

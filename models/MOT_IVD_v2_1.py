import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- AUDIO TEMPORAL ENCODER ----------
class AudioTemporalEncoder(nn.Module):
    """
    输入 : audio [BN, 6, 128, 469]
    输出 : audio_embed [BN, 16, D_t]   # 与 L=16 帧对齐
    """
    def __init__(self, d_spatial=64, d_temporal=128, out_len=16):
        super().__init__()
        # 6 通道 × 128 Mel → conv 提取空间特征
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, d_spatial, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))   # 每个 time‑step → 向量
        )
        # 沿时间 469 建模
        self.gru = nn.GRU(input_size=d_spatial,
                          hidden_size=d_temporal,
                          batch_first=True,
                          bidirectional=False)
        # 469 → 16
        self.temporal_pool = nn.AdaptiveAvgPool1d(out_len)

    def forward(self, audio):                       # [BN, 6, 128, 469]
        BN, C, M, T = audio.shape                  # C=6, M=128, T=469
        # 把 time 维展开到 batch 维度做 Conv2d : (BN*T) × 6 × 128 × 1
        audio = audio.permute(0, 3, 1, 2).contiguous()   # [BN, 469, 6, 128]
        audio = audio.reshape(BN * T, C, M, 1)           # 宽度为1即可
        spatial = self.spatial_cnn(audio)                # [BN*T, d_s, 1, 1]
        spatial = spatial.view(BN, T, -1)                # [BN, 469, d_s]
        gru_out, _ = self.gru(spatial)                   # [BN, 469, d_t]
        # 469 → 16
        pooled = self.temporal_pool(gru_out.permute(0, 2, 1))  # [BN, d_t, 16]
        pooled = pooled.permute(0, 2, 1).contiguous()          # [BN, 16, d_t]
        return pooled


# ---------- AV MOTION NET ----------
class MOT_IVD_v2_1(nn.Module):
    """
    融合 bbox 时序 & audio 时序 , 输出 [BN, 16, num_classes]
    """
    def __init__(self, hidden_dim=128, num_classes=3):
        super(MOT_IVD_v2_1, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.L = 16

        # bbox branch
        self.bbox_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.bbox_conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # audio branch (new)
        self.audio_encoder = AudioTemporalEncoder(
            d_spatial=64, d_temporal=hidden_dim, out_len=self.L
        )

        # fusion + prediction
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    # -------- forward --------
    def forward(self, bboxes, audio):
        """
        bboxes : [BN, 16, 4]
        audio  : [BN, 6, 128, 469]
        return : logits [BN, 16, C]
        """
        BN, L, _ = bboxes.shape
        if BN == 0:
            return torch.empty(0, L, self.num_classes, device=bboxes.device)

        # --- bbox path ---
        delta = bboxes[:, 1:, :] - bboxes[:, :-1, :]
        zero_pad = torch.zeros_like(bboxes[:, :1, :])
        delta = torch.cat([zero_pad, delta], dim=1)
        bx = self.bbox_mlp(delta)           # [BN, L, D]
        bx = bx.permute(0, 2, 1)             # [BN, D, L]
        bx = self.bbox_conv1d(bx)            # [BN, D, L]
        bx = bx.permute(0, 2, 1)             # [BN, L, D]

        # --- audio path ---
        ax = self.audio_encoder(audio)       # [BN, L, D]

        # --- fuse & predict ---
        fused = torch.cat([bx, ax], dim=-1)  # [BN, L, 2D]
        logits = self.head(fused)            # [BN, L, C]
        return logits

    
if __name__ == "__main__":
    import torch

    # 假设有 2 个样本，每个样本最多 5 辆车，每辆车有 16 帧
    BN, L = 21, 16
    B = 2
    C = 3  # 类别数

    # 随机 bbox 输入，范围在 [0, 224] 内
    dummy_bboxes = torch.rand(BN, L, 4) * 224

    # 随机 audio 输入：6 通道 mel 频谱图，128 × 469 分辨率
    dummy_audio = torch.randn(BN, 6, 128, 469)

    # 创建模型实例
    model = MOT_IVD_v2_1(num_classes=C)

    # 推理
    output = model(dummy_bboxes, dummy_audio)

    print("输出 shape:", output.shape)  # 应为 [B, N, L, C]
    print("每辆车每帧的状态 logits:", output[0, 0, 0])  # 打印第一个样本第一个车第一帧的分类logits

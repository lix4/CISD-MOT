import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ---------- AUDIO TEMPORAL ENCODER ----------
class AudioTemporalEncoder(nn.Module):
    """
    输入 : audio [BN, 6, 128, 469]
    输出 : audio_embed [BN, 16, D_t]   # 与 L=16 帧对齐
    """
    def __init__(self, d_spatial=3456, d_temporal=128, out_len=16):
        super().__init__()

        # === Audio Encoder using MobileNetV3 ===
        self.a_model = timm.create_model('mobilenetv3_small_100', pretrained=True)
        self.a_model.global_pool = nn.Identity()
        self.a_model.conv_head = nn.Identity()
        self.a_model.act2 = nn.Identity()
        self.a_model.flatten = nn.Identity()
        self.a_model.classifier = nn.Identity()

        self.space_conv = nn.Sequential(
            nn.Conv2d(d_spatial, d_temporal, kernel_size=(4,1)),  # [BN,out_dim,1,15]
            nn.ReLU(inplace=True)
        )
        self.time_interpolation = nn.Upsample(size=out_len, mode='linear', align_corners=False)

    def forward(self, audio):                       # [BN, 6, 128, 469]
        # print(audio.shape)
        BN, C, M, T = audio.shape                  # C=6, M=128, T=469
        # 把 time 维展开到 batch 维度做 Conv2d : (BN*T) × 6 × 128 × 1
        audio_a = audio.view(-1, 1, M, T).repeat(1, 3, 1, 1)
        audio_feats=self.a_model(audio_a)
        audio_feats=audio_feats.reshape(BN, -1, 4, 15)
        # print("audio feats", audio_feats.shape)
        # 空间维度 (麦克风特征降维)
        spatial = self.space_conv(audio_feats)  # [BN,out_dim,1,15]
        spatial = spatial.squeeze(2)  # [BN,out_dim,15]

        # 时间维度 (插值到16帧)
        temporal = self.time_interpolation(spatial)  # [BN,out_dim,16]

        # Permute到[BN,16,D]
        audio_final = temporal.permute(0,2,1).contiguous()  # [BN,16,out_dim]

        return audio_final


# ---------- AV MOTION NET ----------
class MOT_IVD_v2_2(nn.Module):
    """
    融合 bbox 时序 & audio 时序 , 输出 [BN, 16, num_classes]
    """
    def __init__(self, hidden_dim=128, num_classes=3):
        super(MOT_IVD_v2_2, self).__init__()
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
            d_temporal=hidden_dim, out_len=self.L
        )

        # 最后的GRU (融合后做时序建模)
        self.final_gru = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)

        # Classifier
        self.head = nn.Linear(hidden_dim, num_classes)

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
        # print("bx", bx.shape)
        # --- audio path ---
        ax = self.audio_encoder(audio)       # [BN, L, D]
        # print("ax", ax.shape)
        # --- fuse & predict ---
        fused = torch.cat([bx, ax], dim=-1)  # [BN, L, 2D]
        fused_feat, _ = self.final_gru(fused) # [BN,L,D]
        logits = self.head(fused_feat)            # [BN, L, C]
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
    model = MOT_IVD_v2_2(num_classes=C)

    # 推理
    output = model(dummy_bboxes, dummy_audio)

    print("输出 shape:", output.shape)  # 应为 [B, N, L, C]
    print("每辆车每帧的状态 logits:", output[0, 0, 0])  # 打印第一个样本第一个车第一帧的分类logits

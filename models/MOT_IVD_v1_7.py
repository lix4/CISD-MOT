import torch, torch.nn as nn, torch.nn.functional as torch_F
from einops import rearrange
import timm

import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
import timm

# ---------- Helper Blocks ----------
class BGRU(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.gru_forward = nn.GRU(channel, channel, num_layers=1, bidirectional=False, batch_first=True)
        self.gru_backward = nn.GRU(channel, channel, num_layers=1, bidirectional=False, batch_first=True)
        self.gelu = nn.GELU()
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                nn.init.kaiming_normal_(m.weight_ih_l0)
                nn.init.kaiming_normal_(m.weight_hh_l0)
                nn.init.zeros_(m.bias_ih_l0)
                nn.init.zeros_(m.bias_hh_l0)

    def forward(self, x):
        # forward direction
        x, _ = self.gru_forward(x)
        x = self.gelu(x)
        # backward direction
        x = torch.flip(x, dims=[1])
        x, _ = self.gru_backward(x)
        x = torch.flip(x, dims=[1])
        x = self.gelu(x)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kw):
        return x + self.fn(self.norm(x), **kw)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden=512, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop)
        )
    def forward(self, x):
        return self.net(x)

class CrossAttentionWrapper(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
    def forward(self, x, y):
        return self.attn(x, context=y)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, drop=0.1):
        super().__init__()
        self.h = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(drop))
    def forward(self, x, context=None):
        context = x if context is None else context
        qkv = self.to_qkv(torch.cat([x, context], dim=1))
        d = x.shape[-1]
        n = x.shape[1]
        q, k, v = qkv.split([d, d, d], dim=-1)
        q = rearrange(q[:, :n], 'b n (h d) -> b h n d', h=self.h)
        k = rearrange(k[:, n:], 'b n (h d) -> b h n d', h=self.h)
        v = rearrange(v[:, n:], 'b n (h d) -> b h n d', h=self.h)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(-1)
        out = rearrange(attn @ v, 'b h n d -> b n (h d)')
        return self.out(out)

def make_block(dim, heads=4, drop=0.1, cross=False):
    attn = Attention(dim, heads, drop)
    if cross:
        return nn.ModuleList([
            PreNorm(dim, CrossAttentionWrapper(attn)),
            PreNorm(dim, FeedForward(dim))
        ])
    else:
        return nn.ModuleList([
            PreNorm(dim, attn),
            PreNorm(dim, FeedForward(dim))
        ])

# ---------- Audio Patch Encoder ----------
class AudioPatchEncoder(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.blocks = nn.ModuleList(make_block(dim, heads) for _ in range(2))
    def forward(self, x):
        for attn, ff in self.blocks:
            x = attn(x)
            x = ff(x)
        return x

# ---------- Cross Fusion ----------
class CrossFusion(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.blocks = nn.ModuleList(make_block(dim, heads, cross=True) for _ in range(2))
    def forward(self, query, context):
        for attn, ff in self.blocks:
            query = attn(query, y=context)
            query = ff(query)
        return query

# ---------- Main Model ----------
class MOT_IVD_v1_7(nn.Module):
    """改进版：
    1. 用 1×1 Conv 将 6 通道谱图映射到 3 通道（可学习），替换原先 repeat。
    2. delta 与 bbox 先求和后投影，避免简单拼接。
    3. 时序压缩改为均值池化，专注整段判别。
    4. Contrastive projector 改成两层。
    5. 移除调试 print。
    """
    def __init__(self, num_classes=3, d=256, heads=4):
        super().__init__()
        self.d = d
        self.L = 16
        self.num_classes = num_classes

        # bbox & delta branches
        self.delta_mlp = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, d))
        self.delta_conv = nn.Conv1d(d, d, 3, padding=1)

        self.bbox_mlp = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, d))
        self.bbox_conv = nn.Conv1d(d, d, 3, padding=1)

        # -------- 音频 backbone --------
        # 6 通道谱图 → 3 通道，可学习映射
        self.rgb_conv = nn.Conv2d(6, 3, kernel_size=1, bias=False)

        self.mob = timm.create_model('mobilenetv3_small_100', pretrained=True)
        # 去掉 mobilenet 的头
        self.mob.global_pool = nn.Identity()
        self.mob.conv_head = nn.Identity()
        self.mob.act2 = nn.Identity()
        self.mob.flatten = nn.Identity()
        self.mob.classifier = nn.Identity()

        # -------- 下采样与线性降维 --------
        # mobilenet 输出 [B,576,4,15]
        self.audio_down = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 4×15 → 2×7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.delta_down = nn.Sequential(nn.Linear(d, 128), nn.ReLU(inplace=True))
        self.bbox_down = nn.Sequential(nn.Linear(d, 128), nn.ReLU(inplace=True))

        # 5888 = 2048 + 2048 + 1792
        self.cl_projector = nn.Sequential(
            nn.Linear(5888, 512), nn.ReLU(inplace=True), nn.Linear(512, 256)
        )

        self.audio_proj = nn.Conv2d(576, d, kernel_size=1)
        self.audio_enc = AudioPatchEncoder(dim=d, heads=heads)
        self.cross = CrossFusion(dim=d, heads=heads)
        self.bi_gru = BGRU(d)

        # 2D sinusoidal pos enc for audio 4×15 patch
        self.register_buffer('pe2d', self._make_2d_pos(4, 15, d))

        # bbox & delta 融合投影
        self.fuse_proj = nn.Linear(d, d)

        # clip-level classifier
        self.cls = nn.Linear(d, num_classes)

    @staticmethod
    def _make_2d_pos(H, W, dim):
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        pos = torch.stack([y, x], dim=-1).float().reshape(-1, 2)
        div = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.)) / dim))
        pe = torch.zeros(pos.size(0), dim)
        pe[:, 0::2] = torch.sin(pos[:, 0:1] * div)
        pe[:, 1::2] = torch.cos(pos[:, 0:1] * div)
        return pe  # [60,dim]

    def forward(self, bboxes, audio):
        BN, L, _ = bboxes.shape  # L 应为 16
        if BN == 0:
            return torch.empty(0, self.num_classes, device=bboxes.device)

        # ===== bbox delta =====
        delta = bboxes[:, 1:] - bboxes[:, :-1]
        delta = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1)  # pad 第 0 帧
        delta = self.delta_mlp(delta)  # [BN,16,d]
        delta = self.delta_conv(delta.permute(0, 2, 1)).permute(0, 2, 1)

        bboxes_emb = self.bbox_mlp(bboxes)  # [BN,16,d]
        bboxes_emb = self.bbox_conv(bboxes_emb.permute(0, 2, 1)).permute(0, 2, 1)

        # ===== audio branch =====
        B, C, T, F_ = audio.shape  # C==6
        audio_rgb = self.rgb_conv(audio)  # [BN,3,T,F]
        feat = self.mob(audio_rgb)        # [BN,576,4,15]

        # ↓ contrastive 投影用的降采样特征
        proj_a = self.audio_down(feat).flatten(1)  # [BN,1792]

        bx_proj = self.bbox_down(bboxes_emb).reshape(BN, -1)   # 2048
        delta_proj = self.delta_down(delta).reshape(BN, -1)    # 2048

        joint_emb = torch.cat([bx_proj, delta_proj, proj_a], dim=1)
        # print(joint_emb.shape)
        joint_emb_n = F.normalize(self.cl_projector(joint_emb), dim=1)

        # ↓ attention 走的是原始 patch & bbox 表征
        feat2 = self.audio_proj(feat).flatten(2).permute(0, 2, 1)  # [BN,60,d]
        feat2 = feat2 + self.pe2d.to(feat2.device)
        feat2 = self.audio_enc(feat2)

        # delta+bbox 融合后再投影 (保持长度 16)
        query = self.fuse_proj(delta + bboxes_emb)
        fused = self.cross(query, feat2)  # [BN,16,d]
        fused = self.bi_gru(fused)

        # 整段判别：对 16 帧取均值
        # print(fused.shape)
        clip_feat = fused.mean(dim=1)  # [BN,d]
        logits = self.cls(clip_feat)   # [BN,C]

        return logits, joint_emb_n


    
if __name__ == "__main__":
    import torch

    # 假设有 2 个样本，每个样本最多 5 辆车，每辆车有 16 帧
    BN, L = 21, 16
    B = 2
    C = 3  # 类别数

    # 随机 bbox 输入，范围在 [0, 224] 内
    dummy_bboxes = torch.rand(BN, L, 4).cuda() * 224

    # 随机 audio 输入：6 通道 mel 频谱图，128 × 469 分辨率
    dummy_audio = torch.randn(BN, 6, 128, 469).cuda()

    # 创建模型实例
    model = MOT_IVD_v1_7(num_classes=C).cuda()

    # 推理
    output = model(dummy_bboxes, dummy_audio)

    print("输出 shape:", output[0].shape)  # 应为 [B, N, L, C]

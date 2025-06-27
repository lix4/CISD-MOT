import torch
import torch.nn as nn
import timm

class MOT_IVD_v1_1(nn.Module):
    def __init__(self, num_classes, embed_dim=256, num_heads=4, num_encoder_layers=12, num_decoder_layers=6, max_seq_len=16):
        super(MOT_IVD_v1_1, self).__init__()

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # === BBox Encoder: [BN, L, 4] → [BN, L, D] ===
        self.bbox_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, embed_dim))


        # === Audio Encoder using MobileNetV3 ===
        self.a_model = timm.create_model('mobilenetv3_small_100', pretrained=True)
        self.a_model.global_pool = nn.Identity()
        self.a_model.conv_head = nn.Identity()
        self.a_model.act2 = nn.Identity()
        self.a_model.flatten = nn.Identity()
        self.a_model.classifier = nn.Identity()

        self.audio_proj = nn.Conv2d(576 * 6, embed_dim, kernel_size=1)

        # === Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # === Transformer Decoder ===
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # === Learnable Queries (L slots)
        self.query_embed = nn.Embedding(max_seq_len, embed_dim)

        # === Classification Head ===
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, bboxes, audio):
        """
        bboxes: [BN, L, 4]
        audio:  [BN, 6, 128, 469]
        returns: logits [BN, L, C]
        """
        BN, L, _ = bboxes.shape

        # === Encode bbox
        bbox_feat = self.bbox_encoder(bboxes)  # [BN, L, D]
        bbox_feat = bbox_feat.permute(1, 0, 2)  # [L, BN, D] to match transformer
        bbox_feat += self.pos_embed.unsqueeze(1)  # [L, BN, D] 直接相加位置编码

        # === Encode audio
        B, M, T, F = audio.shape
        audio = audio.view(-1, 1, T, F).repeat(1, 3, 1, 1)  # [BN*6, 3, T, F]
        audio_feat = self.a_model(audio)  # [BN*6, 576, 4, 15]
        # print(audio_feat.shape)
        audio_feat = audio_feat.view(BN, M * 576, 4, 15)  # [BN, 576*6, 4, 15]
        audio_feat = self.audio_proj(audio_feat)  # [BN, D, 4, 15]
        audio_feat = audio_feat.flatten(2).permute(2, 0, 1)  # [S_audio, BN, D]

        print(audio_feat.shape)
        # === Concatenate bbox and audio for encoder ===
        encoder_input = torch.cat([bbox_feat, audio_feat], dim=0)  # [(L + S_audio), BN, D]

        # === Transformer Encoder ===
        memory = self.transformer_encoder(encoder_input)  # [(L + S_audio), BN, D]

        # === Prepare decoder queries ===
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, BN, 1)  # [L, BN, D]

        # === Transformer Decoder
        decoded = self.transformer_decoder(query_embed, memory)  # [L, BN, D]
        decoded = decoded.permute(1, 0, 2)  # [BN, L, D]

        logits = self.cls_head(decoded)  # [BN, L, C]
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
    model = MOT_IVD_v1_1(num_classes=C)

    # 推理
    output = model(dummy_bboxes, dummy_audio)

    print("输出 shape:", output.shape)  # 应为 [B, N, L, C]
    print("每辆车每帧的状态 logits:", output[0, 0, 0])  # 打印第一个样本第一个车第一帧的分类logits

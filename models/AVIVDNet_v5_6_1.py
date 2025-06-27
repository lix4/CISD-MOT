import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.mobilenetv2 import get_model
import timm
from .conv import Conv2d
import numpy as np
import torch, torch.nn.functional as F


class DecoupledHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        print('==============================')
        print('Head: Decoupled Head')
        self.num_cls_heads = 1
        self.num_reg_heads = 1
        self.act_type = 'lrelu'
        self.norm_type = 'BN'
        self.head_dim = 64
        self.depthwise = True

        self.cls_head = nn.Sequential(*[
            Conv2d(self.head_dim, 
                   self.head_dim, 
                   k=3, p=1, s=1, 
                   act_type=self.act_type, 
                   norm_type=self.norm_type,
                   depthwise=self.depthwise)
                   for _ in range(self.num_cls_heads)])
        self.reg_head = nn.Sequential(*[
            Conv2d(self.head_dim, 
                   self.head_dim, 
                   k=3, p=1, s=1, 
                   act_type=self.act_type, 
                   norm_type=self.norm_type,
                   depthwise=self.depthwise)
                   for _ in range(self.num_reg_heads)])


    def forward(self, cls_feat, reg_feat):
        cls_feats = self.cls_head(cls_feat)
        reg_feats = self.reg_head(reg_feat)
        return cls_feats, reg_feats

# 多模态 Transformer 编码器
class MultiModalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(MultiModalTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x
    
# DETR 模型（修改版）
class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, num_queries=50):
        super(DETR, self).__init__()
        # Transformer
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=nheads,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)
        # 预测头
        # self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 表示 no-object 类别
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        # self.head_dim=256
        self.num_classes = num_classes

        self.classification_head = nn.Linear(4*256, 3, bias=True)


    def forward(self, src):
        # src: (sequence_length, batch_size, hidden_dim)
        batch_size = src.shape[1]
        # 准备查询嵌入
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # (num_queries, batch_size, hidden_dim)
        # Transformer 解码器需要将 src 和 tgt 维度为 (sequence_length, batch_size, hidden_dim)
        hs = self.transformer(src=src, tgt=query_embed)  # hs: (num_queries, batch_size, hidden_dim)

        x = hs.permute(1, 0, 2).reshape(batch_size, 7, 7, 256)   # (B, 7, 7, 256)
        x = F.pad(x, pad=(0, 0, 0, 1, 0, 1))         # (B, 8, 8, 256)

        H, W = x.shape[1], x.shape[2]                # 8, 8  现在都是偶数

        # ② 2×2 window 合并到通道 → (B, H/2, W/2, 4C)
        x = x.reshape(batch_size, H//2, 2, W//2, 2, 256)        # (B, 4, 2, 4, 2, 256)
        x = x.permute(0, 1, 3, 5, 2, 4)              # (B, 4, 4, 256, 2, 2)
        x = x.reshape(batch_size, H//2, W//2, 4 * 256)          # (B, 4, 4, 1024)

        # ③ 展平成 sequence 再线性降维到你想要的 C′
        x = x.view(batch_size, -1, 4 * 256)                     # (B, 16, 1024)
        out = self.classification_head(x)                                  # (B, 16, 256)
        # print(out.shape)

        return out

# 整合模型
class AVIVDNetV5_6_1(nn.Module):

    def __init__(self, num_classes, embed_dim=256, num_heads=8, num_layers=6, num_queries=49):
        super(AVIVDNetV5_6_1, self).__init__()
        ########## v model ##########
        self.backbone_3d = get_model(width_mult=1.)
        new_state_dict = {}
        pretrain_weights=torch.load('./pretrained_weights/trainlist_e2e_new_1_motion/yowo_car_16f_best.pth')['state_dict']
        for k, v in pretrain_weights.items():
            new_key = k.replace("module.backbone_3d.", "")
            if 'conv_final' in new_key:
                continue
            new_state_dict[new_key] = v
        self.backbone_3d.load_state_dict(new_state_dict)
        # self.backbone_3d = self.backbone_3d.eval()
        # for param in self.backbone_3d.parameters():
        #     param.requires_grad = False
        ########## v model ##########
        # self.visual_encoder = VisualEncoder(embed_dim)
        num_channels=6

        ########## a model ##########
        self.a_model = timm.create_model('mobilenetv3_small_100', pretrained=True)
        self.a_model.global_pool=nn.Identity()
        self.a_model.conv_head=nn.Identity()
        self.a_model.act2=nn.Identity()
        self.a_model.flatten=nn.Identity()
        self.a_model.classifier=nn.Identity()
        # print(self.a_model)
        ########## a model ##########
        # self.audio_encoder = AudioEncoder(embed_dim)
        self.visual_proj=nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.audio_proj=nn.Conv2d(576*num_channels, 256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.transformer = MultiModalTransformer(embed_dim, num_heads, num_layers)
        self.detr = DETR(num_classes, hidden_dim=embed_dim, nheads=num_heads,
                         num_encoder_layers=num_layers, num_decoder_layers=num_layers, num_queries=num_queries)
    

    def forward(self, video, audio):
        # print("here", self.trainable)
        nB=audio.shape[0]
        # nC=audio.shape[1]

        audio = audio.unsqueeze(2).repeat(1,1,3,1,1).view(-1, 3, 128, 469)
        audio_feat = self.a_model(audio)     # (batch_size, embed_dim)
        
        audio_feat = audio_feat.reshape(nB, -1, 4, 15)
        visual_feat = self.backbone_3d(video)   # (batch_size, embed_dim)
        # print(visual_feat.shape)
        visual_feat = visual_feat.squeeze(2)
        # print(audio_feat.shape)
        visual_feat=self.visual_proj(visual_feat)
        audio_feat=self.audio_proj(audio_feat)
        audio_feat = audio_feat.view(nB, 256, -1).permute(2,0,1)
        # print(audio_feat.shape)
        visual_feat = visual_feat.view(nB, 256, -1).permute(2,0,1)

        # print(audio_feat.shape, visual_feat.shape)
        # 合并特征，增加一个序列维度
        combined_feat = torch.cat([audio_feat, visual_feat], dim=0)  # (2, batch_size, embed_dim)
        # print(combined_feat.shape)
        # 通过 Transformer 编码器
        transformed_feat = self.transformer(combined_feat)  # (2, batch_size, embed_dim)
        # print(transformed_feat.shape)
        # 使用融合后的特征进行检测
        # print(self.trainable)
        # if not trainable:
            # return self.detr.inference(transformed_feat)
        output = self.detr(transformed_feat)  # 注意，这里 src 是 (sequence_length, batch_size, embed_dim)

        return output


if __name__ == '__main__':
    nB=55
    v_clip=torch.randn([nB,3,16, 112, 112]).cuda()
    audio=torch.randn([nB, 6, 128, 469]).cuda()
    # # mic_meta={0:(1,2), 1:(322,435), 2:(234,432), 3:(110,128), 4:(123, 42), 5:(123, 33)}
    # # mic_meta = {k: v.to(device='gpu', non_blocking=True) for k, v in mic_meta.items()}

    model=AVIVDNetV5_6_1(num_classes=3, ).cuda()
    # model.load_weights(True)
    # # model=model.cuda()
    # # print(model)
    out=model(v_clip, audio)
    print(out.shape)
    # print(len(out[0]), len(out[1]), len(out[2]))
    # print(model)
    # print(out['pred_conf'][0].shape, out['pred_cls'][0].shape, out['pred_box'][0].shape, out['anchors'][0].shape)
    
    # out_va: num_heads=4 d_model=128
    # Total params: 71,634,688
    # Trainable params: 71,634,688
    # Non-trainable params: 0
    # Flops:  68745395584
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.numel()}")
    # summary(model, input_size=((nB,3,16,224,224), (nB, 6, 128, 469)))
    # flops = FlopCountAnalysis(model, (torch.randn(nB,3,16,224,224).cuda(), torch.randn(nB, 6, 128, 469).cuda()))
    # print(f"FLOPs: {flops.total()}")

    # params = parameter_count_table(model)
    # print(params)
    
    # out_va: num_heads=1 d_model=128
    # # of params: 
    # FLOPs:
    # summary(model, input_size=((nB,3,16,224,224),(nB, 6, 128, 469)))



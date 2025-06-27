import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.mobilenetv2 import get_model
import timm


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
        self.time_conv = nn.Conv1d(49, 16, kernel_size=1)
        self.final_classifier = nn.Linear(256, 3)
        # self.final_conv = nn.Conv2d(256, 5*(3+4+1), kernel_size=1, bias=False)

    def forward(self, src):
        # src: (sequence_length, batch_size, hidden_dim)
        batch_size = src.shape[1]
        # 准备查询嵌入
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # (num_queries, batch_size, hidden_dim)
        # Transformer 解码器需要将 src 和 tgt 维度为 (sequence_length, batch_size, hidden_dim)
        hs = self.transformer(src=src, tgt=query_embed)  # hs: (num_queries, batch_size, hidden_dim)
        # hs = hs.view(7,7,batch_size, -1).permute(2,3,0,1)
        hs = hs.permute(1, 0, 2)
        # print(hs.shape)
        hs = self.time_conv(hs)
        hs = self.final_classifier(hs)
        # print(hs.shape)
        # # 输出
        # outputs_class = self.class_embed(hs)  # (num_queries, batch_size, num_classes + 1)
        # outputs_coord = self.bbox_embed(hs).sigmoid()  # (num_queries, batch_size, 4)
        
        # # 调整输出维度为 (batch_size, num_queries, ...)
        # outputs_class = outputs_class.permute(1, 0, 2)  # (batch_size, num_queries, num_classes + 1)
        # outputs_coord = outputs_coord.permute(1, 0, 2)  # (batch_size, num_queries, 4)
        # return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return hs

# 整合模型
class MOT_IVD_v3_1(nn.Module):

    def __init__(self, num_classes, embed_dim=256, num_heads=8, num_layers=6, num_queries=49):
        super(MOT_IVD_v3_1, self).__init__()
        self.seen = 0
        ########## v model ##########
        self.backbone_3d = get_model(width_mult=1.)
        new_state_dict = {}
        pretrain_weights=torch.load('../CISD-Research/backup/trainlist_e2e_new_1_motion/yowo_car_16f_best.pth')['state_dict']
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
        nB=audio.shape[0]
        nC=audio.shape[1]
        # print(audio.shape)
        audio = audio.unsqueeze(2).repeat(1,1,3,1,1).view(-1, 3, 128, 469)
        audio_feat = self.a_model(audio)     # (batch_size, embed_dim)
        
        audio_feat = audio_feat.reshape(nB, -1, 4, 15)
        visual_feat = self.backbone_3d(video)   # (batch_size, embed_dim)
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
        # 通过 Transformer 编码器
        transformed_feat = self.transformer(combined_feat)  # (2, batch_size, embed_dim)
        # print(transformed_feat.shape)
        # 使用融合后的特征进行检测
        output = self.detr(transformed_feat)  # 注意，这里 src 是 (sequence_length, batch_size, embed_dim)
        return output


if __name__ == '__main__':
    nB=15
    v_clip=torch.randn([nB,3,16,112, 112]).cuda()
    audio=torch.randn([nB, 6, 128, 469]).cuda()
    # # mic_meta={0:(1,2), 1:(322,435), 2:(234,432), 3:(110,128), 4:(123, 42), 5:(123, 33)}
    # # mic_meta = {k: v.to(device='gpu', non_blocking=True) for k, v in mic_meta.items()}

    model=MOT_IVD_v3_1(num_classes=3, ).cuda()
    # model.load_weights(True)
    # # model=model.cuda()
    # # print(model)
    out=model(v_clip,audio)
    # print(model)
    # print(out['pred_logits'].shape, out['pred_boxes'].shape)
    
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

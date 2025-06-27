import torch, torch.nn as nn
import torch.nn.functional as torch_F
from einops import rearrange
import timm

# ---------- Helper Blocks ----------
# class BGRU(nn.Module):
#     def __init__(self, channel):
#         super(BGRU, self).__init__()

#         self.gru_forward = nn.GRU(input_size = channel, hidden_size = channel, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
#         self.gru_backward = nn.GRU(input_size = channel, hidden_size = channel, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
        
#         self.gelu = nn.GELU()
#         self.__init_weight()

#     def forward(self, x):
#         x, _ = self.gru_forward(x)
#         x = self.gelu(x)
#         x = torch.flip(x, dims=[1])
#         x, _ = self.gru_backward(x)
#         x = torch.flip(x, dims=[1])
#         x = self.gelu(x)

#         return x

#     def __init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.GRU):
#                 torch.nn.init.kaiming_normal_(m.weight_ih_l0)
#                 torch.nn.init.kaiming_normal_(m.weight_hh_l0)
#                 m.bias_ih_l0.data.zero_()
#                 m.bias_hh_l0.data.zero_()

# class PreNorm(nn.Module):
#     def __init__(self, dim, fn): super().__init__(); self.norm=nn.LayerNorm(dim); self.fn=fn
#     def forward(self,x,**kw): return x+self.fn(self.norm(x),**kw)

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden=512, drop=0.1):
#         super().__init__()
#         self.net=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(drop),
#                                nn.Linear(hidden,dim),nn.Dropout(drop))
#     def forward(self,x): return self.net(x)

# class CrossAttentionWrapper(nn.Module):
#     def __init__(self, attn):
#         super().__init__()
#         self.attn = attn

#     def forward(self, x, y):
#         return self.attn(x, context=y)

# class Attention(nn.Module):
#     def __init__(self, dim, heads=4, drop=0.1):
#         super().__init__()
#         self.h=heads; self.scale=(dim//heads)**-0.5
#         self.to_qkv=nn.Linear(dim, dim*3, bias=False)
#         self.out=nn.Sequential(nn.Linear(dim,dim), nn.Dropout(drop))
#     def forward(self,x,context=None):
#         context=x if context is None else context
#         # print(x.device, context.device, torch.cat([x,context],dim=1).device)
#         qkv=self.to_qkv(torch.cat([x,context],dim=1))
#         d=x.shape[-1]; n=x.shape[1]
#         q,k,v=qkv.split([d,d,d],dim=-1)
#         q=rearrange(q[:,:n],'b n (h d)->b h n d',h=self.h)
#         k=rearrange(k[:,n:],'b n (h d)->b h n d',h=self.h)
#         v=rearrange(v[:,n:],'b n (h d)->b h n d',h=self.h)
#         attn=(q@k.transpose(-1,-2))*self.scale
#         attn=attn.softmax(-1)
#         out=rearrange(attn@v,'b h n d->b n (h d)')
#         return self.out(out)

# def make_block(dim, heads=4, drop=0.1, cross=False):
#     attn = Attention(dim, heads, drop)
#     if cross:
#         return nn.ModuleList([
#             PreNorm(dim, CrossAttentionWrapper(attn)),
#             PreNorm(dim, FeedForward(dim))
#         ])
#     else:
#         return nn.ModuleList([
#             PreNorm(dim, attn),
#             PreNorm(dim, FeedForward(dim))
#         ])


# # ---------- Audio Patch Encoder ----------
# class AudioPatchEncoder(nn.Module):
#     def __init__(self, dim=256, heads=4):
#         super().__init__()
#         self.blocks=nn.ModuleList(make_block(dim,heads) for _ in range(2))
#     def forward(self,x):
#         for attn,ff in self.blocks:
#             x=attn(x); x=ff(x)
#         return x

# class BBoxBranch(nn.Module):
#     def __init__(self, d=256):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(4, 128),
#             nn.ReLU(),
#             nn.Linear(128, d))
#         # Dilated temporal conv + residual
#         self.tconv = nn.Conv1d(d, d, kernel_size=3, padding=2, dilation=2)
#         self.ln = nn.LayerNorm(d)

#     def forward(self, x):            # x: [B, T, 4]
#         x = self.mlp(x)              # [B, T, d]
#         x = x + self.tconv(x.permute(0, 2, 1)).permute(0, 2, 1)
#         return self.ln(x)


# ---------- Main Model ----------
class MOT_IVD_CL(nn.Module):
    def __init__(self, num_classes=3, d=256):
        super(MOT_IVD_CL, self).__init__()
        self.d=d; self.L=16; self.num_classes=num_classes
        # bbox branch
        self.bbox_mlp=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,d))
        self.displacement_mlp=nn.Sequential(nn.Linear(4,128),nn.ReLU(),nn.Linear(128,d))
        self.bbox_conv=nn.Conv1d(d,d,3,padding=1)
        # audio backbone
        # self.mob=timm.create_model('mobilenetv3_small_100',pretrained=True,features_only=True)
        self.mob = timm.create_model('mobilenetv3_small_100', pretrained=True)
        self.mob.global_pool=nn.Identity()
        self.mob.conv_head=nn.Identity()
        self.mob.act2=nn.Identity()
        self.mob.flatten=nn.Identity()
        self.mob.classifier=nn.Identity()
        
        self.audio_proj=nn.Conv2d(576*6,d,kernel_size=1)
        # self.audio_enc=AudioPatchEncoder(dim=d,heads=heads)
        # self.cross=CrossFusion(dim=d,heads=heads)
        # self.gru=nn.GRU(d,d,batch_first=True)
        # self.bi_gru = BGRU(d)
        self.proj_1 = nn.Linear(256, 128)
        self.proj_2 = nn.Linear(128, 1)
        
        self.cl_proj=nn.Linear(92, 64)

        # self.conv = nn.Conv1d(16, 1, kernel_size=1)


    def forward(self, bboxes, audio):
        BN,L,_=bboxes.shape
        if BN==0:
            return torch.empty(0,self.num_classes,device=bboxes.device)

        # === bbox delta ===
        delta=bboxes[:,1:]-bboxes[:,:-1]
        delta=torch.cat([torch.zeros_like(delta[:,:1]),delta],1)  # pad
        delta_emb=self.displacement_mlp(delta)            # [BN,16,d]
        delta_emb=self.bbox_conv(delta_emb.permute(0,2,1)).permute(0,2,1)  # Conv1d
        
        bboxes_emb = self.bbox_mlp(bboxes)
        bboxes_emb=self.bbox_conv(bboxes_emb.permute(0,2,1)).permute(0,2,1)  # Conv1d
        
        # print(delta_emb.shape, bboxes_emb.shape)
        # === audio patch ===
        B,Nc,T,F=audio.shape              # [BN,6,128,469]
        audio=audio.view(-1,1,T,F).repeat(1,3,1,1)           # 灰度→3ch
        feat=self.mob(audio)              # [BN*6,576,4,15]
        feat=feat.view(BN,-1,4,15)        # [BN,3456,4,15]
        feat=self.audio_proj(feat)        # [BN,d,4,15]
        feat=feat.flatten(2).permute(0,2,1)  # [BN,60,d]
        # print(feat.shape)
        # feat=feat+self.pe2d.to(feat.device)   # 加位置
        # cat
        fused = torch.cat([delta_emb, bboxes_emb, feat], dim=1)
        # print("A", fused.shape)
        fused = self.proj_2(torch_F.relu(self.proj_1(fused))).squeeze(2)
        x = self.cl_proj(fused)  # [B, D, T]
        n_feature_emb = torch_F.normalize(x, dim=1)
        # print(n_feature_emb.shape)
        return fused, n_feature_emb

    
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
    model = MOT_IVD_CL(num_classes=C).cuda()

    # 推理
    fused, n_feature_emb = model(dummy_bboxes, dummy_audio)

    print("输出 shape:", fused.shape)  # 应为 [B, N, L, C]
    # print("每辆车每帧的状态 logits:", fused[0, 0, 0])  # 打印第一个样本第一个车第一帧的分类logits

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.mobilenetv2 import get_model
import timm
from .conv import Conv2d
import numpy as np
import torch.nn.functional as F


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
        self.num_cls_heads = 2
        self.num_reg_heads = 2
        # self.head_dim=256
        self.stride=[8, 16, 32]
        self.num_classes = num_classes
        self.topk = 50
        self.conf_thresh = 0.005
        self.nms_thresh = 0.4


        t_dims = [4, 2, 1]
        dims = [32, 96, 1280]
        self.dim_converter_1 = nn.ModuleList(
            [nn.Sequential(nn.Conv3d(dims[i], 256, kernel_size=(t_dims[i], 1, 1), padding=0, dilation=1, groups=1, bias=False),
                                      nn.BatchNorm3d(256),
                                      nn.LeakyReLU(0.1, inplace=True)) for i in range(len(t_dims))]
        )

        self.dim_converter_2 = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(512, 256, 1, stride=1, padding=0, dilation=1, groups=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.LeakyReLU(0.1, inplace=True)) for _ in range(len(dims))]
        )
        
        # branching convs
        self.cls_conv_sets = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(256, 64, 1, stride=1, padding=0, dilation=1, groups=1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.LeakyReLU(0.1, inplace=True)) for _ in range(len(self.stride))]
        )

        self.reg_conv_sets = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(256, 64, 1, stride=1, padding=0, dilation=1, groups=1, bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.LeakyReLU(0.1, inplace=True)) for _ in range(len(self.stride))]
        )

        
        ## head
        # self.head = DecoupledHead(None)
        self.heads = nn.ModuleList(
            [DecoupledHead(None) for _ in range(len(self.stride))]
        )

        ## pred
        head_dim = 64
        self.conf_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, 1, kernel_size=1)
                for _ in range(len(self.stride))
                ]) 
        self.cls_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, self.num_classes, kernel_size=1)
                for _ in range(len(self.stride))
                ]) 
        self.reg_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, 4, kernel_size=1) 
                for _ in range(len(self.stride))
                ])         
    
    def decode_boxes(self, anchors, pred_reg, stride):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [B, M, 4]
        """
        # center of bbox
        pred_ctr_xy = anchors + pred_reg[..., :2] * stride
        # size of bbox
        pred_box_wh = pred_reg[..., 2:].exp() * stride

        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box
    
    def generate_anchors(self, fmp_size, stride):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= stride
        anchors = anchor_xy.cuda()
        return anchors
    
    def nms(self, bboxes, scores, nms_thresh):
        """"Pure Python NMS."""
        x1 = bboxes[:, 0]  #xmin
        y1 = bboxes[:, 1]  #ymin
        x2 = bboxes[:, 2]  #xmax
        y2 = bboxes[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(iou <= nms_thresh)[0]
            order = order[inds + 1]

        return keep
    
    # def multiclass_nms_class_aware(self, scores, labels, bboxes, nms_thresh, num_classes):
    #     # nms
    #     keep = np.zeros(len(bboxes), dtype=np.int32)
    #     for i in range(num_classes):
    #         inds = np.where(labels == i)[0]
    #         if len(inds) == 0:
    #             continue
    #         c_bboxes = bboxes[inds]
    #         c_scores = scores[inds]
    #         c_keep = self.nms(c_bboxes, c_scores, nms_thresh)
    #         keep[inds[c_keep]] = 1

    #     keep = np.where(keep > 0)
    #     scores = scores[keep]
    #     labels = labels[keep]
    #     bboxes = bboxes[keep]

    #     return scores, labels, bboxes

    def post_process_one_hot(self, conf_preds, cls_preds, reg_preds, anchors):
        """
        Input:
            conf_preds: (Tensor) [H x W, 1]
            cls_preds: (Tensor) [H x W, C]
            reg_preds: (Tensor) [H x W, 4]
        """
        
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for level, (conf_pred_i, cls_pred_i, reg_pred_i, anchors_i) in enumerate(zip(conf_preds, cls_preds, reg_preds, anchors)):
            # (H x W x C,)
            scores_i = (torch.sqrt(conf_pred_i.sigmoid() * cls_pred_i.sigmoid())).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk, reg_pred_i.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            reg_pred_i = reg_pred_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]

            # decode box: [M, 4]
            bboxes = self.decode_boxes(anchors_i, reg_pred_i, self.stride[level])

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        # scores, labels, bboxes = self.multiclass_nms_class_aware(
        #     scores, labels, bboxes, self.nms_thresh, self.num_classes)
        keep = self.nms(bboxes, scores, self.nms_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        return scores, labels, bboxes
    
    @torch.no_grad()
    def inference(self, src, spatial_feats):
        # src: (sequence_length, batch_size, hidden_dim)
        batch_size = src.shape[1]
        # 准备查询嵌入
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # (num_queries, batch_size, hidden_dim)
        # Transformer 解码器需要将 src 和 tgt 维度为 (sequence_length, batch_size, hidden_dim)
        hs = self.transformer(src=src, tgt=query_embed)  # hs: (num_queries, batch_size, hidden_dim)

        hs = hs.view(7,7,batch_size, -1).permute(2,3,0,1)

        all_conf_preds = []
        all_cls_preds = []
        all_reg_preds = []
        all_anchors = []
        for level, spatial_feat in enumerate(spatial_feats):
            # transformer feature upsample
            feat_fusion_up = F.interpolate(hs, scale_factor=2 ** (2 - level))
            # print(feat_fusion_up.shape, spatial_feat.shape)
            fused_spa_hs = torch.cat([feat_fusion_up, self.dim_converter_1[level](spatial_feat).squeeze(2)], axis=1)
            fused_spa_hs = self.dim_converter_2[level](fused_spa_hs)

            # two sets convs to decompose cls and bbox
            cls_feat = self.cls_conv_sets[level](fused_spa_hs)
            reg_feat = self.reg_conv_sets[level](fused_spa_hs)

            # heads
            cls_feat, reg_feat = self.heads[level](cls_feat, reg_feat)
            
            # pred
            conf_pred = self.conf_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)

            # generate anchors
            fmp_size = conf_pred.shape[-2:]
            anchors = self.generate_anchors(fmp_size, self.stride[level])

            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            all_conf_preds.append(conf_pred)
            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_anchors.append(anchors)

        batch_scores = []
        batch_labels = []
        batch_bboxes = []
        for batch_idx in range(conf_pred.size(0)):
            # [B, M, C] -> [M, C]
            cur_conf_preds = []
            cur_cls_preds = []
            cur_reg_preds = []
            for conf_preds, cls_preds, reg_preds in zip(all_conf_preds, all_cls_preds, all_reg_preds):
                # [B, M, C] -> [M, C]
                cur_conf_preds.append(conf_preds[batch_idx])
                cur_cls_preds.append(cls_preds[batch_idx])
                cur_reg_preds.append(reg_preds[batch_idx])

            # post-process
            scores, labels, bboxes = self.post_process_one_hot(
                cur_conf_preds, cur_cls_preds, cur_reg_preds, all_anchors)

            # normalize bbox
            bboxes /= max(224, 224)
            bboxes = bboxes.clip(0., 1.)

            batch_scores.append(scores)
            batch_labels.append(labels)
            batch_bboxes.append(bboxes)

        return batch_scores, batch_labels, batch_bboxes

    def forward(self, src, spatial_feats):
        # src: (sequence_length, batch_size, hidden_dim)
        batch_size = src.shape[1]
        # 准备查询嵌入
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # (num_queries, batch_size, hidden_dim)
        # Transformer 解码器需要将 src 和 tgt 维度为 (sequence_length, batch_size, hidden_dim)
        hs = self.transformer(src=src, tgt=query_embed)  # hs: (num_queries, batch_size, hidden_dim)

        hs = hs.view(7, 7, batch_size, -1).permute(2,3,0,1)

        all_conf_preds = []
        all_cls_preds = []
        all_box_preds = []
        all_anchors = []
        for level, spatial_feat in enumerate(spatial_feats):
            # print("B", spatial_feat.shape)
            # transformer feature upsample
            feat_fusion_up = F.interpolate(hs, scale_factor=2 ** (2 - level))
            # print("A", feat_fusion_up.shape, self.dim_converter_1[level](spatial_feat).shape)

            fused_spa_hs = torch.cat([feat_fusion_up, self.dim_converter_1[level](spatial_feat).squeeze(2)], axis=1)
            fused_spa_hs = self.dim_converter_2[level](fused_spa_hs)
            # print(next(self.cls_conv_sets[level].parameters()).device)
            # two sets convs to decompose cls and bbox
            cls_feat = self.cls_conv_sets[level](fused_spa_hs)
            reg_feat = self.reg_conv_sets[level](fused_spa_hs)
            # print(cls_feat.shape)
            # print("C", cls_feat.shape, reg_feat.shape)

            # head
            cls_feat, reg_feat = self.heads[level](cls_feat, reg_feat)
                
            # pred
            conf_pred = self.conf_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)

            # generate anchors
            fmp_size = conf_pred.shape[-2:]
            anchors = self.generate_anchors(fmp_size, self.stride[level])

            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            # decode box: [M, 4]
            box_pred = self.decode_boxes(anchors, reg_pred, self.stride[level])

            all_conf_preds.append(conf_pred)
            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)

        # output dict
        outputs = {"pred_conf": all_conf_preds,       # List(Tensor) [B, M, 1]
                    "pred_cls": all_cls_preds,         # List(Tensor) [B, M, C]
                    "pred_box": all_box_preds,         # List(Tensor) [B, M, 4]
                    "anchors": all_anchors,            # List(Tensor) [B, M, 2]
                    "strides": self.stride}            # List(Int)
        return outputs

# 整合模型
class AVIVDNetV5_7_1(nn.Module):

    def __init__(self, num_classes, embed_dim=256, num_heads=8, num_layers=6, num_queries=49):
        super(AVIVDNetV5_7_1, self).__init__()
        ########## v model ##########
        self.backbone_3d = get_model(width_mult=1., multi_scale=True)
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

        self.backbone_2d = timm.create_model('mobilenetv3_small_100', pretrained=True, features_only=True)
        self.backbone_2d.global_pool=nn.Identity()
        self.backbone_2d.conv_head=nn.Identity()
        self.backbone_2d.act2=nn.Identity()
        self.backbone_2d.flatten=nn.Identity()
        self.backbone_2d.classifier=nn.Identity()
        ########## v model ##########
        # print(self.backbone_3d)
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
    

    def forward(self, video, audio, trainable):
        # print("here", self.trainable)
        nB=audio.shape[0]
        nC=audio.shape[1]
        # print(audio.shape)
        audio = audio.unsqueeze(2).repeat(1,1,3,1,1).view(-1, 3, 128, 469)
        audio_feat = self.a_model(audio)     # (batch_size, embed_dim)
        
        audio_feat = audio_feat.reshape(nB, -1, 4, 15)
        visual_feats = self.backbone_3d(video)   # (batch_size, embed_dim)
        # print(len(visual_feats))
        # print(visual_feats[0].shape, visual_feats[1].shape, visual_feats[2].shape)
        # 7 x 7
        visual_feat = visual_feats[2].squeeze(2)
        
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
        print(transformed_feat.shape)
        # 使用融合后的特征进行检测
        # print(self.trainable)
        if not trainable:
            return self.detr.inference(transformed_feat, visual_feats)
        output = self.detr(transformed_feat, visual_feats)  # 注意，这里 src 是 (sequence_length, batch_size, embed_dim)
        # print(output)
        return output


if __name__ == '__main__':
    nB=10
    v_clip=torch.randn([nB,3,16,224,224]).cuda()
    audio=torch.randn([nB, 6, 128, 469]).cuda()
    # # mic_meta={0:(1,2), 1:(322,435), 2:(234,432), 3:(110,128), 4:(123, 42), 5:(123, 33)}
    # # mic_meta = {k: v.to(device='gpu', non_blocking=True) for k, v in mic_meta.items()}

    model=AVIVDNetV5_7_1(num_classes=3, ).cuda()
    # print(model)
    # model.load_weights(True)
    # # model=model.cuda()
    # # print(model)
    out=model(v_clip, audio, True)
    # print(model)
    print(out['pred_conf'][0].shape, out['pred_cls'][0].shape, out['pred_box'][0].shape, out['anchors'][0].shape)
    print(out['pred_conf'][1].shape, out['pred_cls'][1].shape, out['pred_box'][1].shape, out['anchors'][1].shape)
    print(out['pred_conf'][2].shape, out['pred_cls'][2].shape, out['pred_box'][2].shape, out['anchors'][2].shape)
    
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



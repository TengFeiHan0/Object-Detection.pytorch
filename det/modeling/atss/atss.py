import math
from typing import List, Dict
import torch
import torch.nn.functional as F
from torch import nn
from det.layers import DFConv2d
from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.anchor_generator import build_anchor_generator
from .atss_outputs import ATSSOutputs
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from det.modeling.rpn_utils import (Scale, ModuleListDial, BoxCoder, ATSSAnchorGenerator)
from detectron2.structures import Instances, Boxes
__all__ = ["ATSS"]

INF = 100000000

class ATSSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.ATSS.NUM_CLASSES
        
        self.num_anchors = len(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS) * cfg.MODEL.ANCHOR_GENERATOR.SCALES_PER_OCTAVE

        head_configs = {"cls": (cfg.MODEL.ATSS.NUM_CLS_CONVS,
                                cfg.MODEL.ATSS.USE_DEFORMABLE),
                        "bbox": (cfg.MODEL.ATSS.NUM_BOX_CONVS,
                                 cfg.MODEL.ATSS.USE_DEFORMABLE),
                        "share": (cfg.MODEL.ATSS.NUM_SHARE_CONVS,
                                  False)}
        norm = None if cfg.MODEL.ATSS.NORM == "none" else cfg.MODEL.ATSS.NORM
        self.num_levels = len(input_shape)

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:#是否使用dcn
                    conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d
                tower.append(conv_func(
                    in_channels, in_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm == "NaiveGN":
                    tower.append(NaiveGroupNorm(32, in_channels))
                elif norm == "BN":
                    tower.append(ModuleListDial([
                        nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)
                    ]))
                elif norm == "SyncBN":
                    tower.append(ModuleListDial([
                        NaiveSyncBatchNorm(in_channels) for _ in range(self.num_levels)
                    ]))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_anchors *self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )#分类
        self.bbox_pred = nn.Conv2d(
            in_channels, self.num_anchors* 4, kernel_size=3,
            stride=1, padding=1
        )#box预测
        self.ctrness = nn.Conv2d(
            in_channels, self.num_anchors* 1, kernel_size=3,
            stride=1, padding=1
        )#中心度

        if cfg.MODEL.ATSS.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])
        else:
            self.scales = None

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower, self.cls_logits,
            self.bbox_pred, self.ctrness
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.ATSS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        
        if cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
            assert num_anchors == 1, "regressing from a point only support num_anchors == 1"
            torch.nn.init.constant_(self.bbox_pred.bias, 4)
            
            
    def forward(self, x):
        logits = []
        bbox_reg = []
        ctrness = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)#共享的feature
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved ATSS, instead of exp.
            bbox_reg.append(F.relu(reg))
            
        return logits, bbox_reg, ctrness



@PROPOSAL_GENERATOR_REGISTRY.register()
class ATSS(nn.Module):
    """
    Implement ATSS.
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.in_features          = cfg.MODEL.ATSS.IN_FEATURES
        self.focal_loss_alpha     = cfg.MODEL.ATSS.LOSS_ALPHA
        self.focal_loss_gamma     = cfg.MODEL.ATSS.LOSS_GAMMA
        
        self.pre_nms_thresh_train = cfg.MODEL.ATSS.INFERENCE_TH_TRAIN
        self.pre_nms_thresh_test  = cfg.MODEL.ATSS.INFERENCE_TH_TEST
        self.pre_nms_topk_train   = cfg.MODEL.ATSS.PRE_NMS_TOPK_TRAIN
        self.pre_nms_topk_test    = cfg.MODEL.ATSS.PRE_NMS_TOPK_TEST
        self.nms_thresh           = cfg.MODEL.ATSS.NMS_TH
        self.post_nms_topk_train  = cfg.MODEL.ATSS.POST_NMS_TOPK_TRAIN
        self.post_nms_topk_test   = cfg.MODEL.ATSS.POST_NMS_TOPK_TEST
        self.thresh_with_ctr      = cfg.MODEL.ATSS.THRESH_WITH_CTR
        
        self.atss_head = ATSSHead(cfg, [input_shape[f] for f in self.in_features])
        
        #atss
        octave = cfg.MODEL.ANCHOR_GENERATOR.OCTAVE
        scales_per_octave = cfg.MODEL.ANCHOR_GENERATOR.SCALES_PER_OCTAVE
        anchor_sizes = cfg.MODEL.ANCHOR_GENERATOR.ANCHOR_SIZES
        self.atss_top_k = cfg.MODEL.ATSS.TOPK
        sizes = []
        for size in anchor_sizes:
            per_layer_anchor_sizes = []
            for scale_per_octave in range(scales_per_octave):
                octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
                per_layer_anchor_sizes.append(octave_scale * size)
            sizes.append(tuple(per_layer_anchor_sizes))
        self.aspect_ratios = cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS 
        self.anchor_strides = cfg.MODEL.ANCHOR_GENERATOR.ANCHOR_STRIDES
        self.anchor_generator = ATSSAnchorGenerator(sizes, self.aspect_ratios, self.anchor_strides)

    def forward(self, images, features, gt_instances=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        # del gt_instances
        features = [features[f] for f in self.in_features]
         
        anchors= self.anchor_generator(features)
            
        
        logits_pred, reg_pred, ctrness_pred = self.atss_head(
            features)
        
        

        if self.training:
            pre_nms_thresh = self.pre_nms_thresh_train
            pre_nms_topk = self.pre_nms_topk_train
            post_nms_topk = self.post_nms_topk_train
        else:
            pre_nms_thresh = self.pre_nms_thresh_test
            pre_nms_topk = self.pre_nms_topk_test
            post_nms_topk = self.post_nms_topk_test

        outputs = ATSSOutputs(
            images,
            logits_pred,
            reg_pred,
            ctrness_pred,
            anchors,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.atss_head.num_classes,
            pre_nms_thresh,
            pre_nms_topk,
            self.nms_thresh,
            post_nms_topk,
            self.thresh_with_ctr,
            self.aspect_ratios, 
            self.anchor_strides,
            self.atss_top_k,
            gt_instances
        )

        results = {}

        if self.training:
            losses, extras = outputs.losses()
        else:
            losses = {}
            with torch.no_grad():
                proposals = outputs.predict_proposals(top_feats)
           
                results = proposals
        return results, losses

  


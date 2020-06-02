from typing import Dict, List
import torch
import math
import torch.nn as nn
from detectron2.layers import DeformConv 
from detectron2.structures import Instances, Boxes
from det.layers import BoundedIoULoss

class FeatureAdaption(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size =3, deformable_groups =4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size* kernel_size*2
        self.conv_feat = nn.Conv2d(
            2, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace = True)
    def init_weights(self):
        torch.nn.init.normal_(self.conv_offset.weight, std=0.01)
        torch.nn.init.normal_(self.conv_adaption.weight, std=0.01)
    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x

@RPN_HEAD_REGISTRY.register()
class GuidedAnchorHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        self.num_classes = cfg.MODEL.GA.NUM_CLASSES
        self.in_channels = in_channels[0]
        self.num_anchors =1 
        self.feat_channels =256
        self.deformable_groups =4
        self.cls_out_channels =self.num_classes 
        self.loc_filter_thr =0.1
        
        
        self.relu = nn.ReLU(inplace=True)
        self.conv_loc = nn.Conv2d(self.in_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.in_channels, self.num_anchors * 2, 1)
        self.feature_adaption = FeatureAdaption(
            self.in_channels,
            self.feat_channels,
            kernel_size=3,
            deformable_groups=self.deformable_groups)
        #!use maskconv to replace it
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                     self.num_anchors * self.cls_out_channels,
                                     1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4,
                                     1)
       
        self.feature_adaption.init_weights()
        for modules in [
            self.conv_cls, self.conv_reg,
            self.conv_shape, self.conv_loc
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
                    
        prior_prob = cfg.MODEL.GA.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.conv_loc.bias, bias_value)
    
    def forward(self, features):
        loc_preds =[]
        shape_preds = []
        bbox_preds = []
        cls_score = []
        for x in features:
            loc = self.conv_loc(x)
            loc_preds.append(loc)
            shape = self.conv_shape(x)
            shape_preds.append(shape)
            x = self.feature_adaption(x, shape)
            # masked conv is only used during inference for speed-up
            if not self.training:
                mask = loc.sigmoid()[0] >= self.loc_filter_thr
            else:
                mask = None
            cls_score.append(self.conv_cls(x, mask))
            bbox_preds.apend(self.conv_reg(x, mask))
        return cls_score, bbox_preds, shape_preds, loc_preds

  
@PROPOSAL_GENERATOR_REGISTRY.register()  
class GuidedAnchoring(nn.Module):
    
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        
        #self.loss_loc = SigmoidFocalLoss()
        self.loss_shape = BoundedIoULoss(beta=0.2, loss_weight =1.0)
        
        self.anchor_generator = build_anchor_generator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features])
        
        self.anchor_scale =
        self.anchor_strides = 
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH
        self.center_ratio = cfg.MODEL.CENTER_RATIO
        self.ignore_ratio = cfg.MODEL.IGNORE_RATIO 
        self.use_loc_filter = cfg.MODEL.USE_LOC_FILTER
        self.loc_filter_thr = cfg.MODEL.LOC_FILTER_THRESH
    def forward(self, images, features, gt_instances=None):
        
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        del gt_instances
        features = [features[f] for f in self.in_features]
        cls_score, bbox_pred, shape_pred, loc_pred = self.rpn_head(features)
        
      
    def ga_sampled_approxs(self, images, features, gt_instances):
        
        approxs = self.approx_anchor_generator(features)
        approxs = Boxes.cat(approxs)
        image_sizes = [x.image_size for x in gt_instances]
        inside_flags_list = []
        for image_size_i in image_sizes:
            if self.boundary_threshold >= 0:
                inside_flags = approxs.inside_box(image_size_i, self.boundary_threshhold)
                inside_flags_list.append(inside_flags)
        inside_flags = (
                    torch.stack(inside_flags_list, 0).sum(dim=0) > 0)        
        
        return approxs, inside_flags_list
    
    def ga_loc_targets(self, features, gt_instances):
        
        gt_bboxes_list = [x.gt_boxes.tensor for x in gt_instances]
        
        for stride in self.anchor_strides:
            assert (stride[0] == stride[1])
        anchor_strides = [stride[0] for stride in anchor_strides]
        device = gt_instances.gt_boxes[0].device
        
        all_loc_targets = []
        all_loc_weights = []
        all_ignore_map = []
    
        r1 = (1 - self.center_ratio) / 2
        r2 = (1 - self.ignore_ratio) / 2
        img_per_gpu = len(gt_bboxes_list)
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
           
            loc_targets = torch.zeros(img_per_gpu, 1, h, w, device=device, dtype = torch.float32)
            loc_weights = torch.full_like(loc_targets, -1)
            ignore_map = torch.zero_like(loc_targets)
            
            all_loc_targets.append(loc_targets)
            all_loc_weights.append(loc_weights)
            all_ignore_map.append(ignore_map)
            
        for img_id in range(img_per_gpu):
            gt_bboxes = gt_bboxes_list[img_id]
            scale = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) *
                               (gt_bboxes[:, 3] - gt_bboxes[:, 1]))
            min_anchor_size = scale.new_full(
                (1, ), float(self.anchor_scale * anchor_strides[0]))
            # assign gt bboxes to different feature levels w.r.t. their scales
            target_lvls = torch.floor(
                torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)
            target_lvls = target_lvls.clamp(min=0, max=len(features) - 1).long()
            for gt_id in range(gt_bboxes.size(0)):
                lvl = target_lvls[gt_id].item()
                # rescaled to corresponding feature map
                gt_ = gt_bboxes[gt_id, :4] / anchor_strides[lvl]
                # calculate ignore regions
                ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(
                    gt_, r2, featmap_sizes[lvl])
                # calculate positive (center) regions
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = calc_region(
                    gt_, r1, featmap_sizes[lvl])
                all_loc_targets[lvl][img_id, 0, ctr_y1:ctr_y2 + 1,
                                     ctr_x1:ctr_x2 + 1] = 1
                all_loc_weights[lvl][img_id, 0, ignore_y1:ignore_y2 + 1,
                                     ignore_x1:ignore_x2 + 1] = 0
                all_loc_weights[lvl][img_id, 0, ctr_y1:ctr_y2 + 1,
                                     ctr_x1:ctr_x2 + 1] = 1
                # calculate ignore map on nearby low level feature
                if lvl > 0:
                    d_lvl = lvl - 1
                    # rescaled to corresponding feature map
                    gt_ = gt_bboxes[gt_id, :4] / anchor_strides[d_lvl]
                    ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(
                        gt_, r2, featmap_sizes[d_lvl])
                    all_ignore_map[d_lvl][img_id, 0, ignore_y1:ignore_y2 + 1,
                                          ignore_x1:ignore_x2 + 1] = 1
                # calculate ignore map on nearby high level feature
                if lvl < len(features) - 1:
                    u_lvl = lvl + 1
                    # rescaled to corresponding feature map
                    gt_ = gt_bboxes[gt_id, :4] / anchor_strides[u_lvl]
                    ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(
                        gt_, r2, featmap_sizes[u_lvl])
                    all_ignore_map[u_lvl][img_id, 0, ignore_y1:ignore_y2 + 1,
                                          ignore_x1:ignore_x2 + 1] = 1
            
        for lvl_id in range(len(features)):
                # ignore negative regions w.r.t. ignore map
            all_loc_weights[lvl_id][(all_loc_weights[lvl_id] < 0)
                                    & (all_ignore_map[lvl_id] > 0)] = 0
            # set negative regions with weight 0.1
            all_loc_weights[lvl_id][all_loc_weights[lvl_id] < 0] = 0.1
        
        return all_loc_weights, all_loc_weights         
               
    def ga_shape_targets(self, approxs, inside_flags, squares, gt_instances):
        assert len(approxs) == len(inside_flags) == len(squares)
        approxs_flatten = Boxes.cat(approxs)
        inside_flags_flatten = torch.cat(inside_flags)
        squares_flatten = Boxes.cat(squares)
          
    def get_anchors(self, features, shape_preds, loc_preds):
        
        squares_list = self.square_anchor_generator()
        
        guided_anchor_list = []
        mask_list = []       
        for level in range(len(features)):
            squares = squares_list[level]
            shape_pred = shape_preds[level]
            loc_pred = loc_preds[level].sigmoid().detach()
            if self.use_loc_filter:
                loc_mask = loc_pred >= self.loc_filter_thr 
            else:
                loc_mask = loc_pred >=0 
            mask = loc_mask.permute(1, 2, 0).expand(-1, -1, self.num_anchors)
            mask = mask.contiguous().view(-1)
            squares = squares[mask]
            anchor_deltas = shape_pred.permute(1, 2, 0).contiguous().view(-1, 2).detach()[mask]
            bbox_deltas = anchor_deltas.new_full(squares.size(), 0) 
            bbox_deltas[:, 2:] = anchor_deltas
            guided_anchors = self.anchor_coder.decode(
                squares, bbox_deltas    
            )
            guided_anchor_list.append(guided_anchors)
            mask_list.append(loc_mask)
        return squares_list, guided_anchor_list, mask_list   
            
                   
            
            
        
            
            
               






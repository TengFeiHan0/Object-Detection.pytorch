import logging
import torch
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes, pairwise_iou
from detectron2.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit

from det.utils.comm import reduce_sum
from det.layers import ml_nms

from det.modeling.rpn_utils import BoxCoder, permute_and_flatten, concat_box_prediction_layers

logger = logging.getLogger(__name__)

INF = 100000000

class ATSSOutputs(object):
    def __init__(
            self,
            images,
            box_cls, 
            box_regression, 
            centerness,
            anchors,
            focal_loss_alpha,
            focal_loss_gamma,
            num_classes,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            thresh_with_ctr,
            aspect_ratios,
            anchor_strides,
            atss_top_k,
            gt_instances
    ):
        self.box_cls = box_cls
        self.box_regression = box_regression
        self.centerness = centerness
        self.anchors = anchors
       
        self.gt_instances = gt_instances
        self.num_feature_maps = len(self.box_cls)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
      
        self.num_classes = num_classes
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.thresh_with_ctr = thresh_with_ctr
        self.aspect_ratios = aspect_ratios
        self.anchor_strides = anchor_strides
        self.atss_top_k = atss_top_k
        self.box_coder = BoxCoder()
    
    def prepare_targets(self, targets, anchors):
        cls_labels = []
        reg_targets = []
        target_inds = []
        
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes_per_im = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes
           
            anchors_per_im = Boxes.cat(anchors)
            num_gt = bboxes_per_im.shape[0]
        
            num_anchors_per_loc = len(self.aspect_ratios)
            num_anchors_per_level = [len(anchors_per_level) for anchors_per_level in anchors_per_im]
            ious = pairwise_iou(anchors_per_im, targets_per_im.gt_boxes)

            gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
            gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)

            anchors_cx_per_im = (anchors_per_im.tensor[:, 2] + anchors_per_im.tensor
                                 [:, 0]) / 2.0
            anchors_cy_per_im = (anchors_per_im.tensor[:, 3] + anchors_per_im.tensor[:, 1]) / 2.0
            anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

            distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

            # Selecting candidates based on the center distance between anchor box and object
            candidate_idxs = []
            star_idx = 0
            for level, anchors_per_level in enumerate(anchors):
                end_idx = star_idx + num_anchors_per_level[level]
                distances_per_level = distances[star_idx:end_idx, :]
                
                topk = min(self.atss_top_k  * num_anchors_per_loc, num_anchors_per_level[level])
                _, topk_idxs_per_level = distances_per_level.topk(topk,dim=0, largest=False)
                candidate_idxs.append(topk_idxs_per_level + star_idx)
                star_idx = end_idx
            candidate_idxs = torch.cat(candidate_idxs, dim=0)

            # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
            candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

            # Limiting the final positive samples’ center to object
            anchor_num = anchors_cx_per_im.shape[0]
            for ng in range(num_gt):
                candidate_idxs[:, ng] += ng * anchor_num
            e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            candidate_idxs = candidate_idxs.view(-1)
            l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 0]
            t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
            r = bboxes_per_im[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
            b = bboxes_per_im[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
            is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
            is_pos = is_pos & is_in_gts

            # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
            ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
            index = candidate_idxs.view(-1)[is_pos.view(-1)]
            ious_inf[index] = ious.t().contiguous().view(-1)[index]
            ious_inf = ious_inf.view(num_gt, -1).t()

            anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
            cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
            
            cls_labels_per_im[anchors_to_gt_values == INF] = self.num_classes
            
            matched_gts = bboxes_per_im[anchors_to_gt_indexs]
            
            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.tensor)
         
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)
            
        return cls_labels, reg_targets    

        
    def losses(self):
        
        labels, reg_targets = self.prepare_targets(self.gt_instances, self.anchors)
          
        N = len(labels)
        
        box_cls_flatten, box_regression_flatten = concat_box_prediction_layers(self.box_cls, self.box_regression)
        centerness_flatten = [ct.permute(0, 2, 3, 1).reshape(N, -1, 1) for ct in self.centerness]
        centerness_flatten = torch.cat(centerness_flatten, dim=1).reshape(-1)      
        labels_flatten = torch.cat(labels, dim=0)   
        reg_targets_flatten = torch.cat(reg_targets, dim=0)
        
        anchors =Boxes.cat(self.anchors)
        anchors_flatten = torch.cat([anchors.tensor], dim=0)
        
       
        pos_inds = torch.nonzero(labels_flatten >0).squeeze(1)
        
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
           
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)
        
        class_target = torch.zeros_like(box_cls_flatten)
        class_target[pos_inds, labels_flatten[pos_inds]]=1
       

        class_loss = sigmoid_focal_loss_jit(
            box_cls_flatten,
            class_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg
        # 根据pos_inds提取正样本
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
      
        anchors_flatten = anchors_flatten[pos_inds]
       
        centerness_flatten = centerness_flatten[pos_inds]
        
        centerness_targets = self.compute_centerness_targets(reg_targets_flatten, anchors_flatten)
        
        loss_denorm = reduce_sum(centerness_targets.sum()).item() / float(num_gpus)
         
        if pos_inds.numel() > 0:
            reg_loss = self.GIoULoss(box_regression_flatten, reg_targets_flatten, anchors_flatten,
                                     weight=centerness_targets) / loss_denorm
            centerness_loss = F.binary_cross_entropy_with_logits(centerness_flatten, centerness_targets) / num_pos_avg
        else:
            reg_loss = box_regression_flatten.sum()*0
            centerness_loss = centerness_flatten.sum()*0
            
       
        
        losses = {
        "loss_fcos_cls": class_loss,
        "loss_fcos_loc": reg_loss,
        "loss_fcos_ctr": centerness_loss
        }
        extras = {
            "pos_inds": pos_inds,
            "gt_ctr": centerness_targets,
            "loss_denorm": loss_denorm
        }   

        return losses, extras
        
    def compute_centerness_targets(self, reg_targets, anchors):
        gts = self.box_coder.decode(reg_targets, anchors)
        
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l = anchors_cx - gts[:, 0]
        t = anchors_cy - gts[:, 1]
        r = gts[:, 2] - anchors_cx
        b = gts[:, 3] - anchors_cy
        left_right = torch.stack([l, r], dim=1)
        top_bottom = torch.stack([t, b], dim=1)
        centerness = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
       
        assert not torch.isnan(centerness).any()
        return centerness

    def GIoULoss(self, pred, target, anchor, weight=None):
       
        pred_boxes = self.box_coder.decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        gt_boxes = self.box_coder.decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()
        
    def predict_proposals(self):
        sampled_boxes = []
        anchors = list(zip(*anchors))
        for _, (o, b, c, a) in enumerate(zip(self.box_cls, self.box_regression, self.centerness, anchors)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(o, b, c, a)
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        
        return boxlists    
        
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
    
    def forward_for_single_feature_map(self, box_cls, box_regression, centerness, anchors):
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A

        # put in the same format as anchors
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)
        
        centerness = permute_and_flatten(centerness, N, A, 1, H, W)
        centerness = centerness.reshape(N, -1).sigmoid()

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)


        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for per_box_cls, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_anchors \
                in zip(box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors):

            per_box_cls = per_box_cls[per_candidate_inds]

            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)

            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.gt_boxes.tensor[per_box_loc, :].view(-1, 4)
            )

            boxlist = Instances(per_anchors.size)
            boxlist.pred_boxes = Boxes(detections)
            boxlist.pred_classes = per_class
            boxlist.scores = torch.sqrt(per_box_cls)
            results.append(boxlist)

        return results    
        
        
   
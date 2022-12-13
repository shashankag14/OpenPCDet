import math
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from ...ops.iou3d_nms import iou3d_nms_utils

class PreLossSampler(nn.Module):
    def __init__(self, pred_sampler_cfg):
        super().__init__()
        self.pred_sampler_cfg = pred_sampler_cfg

    # Localization-based samplers ======================================================================================
    # Should be focused more since the localization loss is x3 significant than the cls loss!

    def bbox_uncertainty_sampler(self, **kwargs):
        raise NotImplementedError

    '''
    Apply NMS on rcnn_cls_labels, then perform top-k sampling on decayed scores
    '''
    def gt_nms_sampler(self, forward_ret_dict, index):
        reg_valid_mask = forward_ret_dict['reg_valid_mask'][index].clone().detach()
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'][index].clone().detach()
        gt_boxes = forward_ret_dict['gt_of_rois_src'][index]
        rcnn_cls_preds = forward_ret_dict['rcnn_cls'].view_as(forward_ret_dict['rcnn_cls_labels'])[index].clone().detach()
        rcnn_cls_preds = torch.sigmoid(rcnn_cls_preds).unsqueeze(0)
        
        # ----------- REG_VALID_MASK -----------
        reg_fg_thresh = self.pred_sampler_cfg.UNLABELED_REG_FG_THRESH
        filtering_mask = (rcnn_cls_preds > reg_fg_thresh) & (rcnn_cls_labels > reg_fg_thresh)
        reg_valid_mask = filtering_mask.long()

        # ----------- RCNN_CLS_LABELS -----------
        sampled_inds = torch.zeros_like(rcnn_cls_labels, dtype=torch.bool)

        nms_type = getattr(iou3d_nms_utils, self.pred_sampler_cfg.NMS_CONFIG.NMS_TYPE)
        nms_inds, _ = nms_type(gt_boxes[:, 0:7], rcnn_cls_labels, 
                            self.pred_sampler_cfg.NMS_CONFIG.NMS_THRESH, **self.pred_sampler_cfg.NMS_CONFIG)
        
        if self.pred_sampler_cfg.NMS_CONFIG.NMS_TYPE == 'soft_nms':
            keep_inds = nms_inds[:self.pred_sampler_cfg.NMS_CONFIG.NMS_POST_MAXSIZE].long()
            sampled_inds[keep_inds] = True
        else:
            sampled_inds[nms_inds] = True
        
        fg_mask = rcnn_cls_labels > self.pred_sampler_cfg.CLS_FG_THRESH
        bg_mask = rcnn_cls_labels < self.pred_sampler_cfg.CLS_BG_THRESH
        ignore_mask = (~sampled_inds | torch.eq(gt_boxes, 0).all(dim=-1))
        rcnn_cls_labels[fg_mask] = 1
        rcnn_cls_labels[bg_mask] = 0
        rcnn_cls_labels[ignore_mask] = -1

        return reg_valid_mask, rcnn_cls_labels

    # Confidence-based samplers ========================================================================================
    '''
    Ignore rcnn_cls_labels based on semantic scores, sample based on objectness scrores 
    '''
    def classwise_hybrid_thresholds_sampler(self, forward_ret_dict, index):
        # (mis?) using pseudo-label objectness scores as a proxy for iou!
        reg_valid_mask = forward_ret_dict['reg_valid_mask'][index].clone().detach()
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'][index].clone().detach()
        gt_boxes = forward_ret_dict['gt_of_rois_src'][index]
        roi_scores = torch.sigmoid(forward_ret_dict['roi_scores'][index])
        rcnn_cls_preds = forward_ret_dict['rcnn_cls'].view_as(forward_ret_dict['rcnn_cls_labels'])[index].clone().detach()
        rcnn_cls_preds = torch.sigmoid(rcnn_cls_preds).unsqueeze(0)
        
        # ----------- REG_VALID_MASK -----------
        reg_fg_thresh = self.pred_sampler_cfg.UNLABELED_REG_FG_THRESH
        filtering_mask = (rcnn_cls_preds > reg_fg_thresh) & (rcnn_cls_labels > reg_fg_thresh)
        reg_valid_mask = filtering_mask.long()

        # ----------- RCNN_CLS_LABELS -----------
        sampled_inds = torch.zeros_like(rcnn_cls_labels, dtype=torch.bool)

        # Ignore rcnn_cls_labels based on semantic scores, assign hard labels based on objectness scrores
        thresh_inds = (roi_scores > 0.2).nonzero().view(-1)
        subsampled_inds = self.subsample_preds(max_overlaps=rcnn_cls_labels[thresh_inds], preds_per_image = len(thresh_inds))
        sampled_inds[thresh_inds[subsampled_inds]] = True
        
        fg_mask = rcnn_cls_labels > self.pred_sampler_cfg.CLS_FG_THRESH
        bg_mask = rcnn_cls_labels < self.pred_sampler_cfg.CLS_BG_THRESH
        ignore_mask = (~sampled_inds | torch.eq(gt_boxes, 0).all(dim=-1))
        rcnn_cls_labels[fg_mask] = 1
        rcnn_cls_labels[bg_mask] = 0
        rcnn_cls_labels[ignore_mask] = -1

        return reg_valid_mask, rcnn_cls_labels

    '''
    # Sample based on Teacher's (pseudo boxes) objectness score using Flexmatch idea 
    # This sampling would be based on classwise thresholds which would be flexible i.e. in case the number of PLs
    #  for a particular class is less, we would lower the threshold while for dominant class in PLs, the threshold would be 
    #  set higher - in order to pass balanced unlabeled data taking into accoutn the learning difficulties of diff classes
    Adapted from : https://github.com/TorchSSL/TorchSSL
    '''
    def classwise_adapative_thresholds_sampler(self, forward_ret_dict, index):
        reg_valid_mask = forward_ret_dict['reg_valid_mask'][index].clone().detach()
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'][index].clone().detach()
        gt_boxes = forward_ret_dict['gt_of_rois_src'][index]
        gt_labels = gt_boxes[:, -1].long()
        rcnn_cls_preds = forward_ret_dict['rcnn_cls'].view_as(forward_ret_dict['rcnn_cls_labels'])[index].clone().detach()
        rcnn_cls_preds = torch.sigmoid(rcnn_cls_preds).unsqueeze(0)
        
        # ----------- REG_VALID_MASK -----------
        reg_fg_thresh = self.pred_sampler_cfg.UNLABELED_REG_FG_THRESH
        filtering_mask = (rcnn_cls_preds > reg_fg_thresh) & (rcnn_cls_labels > reg_fg_thresh)
        reg_valid_mask = filtering_mask.long()

        # ----------- RCNN_CLS_LABELS -----------
        sampled_inds = torch.zeros_like(rcnn_cls_labels, dtype=torch.bool)

        fixed_thresh = self.pred_sampler_cfg.UNLABELED_REG_FG_THRESH
        selected_labels = torch.ones((len(gt_labels),), dtype=torch.long, ).cuda() * -1
        classwise_acc = torch.zeros((3,)).cuda()
        
        # Use fixed threshold to estimate the learning status of each class, assign remaining as BG
        select = rcnn_cls_labels.ge(fixed_thresh).long()
        selected_labels[select == 1] = gt_labels[select == 1].long()

        # Counter for classwise learning based on fixed threshold 
        pseudo_counter = Counter(selected_labels.tolist())
        if max(pseudo_counter.values()) < len(gt_labels):  
            if self.pred_sampler_cfg.UNLABELED_SAMPLER_THRESH_WARMUP:
                for i in range(3):
                    classwise_acc[i] = pseudo_counter[i+1] / max(pseudo_counter.values())
            else:
                wo_negative_one = pseudo_counter.clone().detach()
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for i in range(3):
                    classwise_acc[i] = pseudo_counter[i+1] / max(wo_negative_one.values())

        sampled_inds = rcnn_cls_labels.ge(fixed_thresh * classwise_acc[gt_labels-1]).bool()  # linear
        # Non linear mapping function based on Flexmatch sec3.3
        # sampled_inds = gt_scores.ge(fixed_thresh * (classwise_acc[gt_labels-1] / (2. - classwise_acc[gt_labels-1]))).long()  # convex
        
        fg_mask = rcnn_cls_labels > self.pred_sampler_cfg.CLS_FG_THRESH
        bg_mask = rcnn_cls_labels < self.pred_sampler_cfg.CLS_BG_THRESH
        ignore_mask = (~sampled_inds | torch.eq(gt_boxes, 0).all(dim=-1))
        rcnn_cls_labels[fg_mask] = 1
        rcnn_cls_labels[bg_mask] = 0
        rcnn_cls_labels[ignore_mask] = -1
            
        return reg_valid_mask, rcnn_cls_labels

    '''
    Increased TEST:NMS_POST_MAXSIZE form 100->256 to perform this sampling.
    This method fetches the classwise top-k indices based on gt_scores (teacher's rcnn cls preds). 
    It further truncates those classwise top-k inds based on the proportion of rois predicted for each of those classes originally.
    '''
    def classwise_top_k_sampler(self, forward_ret_dict, index):
        reg_valid_mask = forward_ret_dict['reg_valid_mask'][index].clone().detach()
        roi_labels = forward_ret_dict['roi_labels'][index]       
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'][index].clone().detach()
        gt_boxes = forward_ret_dict['gt_of_rois_src'][index]
        rcnn_cls_preds = forward_ret_dict['rcnn_cls'].view_as(forward_ret_dict['rcnn_cls_labels'])[index].clone().detach()
        rcnn_cls_preds = torch.sigmoid(rcnn_cls_preds).unsqueeze(0)
        
        # ----------- REG_VALID_MASK -----------
        reg_fg_thresh = self.pred_sampler_cfg.UNLABELED_REG_FG_THRESH
        filtering_mask = (rcnn_cls_preds > reg_fg_thresh) & (rcnn_cls_labels > reg_fg_thresh)
        reg_valid_mask = filtering_mask.long()

        # ----------- RCNN_CLS_LABELS -----------
        sampled_inds = torch.zeros_like(rcnn_cls_labels, dtype=torch.bool)

        classwise_topk_inds = []
        class_proportion_list = []
        for k in range(1,4):  # TODO(Farzad) fixed num class
            roi_mask = (roi_labels == k)
            if roi_mask.sum() > 0:
                cur_gt_scores = rcnn_cls_labels[roi_mask]
                cur_inds = roi_mask.nonzero().view(-1)
                _, top_k_inds = torch.topk(cur_gt_scores, k=min(len(roi_labels), len(cur_inds)))  # TODO(Farzad) fixed k

                # Compute proportionate number of rois to be sampled for each class
                class_proportion = math.ceil((len(cur_inds) / len(roi_labels)) * self.pred_sampler_cfg.PREDS_PER_IMAGE)
                class_proportion_list.append(class_proportion)
                # Store sampled classwise top-k inds as per the proportion
                classwise_topk_inds.append(cur_inds[top_k_inds][:class_proportion])  

        # total sampled rois might exceed self.pred_sampler_cfg.PREDS_PER_IMAGE due to ceil operator 
        # if so, remove extra rois from the class with maximum rois
        total_rois = sum(class_proportion_list)
        extra_rois = total_rois - self.pred_sampler_cfg.PREDS_PER_IMAGE 
        if extra_rois != 0:
            max_rois_class = class_proportion_list.index(max(class_proportion_list))
            classwise_topk_inds[max_rois_class] = classwise_topk_inds[max_rois_class][:-extra_rois]
        
        keep_inds = torch.cat(classwise_topk_inds)
        sampled_inds[keep_inds] = True

        # filter GT labels based on FG/BG thresholds 
        iou_fg_thresh = self.pred_sampler_cfg.CLS_FG_THRESH
        iou_bg_thresh = self.pred_sampler_cfg.CLS_BG_THRESH
        fg_mask = rcnn_cls_labels > iou_fg_thresh
        bg_mask = rcnn_cls_labels < iou_bg_thresh
        ignore_mask = (~sampled_inds | torch.eq(gt_boxes, 0).all(dim=-1))
        rcnn_cls_labels[fg_mask] = 1
        rcnn_cls_labels[bg_mask] = 0
        rcnn_cls_labels[ignore_mask] = -1

        return reg_valid_mask, rcnn_cls_labels

    def roi_scores_sampler(self, **kwargs):
        roi_scores = kwargs.get("roi_scores")
        reg_valid_mask = torch.ge(roi_scores, 0.7).long()
        raise NotImplementedError

    '''
    Samples teacher's final predictions(rcnn labels) based on FG:BG ratio
    # TODO (shashank) : Fix appropriate PREDS_PER_IMAGE
    '''
    def gt_score_sampler(self, forward_ret_dict, index):
        reg_valid_mask = forward_ret_dict['reg_valid_mask'][index].clone().detach()
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'][index].clone().detach()
        rcnn_cls_preds = forward_ret_dict['rcnn_cls'].view_as(forward_ret_dict['rcnn_cls_labels'])[index].clone().detach()
        rcnn_cls_preds = torch.sigmoid(rcnn_cls_preds).unsqueeze(0)
        
        # ----------- REG_VALID_MASK -----------
        reg_fg_thresh = self.pred_sampler_cfg.UNLABELED_REG_FG_THRESH
        filtering_mask = (rcnn_cls_preds > reg_fg_thresh) & (rcnn_cls_labels > reg_fg_thresh)
        reg_valid_mask = filtering_mask.long()

        # ----------- RCNN_CLS_LABELS -----------
        sampled_inds = torch.zeros_like(rcnn_cls_labels, dtype=torch.bool)
        
        keep_inds = self.subsample_preds(max_overlaps=rcnn_cls_labels)
        sampled_inds[keep_inds] = True

        fg_mask = rcnn_cls_labels > self.pred_sampler_cfg.CLS_FG_THRESH
        bg_mask = rcnn_cls_labels < self.pred_sampler_cfg.CLS_BG_THRESH
        ignore_mask = (~sampled_inds | torch.eq(forward_ret_dict['gt_of_rois'][index], 0).all(dim=-1)) 
        rcnn_cls_labels[fg_mask] = 1
        rcnn_cls_labels[bg_mask] = 0
        rcnn_cls_labels[ignore_mask] = -1

        return reg_valid_mask, rcnn_cls_labels

    def subsample_preds(self, max_overlaps, preds_per_image=None):
        preds_per_image = self.pred_sampler_cfg.PREDS_PER_IMAGE if preds_per_image is None else preds_per_image
        # sample fg, easy_bg, hard_bg
        fg_preds_per_image = int(np.round(self.pred_sampler_cfg.FG_RATIO * preds_per_image))
        fg_thresh = min(self.pred_sampler_cfg.REG_FG_THRESH, self.pred_sampler_cfg.CLS_FG_THRESH)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)  # > 0.55
        easy_bg_inds = ((max_overlaps < self.pred_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)  # < 0.1
        hard_bg_inds = ((max_overlaps < self.pred_sampler_cfg.REG_FG_THRESH) &
                        (max_overlaps >= self.pred_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        fg_num_preds = fg_inds.numel()
        bg_num_preds = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_preds > 0 and bg_num_preds > 0:
            # sampling fg
            fg_preds_per_this_image = min(fg_preds_per_image, fg_num_preds)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_preds)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_preds_per_this_image]]

            # sampling bg
            bg_preds_per_this_image = preds_per_image - fg_preds_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_preds_per_this_image, self.pred_sampler_cfg.HARD_BG_RATIO
            )

        elif fg_num_preds > 0 and bg_num_preds == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(preds_per_image) * fg_num_preds)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0] # yield empty tensor

        elif bg_num_preds > 0 and fg_num_preds == 0:
            # sampling bg
            bg_preds_per_this_image = preds_per_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_preds_per_this_image, self.pred_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_preds, bg_num_preds))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_preds_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_preds_num = min(int(bg_preds_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_preds_num = bg_preds_per_this_image - hard_bg_preds_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_preds_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_preds_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_preds_num = bg_preds_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_preds_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_preds_num = bg_preds_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_preds_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds
import copy
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pcdet.datasets.augmentor import augmentor_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils

from ...utils import common_utils
from .detector3d_template import Detector3DTemplate
from collections import defaultdict
from.pv_rcnn import PVRCNN
from ...utils.stats_utils import KITTIEvalMetrics, PredQualityMetrics
from torchmetrics.collections import MetricCollection
import torch.distributed as dist
from visual_utils import visualize_utils as V

def _to_dict_of_tensors(list_of_dicts, agg_mode='stack'):
    new_dict = {}
    for k in list_of_dicts[0].keys():
        vals = []
        for i in range(len(list_of_dicts)):
            vals.append(list_of_dicts[i][k])
        agg_vals = torch.cat(vals, dim=0) if agg_mode == 'cat' else torch.stack(vals, dim=0)
        new_dict[k] = agg_vals
    return new_dict


def _to_list_of_dicts(dict_of_tensors, batch_size):
    new_list = []
    for batch_index in range(batch_size):
        inner_dict = {}
        for key in dict_of_tensors.keys():
            assert dict_of_tensors[key].shape[0] == batch_size
            inner_dict[key] = dict_of_tensors[key][batch_index]
        new_list.append(inner_dict)

    return new_list


def _mean_and_var(batch_dict_a, batch_dict_b, unlabeled_inds, keys=()):
    # !!! Note that the function is inplace !!!
    if isinstance(batch_dict_a, dict) and isinstance(batch_dict_b, dict):
        for k in keys:
            batch_dict_mean_k = torch.zeros_like(batch_dict_a[k])
            batch_dict_emas = torch.stack([batch_dict_a[k][unlabeled_inds], batch_dict_b[k][unlabeled_inds]], dim=-1)
            batch_dict_mean_k[unlabeled_inds] = torch.mean(batch_dict_emas, dim=-1)
            batch_dict_a[k + '_mean'] = batch_dict_mean_k
            batch_dict_var_k = torch.zeros_like(batch_dict_a[k])
            batch_dict_var_k[unlabeled_inds] = torch.var(batch_dict_emas, dim=-1)
            batch_dict_a[k + '_var'] = batch_dict_var_k

    elif isinstance(batch_dict_a, list) and isinstance(batch_dict_b, list):
        for ind in unlabeled_inds:
            for k in keys:
                batch_dict_emas = torch.stack([batch_dict_a[ind][k], batch_dict_b[ind][k]], dim=-1)
                batch_dict_a[ind][k + '_mean'] = torch.mean(batch_dict_emas, dim=-1)
                batch_dict_a[ind][k + '_var'] = torch.var(batch_dict_emas, dim=-1)
    else:
        raise TypeError

def _normalize_scores(batch_dict, score_keys=('batch_cls_preds',)):
    # !!! Note that the function is inplace !!!
    assert all([key in ['batch_cls_preds', 'roi_scores'] for key in score_keys])
    for score_key in score_keys:
        if score_key == 'batch_cls_preds':
            if not batch_dict['cls_preds_normalized']:
                batch_dict[score_key] = torch.sigmoid(batch_dict[score_key])
                batch_dict['cls_preds_normalized'] = True
        else:
            batch_dict[score_key] = torch.sigmoid(batch_dict[score_key])

# TODO(farzad) should be tested and debugged
def _weighted_mean(batch_dict_a, batch_dict_b, unlabeled_inds, score_key='batch_cls_preds', keys=()):
    assert score_key in ['batch_cls_preds', 'roi_scores']
    _normalize_scores(batch_dict_a, score_keys=(score_key,))
    _normalize_scores(batch_dict_b, score_keys=(score_key,))
    scores_a = batch_dict_a[score_key][unlabeled_inds]
    scores_b = batch_dict_b[score_key][unlabeled_inds]
    weights = scores_a / (scores_a + scores_b)

    for k in keys:
        batch_dict_mean_k = torch.zeros_like(batch_dict_a[k])
        batch_dict_mean_k[unlabeled_inds] = weights * batch_dict_a[k][unlabeled_inds] + \
                                          (1 - weights) * batch_dict_b[k][unlabeled_inds]
        batch_dict_a[k + '_mean'] = batch_dict_mean_k

# TODO(farzad) should be tested and debugged
def _max_score_replacement(batch_dict_a, batch_dict_b, unlabeled_inds, score_key='batch_cls_preds', keys=()):
    # !!! Note that the function is inplace !!!
    assert score_key in ['batch_cls_preds', 'roi_scores']
    _normalize_scores(batch_dict_a, score_keys=(score_key,))
    _normalize_scores(batch_dict_b, score_keys=(score_key,))
    batch_dict_cat = torch.stack([batch_dict_a[score_key], batch_dict_b[score_key]], dim=-1)
    max_inds = torch.argmax(batch_dict_cat, dim=-1)
    for key in keys:
        batch_dict_a[key][unlabeled_inds] = batch_dict_cat[key][unlabeled_inds, ..., max_inds]

# TODO(farzad) refactor this with global registry, accessible in different places, not via passing through batch_dict
class MetricRegistry(object):
    def __init__(self, **kwargs):
        self._tag_metrics = {}
        self.dataset = kwargs.get('dataset', None)
        self.cls_bg_thresh = kwargs.get('cls_bg_thresh', None)
        self.model_cfg = kwargs.get('model_cfg', None)
    def get(self, tag=None):
        if tag is None:
            tag = 'default'
        if tag in self._tag_metrics.keys():
            metric = self._tag_metrics[tag]
        else:
            kitti_eval_metric = KITTIEvalMetrics(tag=tag, dataset=self.dataset, config=self.model_cfg)
            pred_qual_metric = PredQualityMetrics(tag=tag, dataset=self.dataset, cls_bg_thresh=self.cls_bg_thresh, config=self.model_cfg)
            metric = MetricCollection({"kitti_eval_metric": kitti_eval_metric,
                                       "pred_quality_metric": pred_qual_metric})
            self._tag_metrics[tag] = metric
        return metric

    def tags(self):
        return self._tag_metrics.keys()



class PVRCNN_SSL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # something changes so need deep copy
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)
        self.pv_rcnn = PVRCNN(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.pv_rcnn_ema = PVRCNN(model_cfg=model_cfg_copy, num_class=num_class, dataset=dataset_copy)
        for param in self.pv_rcnn_ema.parameters():
            param.detach_()
        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)
        self.accumulated_itr = 0

        # self.module_list = self.build_networks()
        # self.module_list_ema = self.build_networks()
        self.thresh = model_cfg.THRESH
        self.sem_thresh = model_cfg.SEM_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.no_nms = model_cfg.NO_NMS
        self.supervise_mode = model_cfg.SUPERVISE_MODE
        cls_bg_thresh = model_cfg.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH
        self.metric_registry = MetricRegistry(dataset=self.dataset, model_cfg=model_cfg)
        vals_to_store = ['iou_roi_pl', 'iou_roi_gt', 'pred_scores', 'roi_scores', 'weights', 'class_labels', 'iteration']
        self.val_dict = {val: [] for val in vals_to_store}

    def forward(self, batch_dict):
        if self.training:
            labeled_mask = batch_dict['labeled_mask'].view(-1)
            labeled_inds = torch.nonzero(labeled_mask).squeeze(1).long()
            unlabeled_inds = torch.nonzero(1-labeled_mask).squeeze(1).long()
            batch_dict['unlabeled_inds'] = unlabeled_inds
            batch_dict_ema = {}
            keys = list(batch_dict.keys())
            for k in keys:
                if k + '_ema' in keys:
                    continue
                if k.endswith('_ema'):
                    batch_dict_ema[k[:-4]] = batch_dict[k]
                else:
                    batch_dict_ema[k] = batch_dict[k]

            # Create new dict for weakly aug.(WA) data for teacher - Eg. flip along x axis
            batch_dict_ema_wa = {}
            # If ENABLE_RELIABILITY is True, run WA (Humble Teacher) along with original teacher
            if self.model_cfg['ROI_HEAD'].get('ENABLE_RELIABILITY', False):
                keys = list(batch_dict.keys())
                for k in keys:
                    if k + '_ema_wa' in keys:
                        continue
                    if k.endswith('_ema_wa'):
                        batch_dict_ema_wa[k[:-7]] = batch_dict[k]
                    else:
                        # TODO(farzad) Warning! Here flip_x values are copied from _ema to _ema_wa which is not correct!
                        batch_dict_ema_wa[k] = batch_dict[k]

                with torch.no_grad():
                    self.pv_rcnn_ema.train()
                    for cur_module in self.pv_rcnn_ema.module_list:
                        # Do not use RPN to produce rois for WA image, instead augment (eg. flip)
                        # the proposal coord. of P horizontally to obtain P^
                        try:
                            batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                            if cur_module.model_cfg['NAME'] == 'AnchorHeadSingle':
                                # Use proposals generated from original (non-augmented) input
                                # to pool features generated from weakly-augmented input
                                # Note that the proposals should be (weakly-) augmented before pooling!
                                batch_dict_ema_wa['batch_cls_preds'] = batch_dict_ema[
                                    'batch_cls_preds'].clone().detach()
                                batch_dict_ema_wa['batch_box_preds'] = batch_dict_ema[
                                    'batch_box_preds'].clone().detach()
                                batch_dict_ema_wa['cls_preds_normalized'] = batch_dict_ema['cls_preds_normalized']

                                enable = [1] * len(unlabeled_inds)
                                batch_dict_ema_wa['batch_box_preds'][unlabeled_inds] = augmentor_utils.random_flip_along_x_bbox(
                                    batch_dict_ema_wa['batch_box_preds'][unlabeled_inds],
                                    enables=enable)
                            else:
                                batch_dict_ema_wa = cur_module(batch_dict_ema_wa, disable_gt_roi_when_pseudo_labeling=True)
                        except:
                            # TODO(farzad) we can concat both batch_dict_ema and batch_dict_ema_wa and
                            #  do a forward pass once. Requires more GPU memory, but faster!
                            batch_dict_ema = cur_module(batch_dict_ema)
                            batch_dict_ema_wa = cur_module(batch_dict_ema_wa)
                    # Reverse preds of wa input to match their original (no-aug) preds
                    batch_dict_ema_wa['batch_box_preds'][unlabeled_inds] = augmentor_utils.random_flip_along_x_bbox(
                        batch_dict_ema_wa['batch_box_preds'][unlabeled_inds], [1] * len(unlabeled_inds))

                    # pseudo-labels used for training rpn head
                    pred_dicts_ens = self.ensemble_post_processing(batch_dict_ema, batch_dict_ema_wa, unlabeled_inds,
                                                                   ensemble_option='mean_pre_nms')
            # Else, run the original teacher only
            else:
                with torch.no_grad():
                    for cur_module in self.pv_rcnn_ema.module_list:
                        try:
                            batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                        except:
                            batch_dict_ema = cur_module(batch_dict_ema)
                pred_dicts_ens, recall_dicts_ema = self.pv_rcnn_ema.post_processing(batch_dict_ema, no_recall_dict=True,
                                                                                    override_thresh=0.0,
                                                                                    no_nms_for_unlabeled=self.no_nms)

            # Used for calc stats before and after filtering
            ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_inds, ...]
            if self.model_cfg.ROI_HEAD.get("ENABLE_EVAL", False):
                # PL metrics before filtering
                self.update_metrics(batch_dict, pred_dicts_ens, unlabeled_inds, labeled_inds)

            pseudo_boxes, pseudo_scores, pseudo_sem_scores, pseudo_boxes_var, pseudo_scores_var = \
                self._filter_pseudo_labels(pred_dicts_ens, unlabeled_inds)

            self._fill_with_pseudo_labels(batch_dict, pseudo_boxes, unlabeled_inds, labeled_inds)

            # apply student's augs on teacher's pseudo-labels (filtered) only (not points)
            batch_dict = self.apply_augmentation(batch_dict, batch_dict, unlabeled_inds, key='gt_boxes')

            # if self.model_cfg.ROI_HEAD.get('ENABLE_VIS', False):
            #     for i, uind in enumerate(unlabeled_inds):
            #         mask = batch_dict['points'][:, 0] == uind
            #         point = batch_dict['points'][mask, 1:]
            #         pred_boxes = batch_dict['gt_boxes'][uind][:, :-1]
            #         pred_labels = batch_dict['gt_boxes'][uind][:, -1].int()
            #         pred_scores = torch.zeros_like(pred_labels).float()
            #         pred_scores[:pseudo_scores[i].shape[0]] = pseudo_scores[i]
            #         V.vis(point, gt_boxes=ori_unlabeled_boxes[i][:, :-1],
            #             pred_boxes=pred_boxes, pred_scores=pred_scores, pred_labels=pred_labels)

            # ori_unlabeled_boxes_list = [ori_box for ori_box in ori_unlabeled_boxes]
            # pseudo_boxes_list = [ps_box for ps_box in batch_dict['gt_boxes'][unlabeled_inds]]
            # metric_inputs = {'preds': pseudo_boxes_list,
            #                  'targets': ori_unlabeled_boxes_list,
            #                  'pred_scores': pseudo_scores,
            #                  'pred_sem_scores': pseudo_sem_scores}
            # self.metrics['after_filtering'].update(**metric_inputs)  # commented to reduce complexity.

            batch_dict['metric_registry'] = self.metric_registry
            batch_dict['ori_unlabeled_boxes'] = ori_unlabeled_boxes
            for cur_module in self.pv_rcnn.module_list:
                if cur_module.model_cfg['NAME'] == 'PVRCNNHead' and self.model_cfg['ROI_HEAD'].get('ENABLE_RCNN_CONSISTENCY', False):
                    # Pass teacher's proposal to the student.
                    # To let proposal_layer continues for labeled data we pass rois with _ema postfix
                    batch_dict['rois_ema'] = batch_dict_ema['rois'].detach().clone()
                    # TODO(farzad) the normalization is done lazily, to be consistent with the other unnormalized roi_scores.
                    batch_dict['roi_scores_ema'] = batch_dict_ema['roi_scores'].detach().clone()
                    batch_dict['roi_labels_ema'] = batch_dict_ema['roi_labels'].detach().clone()
                    batch_dict = self.apply_augmentation(batch_dict, batch_dict, unlabeled_inds, key='rois_ema')
                    if self.model_cfg['ROI_HEAD'].get('ENABLE_RELIABILITY', False):
                        # pseudo-labels used for training roi head
                        pred_dicts = self.ensemble_post_processing(batch_dict_ema, batch_dict_ema_wa, unlabeled_inds,
                                                                   ensemble_option='mean_no_nms')
                    else:
                        pred_dicts, _ = self.pv_rcnn_ema.post_processing(batch_dict_ema, no_recall_dict=True,
                                                                         override_thresh=0.0, no_nms_for_unlabeled=True)
                    boxes, labels, scores, sem_scores, boxes_var, scores_var = self._unpack_predictions(pred_dicts,
                                                                                                        unlabeled_inds)
                    pseudo_boxes = [torch.cat([box, label.unsqueeze(-1)], dim=-1) for box, label in zip(boxes, labels)]
                    self._fill_with_pseudo_labels(batch_dict, pseudo_boxes, unlabeled_inds, labeled_inds)
                    batch_dict = self.apply_augmentation(batch_dict, batch_dict, unlabeled_inds, key='gt_boxes')
                    batch_dict['pred_scores_ema'] = torch.zeros_like(batch_dict['roi_scores_ema'])
                    for i, ui in enumerate(unlabeled_inds):
                        batch_dict['pred_scores_ema'][ui] = scores[i]
                    # TODO(farzad) ENABLE_RELIABILITY option should not necessarily always have var
                    if self.model_cfg['ROI_HEAD'].get('ENABLE_RELIABILITY', False):
                        batch_dict['pred_scores_ema_var'] = torch.zeros_like(batch_dict['roi_scores_ema'])
                        batch_dict['pred_boxes_ema_var'] = torch.zeros_like(batch_dict['rois_ema'])
                        for i, ui in enumerate(unlabeled_inds):
                            batch_dict['pred_scores_ema_var'][ui] = scores_var[i]
                            batch_dict['pred_boxes_ema_var'][ui] = boxes_var[i]

                    # if self.model_cfg.ROI_HEAD.get('ENABLE_VIS', False):
                    #     for i, uind in enumerate(unlabeled_inds):
                    #         mask = batch_dict['points'][:, 0] == uind
                    #         point = batch_dict['points'][mask, 1:]
                    #         pred_boxes = batch_dict['gt_boxes'][uind][:, :-1]
                    #         pred_labels = batch_dict['gt_boxes'][uind][:, -1].int()
                    #         pred_scores = batch_dict['pred_scores_ema'][uind]
                    #         V.vis(point, gt_boxes=ori_unlabeled_boxes[i][:, :-1], pred_boxes=pred_boxes,
                    #             pred_scores=pred_scores, pred_labels=pred_labels)

                batch_dict = cur_module(batch_dict)

            # For metrics calculation
            self.pv_rcnn.roi_head.forward_ret_dict['unlabeled_inds'] = unlabeled_inds
            self.pv_rcnn.roi_head.forward_ret_dict['pl_boxes'] = batch_dict['gt_boxes']
            self.pv_rcnn.roi_head.forward_ret_dict['pl_scores'] = pseudo_scores

            if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False):
                # using teacher to evaluate student's bg/fg proposals through its rcnn head
                with torch.no_grad():
                    batch_dict_std = {}
                    batch_dict_std['unlabeled_inds'] = batch_dict['unlabeled_inds']
                    batch_dict_std['rois'] = batch_dict['rois'].data.clone()
                    batch_dict_std['roi_scores'] = batch_dict['roi_scores'].data.clone()
                    batch_dict_std['roi_labels'] = batch_dict['roi_labels'].data.clone()
                    batch_dict_std['has_class_labels'] = batch_dict['has_class_labels']
                    batch_dict_std['batch_size'] = batch_dict['batch_size']
                    batch_dict_std['point_features'] = batch_dict_ema['point_features'].data.clone()
                    batch_dict_std['point_coords'] = batch_dict_ema['point_coords'].data.clone()
                    batch_dict_std['point_cls_scores'] = batch_dict_ema['point_cls_scores'].data.clone()

                    batch_dict_std = self.reverse_augmentation(batch_dict_std, batch_dict, unlabeled_inds)

                    # Perturb Student's ROIs before using them for Teacher's ROI head
                    if self.model_cfg.ROI_HEAD.ROI_AUG.get('ENABLE', False):
                        augment_rois = getattr(augmentor_utils, self.model_cfg.ROI_HEAD.ROI_AUG.AUG_TYPE, augmentor_utils.roi_aug_ros)
                        # rois_before_aug is used only for debugging, can be removed later
                        batch_dict_std['rois_before_aug'] = batch_dict_std['rois'].clone().detach()
                        batch_dict_std['rois'][unlabeled_inds] = \
                            augment_rois(batch_dict_std['rois'][unlabeled_inds], self.model_cfg.ROI_HEAD)

                    self.pv_rcnn_ema.roi_head.forward(batch_dict_std,
                                                      disable_gt_roi_when_pseudo_labeling=True)
                    batch_dict_std = self.apply_augmentation(batch_dict_std, batch_dict, unlabeled_inds, key='batch_box_preds')

                    pred_dicts_std, recall_dicts_std = self.pv_rcnn_ema.post_processing(batch_dict_std,
                                                                                        no_recall_dict=True,
                                                                                        no_nms_for_unlabeled=True)
                    rcnn_cls_score_teacher = -torch.ones_like(self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_labels'])
                    batch_box_preds_teacher = torch.zeros_like(self.pv_rcnn.roi_head.forward_ret_dict['batch_box_preds'])
                    for uind in unlabeled_inds:
                        rcnn_cls_score_teacher[uind] = pred_dicts_std[uind]['pred_scores']
                        batch_box_preds_teacher[uind] = pred_dicts_std[uind]['pred_boxes']
                    self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_score_teacher'] = rcnn_cls_score_teacher
                    # For metrics
                    self.pv_rcnn.roi_head.forward_ret_dict['batch_box_preds_teacher'] = batch_box_preds_teacher

            disp_dict = {}
            loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
            loss_rcnn_cls, loss_rcnn_box, ulb_loss_cls_dist, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)

            if self.model_cfg.USE_SUM_OVER_LOSS:
                if not self.unlabeled_supervise_cls:
                    loss_rpn_cls = loss_rpn_cls[labeled_inds, ...].sum()
                else:
                    loss_rpn_cls = loss_rpn_cls[labeled_inds, ...].sum() + loss_rpn_cls[unlabeled_inds, ...].sum() * self.unlabeled_weight

                loss_rpn_box = loss_rpn_box[labeled_inds, ...].sum() + loss_rpn_box[unlabeled_inds, ...].sum() * self.unlabeled_weight
                loss_point = loss_point[labeled_inds, ...].sum()
                if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False) or self.model_cfg.get('UNLABELED_SUPERVISE_OBJ', False):
                    loss_rcnn_cls = loss_rcnn_cls[labeled_inds, ...].sum() + loss_rcnn_cls[unlabeled_inds, ...].sum() * self.unlabeled_weight
                else:
                    loss_rcnn_cls = loss_rcnn_cls[labeled_inds, ...].sum()
                if not self.unlabeled_supervise_refine:
                    loss_rcnn_box = loss_rcnn_box[labeled_inds, ...].sum()
                else:
                    loss_rcnn_box = loss_rcnn_box[labeled_inds, ...].sum() + loss_rcnn_box[unlabeled_inds, ...].sum() * self.unlabeled_weight

                loss = loss_rpn_cls + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box
                tb_dict_ = {}
                for key in tb_dict.keys():
                    if 'loss' in key:
                        tb_dict_[key+"_labeled"] = tb_dict[key][labeled_inds, ...].sum()
                        tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_inds, ...].sum()
                    elif 'acc' in key:
                        tb_dict_[key+"_labeled"] = tb_dict[key][labeled_inds, ...].sum()
                        tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_inds, ...].sum()
                    elif 'point_pos_num' in key:
                        tb_dict_[key + "_labeled"] = tb_dict[key][labeled_inds, ...].sum()
                        tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_inds, ...].sum()
                    else:
                        tb_dict_[key] = tb_dict[key]
            
            else:
                if not self.unlabeled_supervise_cls:
                    loss_rpn_cls = loss_rpn_cls[labeled_inds, ...].mean()
                else:
                    loss_rpn_cls = loss_rpn_cls[labeled_inds, ...].mean() + loss_rpn_cls[unlabeled_inds, ...].mean() * self.unlabeled_weight

                loss_rpn_box = loss_rpn_box[labeled_inds, ...].mean() + loss_rpn_box[unlabeled_inds, ...].mean() * self.unlabeled_weight
                loss_point = loss_point[labeled_inds, ...].mean()
                if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False) or self.model_cfg.get('UNLABELED_SUPERVISE_OBJ', False):
                    loss_rcnn_cls = loss_rcnn_cls[labeled_inds, ...].mean() + loss_rcnn_cls[unlabeled_inds, ...].mean() * self.unlabeled_weight
                else:
                    loss_rcnn_cls = loss_rcnn_cls[labeled_inds, ...].mean()
                if not self.unlabeled_supervise_refine:
                    loss_rcnn_box = loss_rcnn_box[labeled_inds, ...].mean()
                else:
                    loss_rcnn_box = loss_rcnn_box[labeled_inds, ...].mean() + loss_rcnn_box[unlabeled_inds, ...].mean() * self.unlabeled_weight
                if self.model_cfg['ROI_HEAD'].get('ENABLE_ULB_CLS_DIST_LOSS', False):
                    loss = loss_rpn_cls + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box + ulb_loss_cls_dist
                else:
                    loss = loss_rpn_cls + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box
                tb_dict_ = {}
                for key in tb_dict.keys():
                    if 'loss' in key:
                        tb_dict_[key+"_labeled"] = tb_dict[key][labeled_inds, ...].mean()
                        tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_inds, ...].mean()
                    elif 'acc' in key:
                        tb_dict_[key+"_labeled"] = tb_dict[key][labeled_inds, ...].mean()
                        tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_inds, ...].mean()
                    elif 'point_pos_num' in key:
                        tb_dict_[key + "_labeled"] = tb_dict[key][labeled_inds, ...].mean()
                        tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_inds, ...].mean()
                    else:
                        tb_dict_[key] = tb_dict[key]

            if self.model_cfg.get('STORE_SCORES_IN_PKL', False) :
                # Store different types of scores over all itrs and epochs and dump them in a pickle for offline modeling 
                # TODO (shashank) : Can be optimized later to save computational time, currently takes about 0.002sec
                batch_roi_labels = self.pv_rcnn.roi_head.forward_ret_dict['roi_labels'][unlabeled_inds]
                batch_roi_labels = [roi_labels.clone().detach() for roi_labels in batch_roi_labels]

                batch_rois = self.pv_rcnn.roi_head.forward_ret_dict['rois'][unlabeled_inds]
                batch_rois = [rois.clone().detach() for rois in batch_rois]

                batch_ori_gt_boxes = self.pv_rcnn.roi_head.forward_ret_dict['ori_unlabeled_boxes']
                batch_ori_gt_boxes = [ori_gt_boxes.clone().detach() for ori_gt_boxes in batch_ori_gt_boxes]

                for i in range(len(batch_rois)):
                    valid_rois_mask = torch.logical_not(torch.all(batch_rois[i] == 0, dim=-1))
                    valid_rois = batch_rois[i][valid_rois_mask]
                    valid_roi_labels = batch_roi_labels[i][valid_rois_mask]
                    valid_roi_labels -= 1                                   # Starting class indices from zero

                    valid_gt_boxes_mask = torch.logical_not(torch.all(batch_ori_gt_boxes[i] == 0, dim=-1))
                    valid_gt_boxes = batch_ori_gt_boxes[i][valid_gt_boxes_mask]
                    valid_gt_boxes[:, -1] -= 1                              # Starting class indices from zero

                    num_gts = valid_gt_boxes_mask.sum()
                    num_preds = valid_rois_mask.sum()

                    cur_unlabeled_ind = unlabeled_inds[i]
                    if num_gts > 0 and num_preds > 0:
                        # Find IoU between Student's ROI v/s Original GTs
                        overlap = iou3d_nms_utils.boxes_iou3d_gpu(valid_rois[:, 0:7], valid_gt_boxes[:, 0:7])
                        preds_iou_max, assigned_gt_inds = overlap.max(dim=1)
                        self.val_dict['iou_roi_gt'].extend(preds_iou_max.tolist())

                        cur_iou_roi_pl = self.pv_rcnn.roi_head.forward_ret_dict['gt_iou_of_rois'][cur_unlabeled_ind]
                        self.val_dict['iou_roi_pl'].extend(cur_iou_roi_pl.tolist())

                        cur_pred_score = torch.sigmoid(batch_dict['batch_cls_preds'][cur_unlabeled_ind]).squeeze()
                        self.val_dict['pred_scores'].extend(cur_pred_score.tolist())

                        roi_scores = torch.sigmoid(self.pv_rcnn.roi_head.forward_ret_dict['roi_scores'][cur_unlabeled_ind])
                        self.val_dict['roi_scores'].extend(roi_scores.tolist())

                        cur_weight = self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_weights'][cur_unlabeled_ind]
                        self.val_dict['weights'].extend(cur_weight.tolist())

                        cur_roi_label = batch_dict['roi_labels'][cur_unlabeled_ind].squeeze()
                        self.val_dict['class_labels'].extend(cur_roi_label.tolist())

                        cur_iteration = torch.ones_like(preds_iou_max) * (batch_dict['cur_iteration'])
                        self.val_dict['iteration'].extend(cur_iteration.tolist())

                # replace old pickle data (if exists) with updated one 
                output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                file_path = os.path.join(output_dir, 'scores.pkl')
                pickle.dump(self.val_dict, open(file_path, 'wb'))

            for key in self.metric_registry.tags():
                metrics = self.compute_metrics(tag=key)
                tb_dict_.update(metrics)

            if dist.is_initialized():
                rank = os.getenv('RANK')
                tb_dict_[f'bs_rank_{rank}'] = int(batch_dict['gt_boxes'].shape[0])
            else:
                tb_dict_[f'bs'] = int(batch_dict['gt_boxes'].shape[0])

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict_, disp_dict

        else:
            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.pv_rcnn.post_processing(batch_dict)

            pseudo_boxes_list = [torch.cat([pred_dict['pred_boxes'], pred_dict['pred_labels'].unsqueeze(-1)], dim=-1)
                                 for pred_dict in pred_dicts]
            pseudo_scores = [pred_dict['pred_scores'] for pred_dict in pred_dicts]
            gt_boxes = [gt_box for gt_box in batch_dict['gt_boxes']]
            metric_inputs = {'preds': pseudo_boxes_list,
                             'pred_scores': pseudo_scores,
                             'ground_truths': gt_boxes}

            self.metric_registry.get('test').update(**metric_inputs)

            return pred_dicts, recall_dicts, {}

    def update_metrics(self, input_dict, pred_dict, unlabeled_inds, labeled_inds):
        """
        Recording PL vs GT statistics BEFORE filtering
        """
        if 'pl_gt_metrics_before_filtering' in self.model_cfg.ROI_HEAD.METRICS_PRED_TYPES:
            pseudo_boxes, pseudo_labels, pseudo_scores, pseudo_sem_scores, _, _ = self._unpack_predictions(
                pred_dict, unlabeled_inds)
            pseudo_boxes = [torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1) \
                            for (pseudo_box, pseudo_label) in zip(pseudo_boxes, pseudo_labels)]

            # Making consistent # of pseudo boxes in each batch
            # NOTE: Need to store them in batch_dict in a new key, which can be removed later
            input_dict['pseudo_boxes_prefilter'] = torch.zeros_like(input_dict['gt_boxes'])
            self._fill_with_pseudo_labels(input_dict, pseudo_boxes, unlabeled_inds, labeled_inds,
                                          key='pseudo_boxes_prefilter')

            # apply student's augs on teacher's pseudo-boxes (w/o filtered)
            batch_dict = self.apply_augmentation(input_dict, input_dict, unlabeled_inds, key='pseudo_boxes_prefilter')

            tag = f'pl_gt_metrics_before_filtering'
            metrics = self.metric_registry.get(tag)

            preds_prefilter = [batch_dict['pseudo_boxes_prefilter'][uind] for uind in unlabeled_inds]
            gts_prefilter = [batch_dict['gt_boxes'][uind] for uind in unlabeled_inds]
            metric_inputs = {'preds': preds_prefilter, 'pred_scores': pseudo_scores, 'roi_scores': pseudo_sem_scores,
                             'ground_truths': gts_prefilter}
            metrics.update(**metric_inputs)
            batch_dict.pop('pseudo_boxes_prefilter')

    def compute_metrics(self, tag):
        results = self.metric_registry.get(tag).compute()
        tag = tag + "/" if tag else ''
        metrics = {tag + key: val for key, val in results.items()}

        return metrics

    def ensemble_post_processing(self, batch_dict_a, batch_dict_b, unlabeled_inds, ensemble_option=None):
        # TODO(farzad) what about roi_labels and roi_scores in following options?
        ens_pred_dicts = None
        if ensemble_option == 'joint_nms':
            cat_keys = ['batch_cls_preds', 'batch_box_preds', 'roi_labels', 'roi_scores']
            for key in cat_keys:
                batch_dict_a[key][unlabeled_inds] = torch.cat(
                    (batch_dict_a[key][unlabeled_inds], batch_dict_b[key][unlabeled_inds]), dim=1)

        elif ensemble_option == 'mean_pre_nms':
            _mean_and_var(batch_dict_a, batch_dict_b, unlabeled_inds,
                          keys=('batch_cls_preds', 'batch_box_preds'))
            # backup original values and replace them with mean values
            for key in ['batch_box_preds', 'batch_cls_preds']:
                batch_dict_a[key + '_src'] = batch_dict_a[key].clone().detach()
                batch_dict_a[key][unlabeled_inds] = batch_dict_a[key + '_mean'][unlabeled_inds]

            ens_pred_dicts, _ = self.pv_rcnn_ema.post_processing(batch_dict_a, no_recall_dict=True)

            # replace means with original values and remove means/vars
            for key in ['batch_box_preds', 'batch_cls_preds']:
                batch_dict_a[key] = batch_dict_a[key + '_src'].clone().detach()
                batch_dict_a.pop(key + '_src')
                batch_dict_a.pop(key + '_mean')
                batch_dict_a.pop(key + '_var')

        elif ensemble_option == 'mean_no_nms':
            # no_nms has been set to True to avoid the filtering and keep the o/p consistent with that of student

            pred_dicts_a, _ = self.pv_rcnn_ema.post_processing(batch_dict_a, no_recall_dict=True,
                                                               no_nms_for_unlabeled=True)
            pred_dicts_b, _ = self.pv_rcnn_ema.post_processing(batch_dict_b, no_recall_dict=True,
                                                               no_nms_for_unlabeled=True)
            _mean_and_var(pred_dicts_a, pred_dicts_b, unlabeled_inds, keys=('pred_scores', 'pred_boxes'))
            # replace original values with mean values
            for ind in unlabeled_inds:
                for key in ['pred_scores', 'pred_boxes']:
                    pred_dicts_a[ind][key] = pred_dicts_a[ind][key + '_mean']
                    pred_dicts_a[ind].pop(key + '_mean')
            ens_pred_dicts = pred_dicts_a

        elif ensemble_option == 'weighted_mean':
            _weighted_mean(batch_dict_a, batch_dict_b, unlabeled_inds, keys=('batch_cls_preds', 'batch_box_preds'))

        elif ensemble_option == 'max_only':
            _max_score_replacement(batch_dict_a, batch_dict_b, unlabeled_inds,
                                   keys=('batch_cls_preds', 'batch_box_preds'))

        elif ensemble_option is None:
            ens_pred_dicts, _ = self.pv_rcnn_ema.post_processing(batch_dict_a, no_recall_dict=True)

        return ens_pred_dicts

    # TODO(farzad) refactor and remove this!
    def _unpack_predictions(self, pred_dicts, unlabeled_inds):
        pseudo_boxes = []
        pseudo_scores = []
        pseudo_sem_scores = []
        pseudo_labels = []
        pseudo_boxes_var = []
        pseudo_scores_var = []
        for ind in unlabeled_inds:
            pseudo_score = pred_dicts[ind]['pred_scores']
            pseudo_box = pred_dicts[ind]['pred_boxes']
            pseudo_label = pred_dicts[ind]['pred_labels']
            pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']
            # TODO(farzad) REFACTOR LATER!
            pseudo_box_var = -1 * torch.ones_like(pseudo_box)
            if "pred_boxes_var" in pred_dicts[ind].keys():
                pseudo_box_var = pred_dicts[ind]['pred_boxes_var']
            pseudo_score_var = -1 * torch.ones_like(pseudo_score)
            if "pred_scores_var" in pred_dicts[ind].keys():
                pseudo_score_var = pred_dicts[ind]['pred_scores_var']
            if len(pseudo_label) == 0:
                pseudo_boxes.append(pseudo_label.new_zeros((1, 7)).float())
                pseudo_boxes_var.append(pseudo_label.new_zeros((1, 7)).float())
                pseudo_sem_scores.append(pseudo_label.new_zeros((1,)).float())
                pseudo_scores.append(pseudo_label.new_zeros((1,)).float())
                pseudo_scores_var.append(pseudo_label.new_zeros((1,)).float())
                pseudo_labels.append(pseudo_label.new_zeros((1,)).float())
                continue

            pseudo_boxes.append(pseudo_box)
            pseudo_boxes_var.append(pseudo_box_var)
            pseudo_sem_scores.append(pseudo_sem_score)
            pseudo_scores.append(pseudo_score)
            pseudo_scores_var.append(pseudo_score_var)
            pseudo_labels.append(pseudo_label)

        return pseudo_boxes, pseudo_labels, pseudo_scores, pseudo_sem_scores, pseudo_boxes_var, pseudo_scores_var

    # TODO(farzad) refactor and remove this!
    def _filter_pseudo_labels(self, pred_dicts, unlabeled_inds):
        pseudo_boxes = []
        pseudo_scores = []
        pseudo_sem_scores = []
        pseudo_scores_var = []
        pseudo_boxes_var = []
        for pseudo_box, pseudo_label, pseudo_score, pseudo_sem_score, pseudo_box_var, pseudo_score_var in zip(
                *self._unpack_predictions(pred_dicts, unlabeled_inds)):

            if pseudo_label[0] == 0:
                pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
                pseudo_sem_scores.append(pseudo_sem_score)
                pseudo_scores.append(pseudo_score)
                pseudo_scores_var.append(pseudo_score_var)
                pseudo_boxes_var.append(pseudo_box_var)
                continue

            conf_thresh = torch.tensor(self.thresh, device=pseudo_label.device).unsqueeze(
                0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label - 1).unsqueeze(-1))

            sem_conf_thresh = torch.tensor(self.sem_thresh, device=pseudo_label.device).unsqueeze(
                0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label - 1).unsqueeze(-1))

            valid_inds = pseudo_score > conf_thresh.squeeze()

            valid_inds = valid_inds & (pseudo_sem_score > sem_conf_thresh.squeeze())

            # TODO(farzad) can this be similarly determined by tag-based stats before and after filtering?
            # rej_labels = pseudo_label[~valid_inds]
            # rej_labels_per_class = torch.bincount(rej_labels, minlength=len(self.thresh) + 1)
            # for class_ind, class_key in enumerate(self.metric_table.metric_record):
            #     if class_key == 'class_agnostic':
            #         self.metric_table.metric_record[class_key].metrics['rej_pseudo_lab'].update(
            #             rej_labels_per_class[1:].sum().item())
            #     else:
            #         self.metric_table.metric_record[class_key].metrics['rej_pseudo_lab'].update(
            #             rej_labels_per_class[class_ind].item())

            pseudo_sem_score = pseudo_sem_score[valid_inds]
            pseudo_box = pseudo_box[valid_inds]
            pseudo_label = pseudo_label[valid_inds]
            pseudo_score = pseudo_score[valid_inds]
            pseudo_box_var = pseudo_box_var[valid_inds]
            pseudo_score_var = pseudo_score_var[valid_inds]
            # TODO : Two stage filtering instead of applying NMS
            # Stage1 based on size of bbox, Stage2 is objectness thresholding
            # Note : Two stages happen sequentially, and not independently.
            # vol_boxes = ((pseudo_box[:, 3] * pseudo_box[:, 4] * pseudo_box[:, 5])/torch.abs(pseudo_box[:,2][0])).view(-1)
            # vol_boxes, _ = torch.sort(vol_boxes, descending=True)
            # # Set volume threshold to 10% of the maximum volume of the boxes
            # keep_ind = int(self.model_cfg.PSEUDO_TWO_STAGE_FILTER.MAX_VOL_PROP * len(vol_boxes))
            # keep_vol = vol_boxes[keep_ind]
            # valid_inds = vol_boxes > keep_vol # Stage 1
            # pseudo_sem_score = pseudo_sem_score[valid_inds]
            # pseudo_box = pseudo_box[valid_inds]
            # pseudo_label = pseudo_label[valid_inds]
            # pseudo_score = pseudo_score[valid_inds]

            # valid_inds = pseudo_score > self.model_cfg.PSEUDO_TWO_STAGE_FILTER.THRESH # Stage 2
            # pseudo_sem_score = pseudo_sem_score[valid_inds]
            # pseudo_box = pseudo_box[valid_inds]
            # pseudo_label = pseudo_label[valid_inds]
            # pseudo_score = pseudo_score[valid_inds]

            pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
            pseudo_sem_scores.append(pseudo_sem_score)
            pseudo_scores.append(pseudo_score)
            pseudo_scores_var.append(pseudo_score_var)
            pseudo_boxes_var.append(pseudo_box_var)

        return pseudo_boxes, pseudo_scores, pseudo_sem_scores, pseudo_boxes_var, pseudo_scores_var

    def _fill_with_pseudo_labels(self, batch_dict, pseudo_boxes, unlabeled_inds, labeled_inds, key=None):
        key = 'gt_boxes' if key is None else key
        max_box_num = batch_dict['gt_boxes'].shape[1]

        # Ignore the count of pseudo boxes if filled with default values(zeros) when no preds are made
        max_pseudo_box_num = max(
            [torch.logical_not(torch.all(ps_box == 0, dim=-1)).sum().item() for ps_box in pseudo_boxes])

        if max_box_num >= max_pseudo_box_num:
            for i, pseudo_box in enumerate(pseudo_boxes):
                diff = max_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                batch_dict[key][unlabeled_inds[i]] = pseudo_box
        else:
            ori_boxes = batch_dict['gt_boxes']
            new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[2]),
                                    device=ori_boxes.device)
            for i, inds in enumerate(labeled_inds):
                diff = max_pseudo_box_num - ori_boxes[inds].shape[0]
                new_box = torch.cat([ori_boxes[inds], torch.zeros((diff, 8), device=ori_boxes[inds].device)], dim=0)
                new_boxes[inds] = new_box
            for i, pseudo_box in enumerate(pseudo_boxes):

                diff = max_pseudo_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                new_boxes[unlabeled_inds[i]] = pseudo_box
            batch_dict[key] = new_boxes

    def apply_augmentation(self, batch_dict, batch_dict_org, unlabeled_inds, key='rois'):
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_x_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_x'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_y_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_y'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_rotation_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['rot_angle'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_scaling_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['scale'][unlabeled_inds])

        batch_dict[key][unlabeled_inds, :, 6] = common_utils.limit_period(
            batch_dict[key][unlabeled_inds, :, 6], offset=0.5, period=2 * np.pi
        )

        return batch_dict

    def reverse_augmentation(self, batch_dict, batch_dict_org, unlabeled_inds, key='rois'):
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_scaling_bbox(
            batch_dict[key][unlabeled_inds], 1.0 / batch_dict_org['scale'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_rotation_bbox(
            batch_dict[key][unlabeled_inds], - batch_dict_org['rot_angle'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_y_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_y'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_x_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_x'][unlabeled_inds])

        batch_dict[key][unlabeled_inds, :, 6] = common_utils.limit_period(
            batch_dict[key][unlabeled_inds, :, 6], offset=0.5, period=2 * np.pi
        )

        return batch_dict

    def get_supervised_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def update_global_step(self):
        self.global_step += 1
        self.accumulated_itr += 1
        if self.accumulated_itr % self.model_cfg.EMA_UPDATE_INTERVAL != 0:
            return
        alpha = self.model_cfg.EMA_ALPHA
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.pv_rcnn_ema.parameters(), self.pv_rcnn.parameters()):
            # TODO(farzad) check this
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)
        self.accumulated_itr = 0

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            new_key = 'pv_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            new_key = 'pv_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

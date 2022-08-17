import copy
import os

import torch
import torch.nn.functional as F
import numpy as np

from pcdet.datasets.augmentor.augmentor_utils import *
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils.cal_quality_utils import QualityMetric

from ...utils import common_utils
from .detector3d_template import Detector3DTemplate

from.pv_rcnn import PVRCNN


def _mean(tensor_list):
    tensor = torch.cat(tensor_list)
    tensor = tensor[~torch.isnan(tensor)]
    mean = tensor.mean() if len(tensor) > 0 else torch.tensor([float('nan')])
    return mean

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

        # self.module_list = self.build_networks()
        # self.module_list_ema = self.build_networks()
        self.thresh = model_cfg.THRESH
        self.sem_thresh = model_cfg.SEM_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.no_nms = model_cfg.NO_NMS
        self.supervise_mode = model_cfg.SUPERVISE_MODE

        self.metrics = QualityMetric()

    def forward(self, batch_dict):
        if self.training:
            labeled_mask = batch_dict['labeled_mask'].view(-1)
            labeled_inds = torch.nonzero(labeled_mask).squeeze(1).long()
            unlabeled_inds = torch.nonzero(1-labeled_mask).squeeze(1).long()
            batch_dict_ema = {}
            keys = list(batch_dict.keys())
            for k in keys:
                if k + '_ema' in keys:
                    continue
                if k.endswith('_ema'):
                    batch_dict_ema[k[:-4]] = batch_dict[k]
                else:
                    batch_dict_ema[k] = batch_dict[k]

            with torch.no_grad():
                # self.pv_rcnn_ema.eval()  # Important! must be in train mode
                for cur_module in self.pv_rcnn_ema.module_list:
                    try:
                        batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                    except:
                        batch_dict_ema = cur_module(batch_dict_ema)
                pred_dicts, recall_dicts = self.pv_rcnn_ema.post_processing(batch_dict_ema,
                                                                            no_recall_dict=True, override_thresh=0.0, no_nms=self.no_nms)

                pseudo_boxes = []
                pseudo_scores = []
                pseudo_sem_scores = []
                max_pseudo_box_num = 0
                for ind in unlabeled_inds:
                    pseudo_score = pred_dicts[ind]['pred_scores'] 
                    pseudo_box = pred_dicts[ind]['pred_boxes']
                    pseudo_label = pred_dicts[ind]['pred_labels']
                    pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']

                    if len(pseudo_label) == 0:
                        pseudo_boxes.append(pseudo_label.new_zeros((0, 8)).float())
                        pseudo_sem_scores.append(pseudo_label.new_zeros((1,)).float())
                        pseudo_scores.append(pseudo_label.new_zeros((1,)).float())
                        continue


                    conf_thresh = torch.tensor(self.thresh, device=pseudo_label.device).unsqueeze(
                        0).repeat(len(pseudo_label), 1).gather(dim=1, index=(pseudo_label-1).unsqueeze(-1))

                    valid_inds = pseudo_score > conf_thresh.squeeze()

                    valid_inds = valid_inds * (pseudo_sem_score > self.sem_thresh[0])

                    pseudo_sem_score = pseudo_sem_score[valid_inds]
                    pseudo_box = pseudo_box[valid_inds]
                    pseudo_label = pseudo_label[valid_inds]
                    pseudo_score = pseudo_score[valid_inds]

                    pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
                    pseudo_sem_scores.append(pseudo_sem_score)
                    pseudo_scores.append(pseudo_score)

                    if pseudo_box.shape[0] > max_pseudo_box_num:
                        max_pseudo_box_num = pseudo_box.shape[0]

                max_box_num = batch_dict['gt_boxes'].shape[1]

                # assert max_box_num >= max_pseudo_box_num
                ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_inds, ...]

                if max_box_num >= max_pseudo_box_num:
                    for i, pseudo_box in enumerate(pseudo_boxes):
                        diff = max_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, 8), device=pseudo_box.device)], dim=0)
                        batch_dict['gt_boxes'][unlabeled_inds[i]] = pseudo_box
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
                    batch_dict['gt_boxes'] = new_boxes
                # apply student's augs on teacher's pseudo-labels only (not points)
                batch_dict['gt_boxes'][unlabeled_inds, ...] = random_flip_along_x_bbox(
                    batch_dict['gt_boxes'][unlabeled_inds, ...], batch_dict['flip_x'][unlabeled_inds, ...]
                )

                batch_dict['gt_boxes'][unlabeled_inds, ...] = random_flip_along_y_bbox(
                    batch_dict['gt_boxes'][unlabeled_inds, ...], batch_dict['flip_y'][unlabeled_inds, ...]
                )

                batch_dict['gt_boxes'][unlabeled_inds, ...] = global_rotation_bbox(
                    batch_dict['gt_boxes'][unlabeled_inds, ...], batch_dict['rot_angle'][unlabeled_inds, ...]
                )

                batch_dict['gt_boxes'][unlabeled_inds, ...] = global_scaling_bbox(
                    batch_dict['gt_boxes'][unlabeled_inds, ...], batch_dict['scale'][unlabeled_inds, ...]
                )

                pseudo_ious = []
                pseudo_accs = []
                pseudo_fgs = []
                sem_score_fgs = []
                sem_score_bgs = []
                for i, ind in enumerate(unlabeled_inds):
                    # statistics
                    anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
                        batch_dict['gt_boxes'][ind, ...][:, 0:7],
                        ori_unlabeled_boxes[i, :, 0:7])
                    cls_pseudo = batch_dict['gt_boxes'][ind, ...][:, 7]
                    nonzero_inds = torch.nonzero(cls_pseudo).squeeze(1).long()
                    cls_pseudo = cls_pseudo[nonzero_inds] 
                    if len(nonzero_inds) > 0:
                        iou_max, asgn = anchor_by_gt_overlap[nonzero_inds, :].max(dim=1)
                        pseudo_ious.append(iou_max.mean().unsqueeze(dim=0))
                        acc = (ori_unlabeled_boxes[i][:, 7].gather(dim=0, index=asgn) == cls_pseudo).float().mean()
                        pseudo_accs.append(acc.unsqueeze(0))
                        fg_thresh = self.model_cfg['ROI_HEAD']['TARGET_CONFIG']['CLS_FG_THRESH']
                        bg_thresh = self.model_cfg['ROI_HEAD']['TARGET_CONFIG']['CLS_BG_THRESH']  # bg_thresh includes both easy and hard bgs
                        fg = (iou_max > fg_thresh).float().sum(dim=0, keepdim=True) / len(nonzero_inds) 
                        
                        # Store tp, fp, fn per batch
                        tp_mask = iou_max >= fg_thresh
                        tp = tp_mask.sum().item()
                        fp = iou_max.shape[0] - tp
                        # gt boxes that missed by tp boxes are fn boxes
                        # fn (gt boxes are fg but pseduo boxes are bg) = total gt boxes - tp
                        fn = ori_unlabeled_boxes.shape[1] - tp
                        self.metrics.tp.append(tp)
                        self.metrics.fp.append(fp)
                        self.metrics.fn.append(fn)

                        # get tp boxes and their corresponding gt boxes
                        tp_pseudo_boxes = batch_dict['gt_boxes'][ind][nonzero_inds[tp_mask]]  
                        tp_gt_boxes = ori_unlabeled_boxes[i][asgn, :][tp_mask]               
                        
                        if tp > 0:
                            # Used for computing "Assignment error" later (based on Combating Noise paper)
                            trans_err, orient_err, scale_err = self.metrics.cal_diff(tp_pseudo_boxes, tp_gt_boxes, tp)
                            self.metrics.assignment_error.append(trans_err + orient_err + scale_err)

                            self.metrics.precision.append(tp / (tp + fp))
                            self.metrics.recall.append(tp / (tp + fn))

                        else :
                            self.metrics.assignment_error.append(float('nan'))

                            self.metrics.precision.append(float('nan'))
                            self.metrics.recall.append(float('nan'))

                        self.metrics.missed_gt_error.append(fn)

                        self.metrics.classification_error.append((1 - acc).item())


                        sem_score_fg = (pseudo_sem_scores[i][nonzero_inds] * (iou_max > fg_thresh).float()).sum(dim=0, keepdim=True) \
                                       / torch.clamp((iou_max > fg_thresh).float().sum(dim=0, keepdim=True), min=1.0)
                        sem_score_bg = (pseudo_sem_scores[i][nonzero_inds] * (iou_max < bg_thresh).float()).sum(dim=0, keepdim=True) \
                                       / torch.clamp((iou_max < bg_thresh).float().sum(dim=0, keepdim=True), min=1.0)
                        pseudo_fgs.append(fg)
                        sem_score_fgs.append(sem_score_fg)
                        sem_score_bgs.append(sem_score_bg)

                        # only for 100% label
                        if self.supervise_mode >= 1:
                            filter = iou_max > 0.3
                            asgn = asgn[filter]
                            batch_dict['gt_boxes'][ind, ...][:] = torch.zeros_like(batch_dict['gt_boxes'][ind, ...][:])
                            batch_dict['gt_boxes'][ind, ...][:len(asgn)] = ori_unlabeled_boxes[i, :].gather(dim=0, index=asgn.unsqueeze(-1).repeat(1, 8))

                            if self.supervise_mode == 2:
                                batch_dict['gt_boxes'][ind, ...][:len(asgn), 0:3] += 0.1 * torch.randn((len(asgn), 3), device=iou_max.device) * \
                                                                                     batch_dict['gt_boxes'][ind, ...][
                                                                                     :len(asgn), 3:6]
                                batch_dict['gt_boxes'][ind, ...][:len(asgn), 3:6] += 0.1 * torch.randn((len(asgn), 3), device=iou_max.device) * \
                                                                                     batch_dict['gt_boxes'][ind, ...][
                                                                                     :len(asgn), 3:6]
                    else:
                        nan = torch.tensor([float('nan')], device=unlabeled_inds.device)
                        sem_score_fgs.append(nan)
                        sem_score_bgs.append(nan)
                        pseudo_ious.append(nan)
                        pseudo_accs.append(nan)
                        pseudo_fgs.append(nan)
                        
                        self.metrics.tp.append(float('nan'))
                        self.metrics.fp.append(float('nan'))
                        self.metrics.fn.append(float('nan'))

                        self.metrics.assignment_error.append(float('nan'))
                        self.metrics.missed_gt_error.append(float('nan'))
                        self.metrics.classification_error.append(float('nan'))  

            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False):
                # using teacher to evaluate student's bg/fg proposals through its rcnn head
                with torch.no_grad():
                    # batch_dict_std = copy.deepcopy(batch_dict) # doesn't work
                    batch_dict_std = {}
                    batch_dict_std['rois'] = batch_dict['rois'].data.clone()
                    batch_dict_std['roi_scores'] = batch_dict['roi_scores'].data.clone() 
                    batch_dict_std['roi_labels'] = batch_dict['roi_labels'].data.clone()
                    batch_dict_std['has_class_labels'] = batch_dict['has_class_labels']
                    batch_dict_std['batch_size'] = batch_dict['batch_size']
                    batch_dict_std['point_features'] = batch_dict_ema['point_features'].data.clone()
                    batch_dict_std['point_coords'] = batch_dict_ema['point_coords'].data.clone()
                    batch_dict_std['point_cls_scores'] = batch_dict_ema['point_cls_scores'].data.clone()

                    # reverse student's augmentation of rois
                    batch_dict_std['rois'][unlabeled_inds] = global_scaling_bbox(
                        batch_dict_std['rois'][unlabeled_inds], batch_dict['scale'][unlabeled_inds])
                    batch_dict_std['rois'][unlabeled_inds] = global_rotation_bbox(
                        batch_dict_std['rois'][unlabeled_inds], batch_dict['rot_angle'][unlabeled_inds])
                    batch_dict_std['rois'][unlabeled_inds] = random_flip_along_y_bbox(
                        batch_dict_std['rois'][unlabeled_inds], batch_dict['flip_y'][unlabeled_inds])
                    batch_dict_std['rois'][unlabeled_inds] = random_flip_along_x_bbox(
                        batch_dict_std['rois'][unlabeled_inds], batch_dict['flip_x'][unlabeled_inds])

                    self.pv_rcnn_ema.roi_head.forward(batch_dict_std,
                                                      disable_gt_roi_when_pseudo_labeling=True)

                    pred_dicts_std, recall_dicts_std = self.pv_rcnn_ema.post_processing(batch_dict_std,
                                                                                no_recall_dict=True, no_nms=True)
                    all_samples = []
                    for pred_dict in pred_dicts_std:
                        all_samples.append(pred_dict['pred_scores'].unsqueeze(dim=0))
                    pred_scores_teacher = torch.cat(all_samples, dim=0)
                    self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_score_teacher'] = pred_scores_teacher.data.clone() 
                    self.pv_rcnn.roi_head.forward_ret_dict['unlabeled_mask'] = unlabeled_inds

            disp_dict = {}
            loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
            loss_rcnn_cls, loss_rcnn_box, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)

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

            tb_dict_['pseudo_ious'] = _mean(pseudo_ious)
            tb_dict_['pseudo_accs'] = _mean(pseudo_accs)
            tb_dict_['sem_score_fg'] = _mean(sem_score_fgs)
            tb_dict_['sem_score_bg'] = _mean(sem_score_bgs)

            tb_dict_['max_box_num'] = max_box_num
            tb_dict_['max_pseudo_box_num'] = max_pseudo_box_num

            tb_dict_['assignment_error'] = np.nanmean(self.metrics.assignment_error)
            tb_dict_['missed_gt_error'] = np.nanmean(self.metrics.missed_gt_error)
            tb_dict_['classification_error'] = np.nanmean(self.metrics.classification_error)
            tb_dict_['precision'] = np.nanmean(self.metrics.precision)
            tb_dict_['recall'] = np.nanmean(self.metrics.recall)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict_, disp_dict

        else:
            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.pv_rcnn.post_processing(batch_dict)

            return pred_dicts, recall_dicts, {}

    def get_supervised_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def update_global_step(self):
        self.global_step += 1
        alpha = 0.999
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.pv_rcnn_ema.parameters(), self.pv_rcnn.parameters()):
            # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

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

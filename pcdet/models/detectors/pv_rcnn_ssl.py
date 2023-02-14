import os
import pickle
import torch
import copy

from pcdet.datasets.augmentor.augmentor_utils import *
from pcdet.ops.iou3d_nms import iou3d_nms_utils
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
        vals_to_store = ['iou_roi_pl', 'iou_roi_gt', 'pred_scores', 'weights', 'class_labels', 'iteration']
        self.val_dict = {val: [] for val in vals_to_store}

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
                max_box_num = batch_dict['gt_boxes'].shape[1]
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

                    # if len(valid_inds) > max_box_num:
                    #     _, inds = torch.sort(pseudo_score, descending=True)
                    #     inds = inds[:max_box_num]
                    #     pseudo_box = pseudo_box[inds]
                    #     pseudo_label = pseudo_label[inds]

                    pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1))
                    pseudo_sem_scores.append(pseudo_sem_score)
                    pseudo_scores.append(pseudo_score)

                    if pseudo_box.shape[0] > max_pseudo_box_num:
                        max_pseudo_box_num = pseudo_box.shape[0]
                    # pseudo_scores.append(pseudo_score)
                    # pseudo_labels.append(pseudo_label)

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
                    unzero_inds = torch.nonzero(cls_pseudo).squeeze(1).long()
                    cls_pseudo = cls_pseudo[unzero_inds]
                    if len(unzero_inds) > 0:
                        iou_max, asgn = anchor_by_gt_overlap[unzero_inds, :].max(dim=1)
                        pseudo_ious.append(iou_max.mean().unsqueeze(dim=0))
                        acc = (ori_unlabeled_boxes[i][:, 7].gather(dim=0, index=asgn) == cls_pseudo).float().mean()
                        pseudo_accs.append(acc.unsqueeze(0))
                        fg = (iou_max > 0.5).float().sum(dim=0, keepdim=True) / len(unzero_inds)

                        sem_score_fg = (pseudo_sem_scores[i][unzero_inds] * (iou_max > 0.5).float()).sum(dim=0, keepdim=True) \
                                       / torch.clamp((iou_max > 0.5).float().sum(dim=0, keepdim=True), min=1.0)
                        sem_score_bg = (pseudo_sem_scores[i][unzero_inds] * (iou_max < 0.5).float()).sum(dim=0, keepdim=True) \
                                       / torch.clamp((iou_max < 0.5).float().sum(dim=0, keepdim=True), min=1.0)
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

            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            disp_dict = {}
            loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
            loss_rcnn_cls, loss_rcnn_box, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)

            if not self.unlabeled_supervise_cls:
                loss_rpn_cls = loss_rpn_cls[labeled_inds, ...].sum()
            else:
                loss_rpn_cls = loss_rpn_cls[labeled_inds, ...].sum() + loss_rpn_cls[unlabeled_inds, ...].sum() * self.unlabeled_weight

            loss_rpn_box = loss_rpn_box[labeled_inds, ...].sum() + loss_rpn_box[unlabeled_inds, ...].sum() * self.unlabeled_weight
            loss_point = loss_point[labeled_inds, ...].sum()
            # Adding supervision of objectness score for unlabeled data as an ablation (not a part of original 3diou, default is False)
            if self.model_cfg.get('UNLABELED_SUPERVISE_OBJ', False):
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

            # Store different types of scores over all itrs and epochs and dump them in a pickle for offline modeling 
            # NOTE (shashank) : this has been adapted from DA-35-iou-metrics branch to compare the scores via plots
            batch_roi_labels = self.pv_rcnn.roi_head.forward_ret_dict['roi_labels'][unlabeled_inds]
            batch_roi_labels = [roi_labels.clone().detach() for roi_labels in batch_roi_labels]

            batch_rois = self.pv_rcnn.roi_head.forward_ret_dict['rois'][unlabeled_inds]
            batch_rois = [rois.clone().detach() for rois in batch_rois]

            batch_ori_gt_boxes = [ori_gt_boxes.clone().detach() for ori_gt_boxes in ori_unlabeled_boxes]

            class_to_name = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
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

                    # Store values in dict
                    cur_iou_roi_pl = self.pv_rcnn.roi_head.forward_ret_dict['gt_iou_of_rois'][cur_unlabeled_ind]
                    self.val_dict['iou_roi_pl'].extend(cur_iou_roi_pl.tolist())

                    self.val_dict['iou_roi_gt'].extend(preds_iou_max.tolist())

                    cur_pred_score = batch_dict['batch_cls_preds'][cur_unlabeled_ind].squeeze()
                    self.val_dict['pred_scores'].extend(cur_pred_score.tolist())

                    # Here weights=1 unlike our algo
                    cur_weight = torch.ones_like(cur_pred_score)
                    self.val_dict['weights'].extend(cur_weight.tolist())

                    cur_roi_label = batch_dict['roi_labels'][cur_unlabeled_ind].squeeze()
                    self.val_dict['class_labels'].extend(cur_roi_label.tolist())

                    cur_iteration = torch.ones_like(preds_iou_max) * (batch_dict['cur_iteration'])
                    self.val_dict['iteration'].extend(cur_iteration.tolist())

            # replace old pickle data (if exists) with updated one 
            # NOTE : extending the updated data to old one
            output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
            file_path = os.path.join(output_dir, 'ious_baseline.pkl')
            pickle.dump(self.val_dict, open(file_path, 'wb'))

            tb_dict_['pseudo_ious'] = _mean(pseudo_ious)
            tb_dict_['pseudo_accs'] = _mean(pseudo_accs)
            tb_dict_['sem_score_fg'] = _mean(sem_score_fgs)
            tb_dict_['sem_score_bg'] = _mean(sem_score_bgs)

            tb_dict_['max_box_num'] = max_box_num
            tb_dict_['max_pseudo_box_num'] = max_pseudo_box_num

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
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

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

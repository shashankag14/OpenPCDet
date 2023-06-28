import os
import pickle

import torch
import copy
import torch.nn.functional as F

from pcdet.datasets.augmentor import augmentor_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .detector3d_template import Detector3DTemplate
from.pv_rcnn import PVRCNN
from ...utils.stats_utils import KITTIEvalMetrics, PredQualityMetrics
from torchmetrics.collections import MetricCollection
import torch.distributed as dist
# from visual_utils import visualize_utils as V

'''
Add argument --fix_random_seed for non-random initialization of network weights
'''

def _mean(tensor_list):
    tensor = torch.cat(tensor_list)
    tensor = tensor[~torch.isnan(tensor)]
    mean = tensor.mean() if len(tensor) > 0 else torch.tensor([float('nan')])
    return mean

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
        self.metric_registry = MetricRegistry(dataset=self.dataset, model_cfg=model_cfg)

        for param in self.pv_rcnn_ema.parameters():
            param.detach_()
        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)
        self.pretrain = model_cfg.PRETRAINED_MODEL.ENABLE
        if not self.pretrain: # Force Teacher and Student to same random initialization
            pv_rcnn_weights = self.pv_rcnn.state_dict()
            self.pv_rcnn_ema.load_state_dict(pv_rcnn_weights)

        self.thresh = model_cfg.THRESH
        self.sem_thresh = model_cfg.SEM_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.no_nms = model_cfg.NO_NMS
        self.supervise_mode = model_cfg.SUPERVISE_MODE
        vals_to_store = []
        self.val_dict = {val: [] for val in vals_to_store}
        self.classes = ['Car','Ped','Cyc']
        self.ema_template= {val: [] for val in self.classes}
        self.updated_template = {val: [] for val in self.classes}


    def forward(self, batch_dict):
        if self.training:
            labeled_mask = batch_dict['labeled_mask'].view(-1)
            labeled_inds = torch.nonzero(labeled_mask).squeeze(1).long()
            batch_dict['labeled_inds'] = labeled_inds
            unlabeled_inds = torch.nonzero(1-labeled_mask).squeeze(1).long()
            batch_dict['unlabeled_inds'] = unlabeled_inds
            batch_dict_ema = {}
            batch_dict['store_scores_in_pkl'] = self.model_cfg.STORE_SCORES_IN_PKL
            keys = list(batch_dict.keys())
            for k in keys:
                if k + '_ema' in keys:
                    continue
                if k.endswith('_ema'):
                    batch_dict_ema[k[:-4]] = batch_dict[k]
                else:
                    batch_dict_ema[k] = batch_dict[k]

            batch_dict_viewA = copy.deepcopy(batch_dict_ema) # RCNN 
            batch_dict_ema['module_type'] = 'Teacher' # teacher WeakAug
            batch_dict['module_type'] = 'Student' # Strong Aug
            batch_dict_viewA['module_type'] = 'WeakAug' # WeakAug


            with torch.no_grad():
                # self.pv_rcnn_ema.eval()  # Important! must be in train mode
                for cur_module in self.pv_rcnn_ema.module_list:
                    try:
                        batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)

                    except:
                        batch_dict_ema = cur_module(batch_dict_ema)

                pred_dicts, recall_dicts = self.pv_rcnn_ema.post_processing(batch_dict_ema,
                                                                            no_recall_dict=True, override_thresh=0.0, no_nms=self.no_nms)
        
            ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_inds, ...]

            """PL metrics before filtering"""
            if self.model_cfg.ROI_HEAD.get("ENABLE_EVAL", False):
                self.update_metrics(batch_dict, pred_dicts, unlabeled_inds, labeled_inds)

            pseudo_boxes, pseudo_scores, pseudo_sem_scores, *_ = \
                self._filter_pseudo_labels(pred_dicts, unlabeled_inds)

            self._fill_with_pseudo_labels(batch_dict, pseudo_boxes, unlabeled_inds, labeled_inds)

            batch_dict = self.apply_augmentation(batch_dict, batch_dict, unlabeled_inds, key='gt_boxes')

            """PL metrics after filtering"""
            if self.model_cfg.ROI_HEAD.get("ENABLE_EVAL", False):
                if 'pl_gt_metrics_after_filtering' in self.model_cfg.ROI_HEAD.METRICS_PRED_TYPES:

                    ori_unlabeled_boxes_list = [ori_box for ori_box in ori_unlabeled_boxes]
                    pseudo_boxes_list = [ps_box for ps_box in batch_dict['gt_boxes'][unlabeled_inds]]
                    metric_inputs = {'preds': pseudo_boxes_list, 'pred_scores': pseudo_scores, 'roi_scores': pseudo_sem_scores,
                             'ground_truths': ori_unlabeled_boxes_list}
                    
                    tag = f'pl_gt_metrics_after_filtering'
                    metrics = self.metric_registry.get(tag)
                    metrics.update(**metric_inputs)

            batch_dict['metric_registry'] = self.metric_registry
            batch_dict['ori_unlabeled_boxes'] = ori_unlabeled_boxes

            for cur_module in self.pv_rcnn.module_list:
                batch_dict =  cur_module(batch_dict) 
            # For metrics calculation
            self.pv_rcnn.roi_head.forward_ret_dict['unlabeled_inds'] = unlabeled_inds
            self.pv_rcnn.roi_head.forward_ret_dict['pl_boxes'] = batch_dict['gt_boxes']
            self.pv_rcnn.roi_head.forward_ret_dict['pl_scores'] = pseudo_scores
            
            disp_dict = {}
            loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
            loss_rcnn_cls, loss_rcnn_box, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)

            if self.model_cfg.PROTO_INTER_LOSS.ENABLE: 
                with torch.no_grad(): 
                    for cur_module in self.pv_rcnn.module_list:                 # weak augmentation
                        batch_dict_viewA = cur_module(batch_dict_viewA) 


                    batch_dict_viewB = {}                                       #strong augmentation
                    batch_dict_viewB['module_type'] = "StrongAug"
                    batch_dict_viewB['unlabeled_inds'] = batch_dict['unlabeled_inds']
                    batch_dict_viewB['cur_iteration'] = batch_dict['cur_iteration']
                    batch_dict_viewB['labeled_inds'] = batch_dict['labeled_inds']
                    batch_dict_viewB['rois'] = batch_dict['rois'].data.clone()
                    batch_dict_viewB['roi_scores'] = batch_dict['roi_scores'].data.clone()
                    batch_dict_viewB['roi_labels'] = batch_dict['roi_labels'].data.clone()
                    batch_dict_viewB['has_class_labels'] = batch_dict['has_class_labels']
                    batch_dict_viewB['batch_size'] = batch_dict['batch_size']
                    batch_dict_viewB['point_features'] = batch_dict_viewA['point_features'].data.clone()
                    batch_dict_viewB['point_coords'] = batch_dict_viewA['point_coords'].data.clone()
                    batch_dict_viewB['point_cls_scores'] = batch_dict_viewA['point_cls_scores'].data.clone()

                    batch_dict_viewB = self.reverse_augmentation(batch_dict_viewB, batch_dict, unlabeled_inds)
                    batch_dict_viewB = self.pv_rcnn.roi_head.get_strong_ulb_features(batch_dict_viewB) 

                    inter_domain_loss = self.pv_rcnn.roi_head.get_proto_inter_loss(batch_dict_viewA,batch_dict_viewB)

            if self.model_cfg.DYNAMIC_ULB_LOSS_WEIGHT.ENABLE:
                if batch_dict['cur_epoch'] <  self.model_cfg.DYNAMIC_ULB_LOSS_WEIGHT.END_EPOCH:

                    start_weight = self.model_cfg.DYNAMIC_ULB_LOSS_WEIGHT.START_WEIGHT
                    end_weight = self.model_cfg.DYNAMIC_ULB_LOSS_WEIGHT.END_WEIGHT
                    alpha = self.model_cfg.DYNAMIC_ULB_LOSS_WEIGHT.ALPHA
                    step_size = self.model_cfg.DYNAMIC_ULB_LOSS_WEIGHT.STEP_SIZE

                    self.unlabeled_weight = min(start_weight + alpha * math.floor((batch_dict['cur_epoch'] - self.model_cfg.DYNAMIC_ULB_LOSS_WEIGHT.get('START_EPOCH'))/step_size), end_weight)
                    tb_dict['unlabeled_weight'] = self.unlabeled_weight            
                else:
                    self.unlabeled_weight = self.model_cfg.DYNAMIC_ULB_LOSS_WEIGHT.get('END_WEIGHT')
            
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

            if self.model_cfg.PROTO_INTER_LOSS.ENABLE:
                loss = loss_rpn_cls + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box + inter_domain_loss
            else : 
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

            return pred_dicts, recall_dicts, {}
        
    def update_metrics(self, input_dict, pred_dict, unlabeled_inds, labeled_inds):
        """
        Recording PL vs GT statistics BEFORE filtering
        """
        if 'pl_gt_metrics_before_filtering' in self.model_cfg.ROI_HEAD.METRICS_PRED_TYPES:
            pseudo_boxes, pseudo_labels, pseudo_scores, pseudo_sem_scores,*_ = self._unpack_predictions(
                pred_dict, unlabeled_inds)
            pseudo_boxes = [torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1) \
                            for (pseudo_box, pseudo_label) in zip(pseudo_boxes, pseudo_labels)]
            
            pseudo_boxes_numel = [pseudo_box.numel() for pseudo_box in pseudo_boxes]
            if 0 in pseudo_boxes_numel:
                print(" 0 no. element pseudo_box")

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
        
        # # not part of 3dioumatch-org
        # batch_dict[key][unlabeled_inds, :, 6] = common_utils.limit_period(
        #     batch_dict[key][unlabeled_inds, :, 6], offset=0.5, period=2 * np.pi
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
        
        # # not part of 3dioumatch-org
        # batch_dict[key][unlabeled_inds, :, 6] = common_utils.limit_period(
        #     batch_dict[key][unlabeled_inds, :, 6], offset=0.5, period=2 * np.pi
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
        alpha = 0.999
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.pv_rcnn_ema.parameters(), self.pv_rcnn.parameters()):
            ema_param.data.mul_(alpha).add_((1 - alpha)* param.data)

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

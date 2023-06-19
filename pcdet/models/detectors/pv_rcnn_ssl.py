import os
import pickle

import torch
import copy
import torch.nn.functional as F

from pcdet.datasets.augmentor.augmentor_utils import *
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .detector3d_template import Detector3DTemplate
from.pv_rcnn import PVRCNN
# from ...utils.stats_utils import KITTIEvalMetrics, PredQualityMetrics
from torchmetrics.collections import MetricCollection
import torch.distributed as dist
# from visual_utils import visualize_utils as V

def _mean(tensor_list):
    tensor = torch.cat(tensor_list)
    tensor = tensor[~torch.isnan(tensor)]
    mean = tensor.mean() if len(tensor) > 0 else torch.tensor([float('nan')])
    return mean

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

        # Initialize pv_rcnn and pv_rcnn_ema the same way
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
        # with open('ema_sh4468_0.9.pkl','rb') as f:
        #     self.rcnn_features = pickle.loads(f.read())
        # rcnn_sh_mean = []
        # for cls in self.classes:
        #     avg = "mean"
        #     param = "sh"
        #     rcnn_sh_mean.append(self.rcnn_features[cls][avg][param].unsqueeze(dim=0))
        # self.rcnn_sh_mean = torch.stack(rcnn_sh_mean)


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
            batch_dict_viewA['module_type'] = 'StudentViewA' # WeakAug
            batch_dict['module_type'] = 'StudentViewB' # Strong Aug

            with torch.no_grad():
                # self.pv_rcnn_ema.eval()  # Important! must be in train mode
                for cur_module in self.pv_rcnn_ema.module_list:
                    try:
                        batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                        # if 'pooled_features' in batch_dict_ema.keys():
                        #     batch_dict['rois_ema'] = batch_dict_ema['rois'].detach().clone()
                        #     batch_dict['roi_scores_ema'] = batch_dict_ema['roi_scores'].detach().clone()
                        #     batch_dict['roi_labels_ema'] = batch_dict_ema['roi_labels'].detach().clone()
                        #     batch_dict['src_prototype'] +=  batch_dict_ema['pooled_features'][batch_dict['labeled_inds']].mean()
                        #     batch_dict['src_prototype_car'] += batch_dict_ema['src_prototype'][batch_dict['roi_labels_ema'] == 1].mean()
                        #     batch_dict['src_prototype_ped'] += batch_dict_ema['src_prototype'][batch_dict['roi_labels_ema'] == 2].mean()
                        #     batch_dict['src_prototype_cyc'] += batch_dict_ema['src_prototype'][batch_dict['roi_labels_ema'] == 3].mean()

                    except:
                        batch_dict_ema = cur_module(batch_dict_ema)

                pred_dicts, recall_dicts = self.pv_rcnn_ema.post_processing(batch_dict_ema,
                                                                            no_recall_dict=True, override_thresh=0.0, no_nms=self.no_nms)
                
                # src_prototype = src_prototype[labeled_inds, ...]

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
                batch_dict =  cur_module(batch_dict) # calculate + view B (strong aug ulb) prototypes here
            
            
            disp_dict = {}
            loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss(scalar=False)
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict, scalar=False)
            loss_rcnn_cls, loss_rcnn_box, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict, scalar=False)
            
            ##MCL losses
# Source prototypes calculated in below snippet. Pooled_features collected from RCNN of student
            with torch.no_grad():
                for cur_module in self.pv_rcnn.module_list:
                    try:
                        batch_dict_viewA = cur_module(batch_dict_viewA, disable_gt_roi_when_pseudo_labeling=True) # PL matching disabled. calculation of weak aug features
                    except:    
                        batch_dict_viewA = cur_module(batch_dict_viewA)
            
            inter_domain_loss = self.pv_rcnn.roi_head.get_proto_inter_loss()
            
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

            loss = loss_rpn_cls + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box + inter_domain_loss
            
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

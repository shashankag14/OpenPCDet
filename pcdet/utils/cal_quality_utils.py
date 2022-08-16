import copy
import os
import torch
import pickle
import glob
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

import math
import numpy as np
from pcdet.utils import common_utils, box_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


class QualityMetric(object):
    def __init__(self):
        self.tp, self.fp, self.fn, self.gt = [], [], [], []
        self.trans_err, self.scale_err, self.orient_err = [], [], []
        self.assignment_error, self.missed_gt_error, self.classification_error = [], [], []
        self.precision, self.recall = [], []

    # Merged 'cal_scale_diff' and 'cal_tp_metric' of st3d for computing trans_err, orient_err, scale_err
    def cal_diff(self, tp_boxes, gt_boxes, tp):
        assert tp_boxes.shape[0] == gt_boxes.shape[0]

        # L2 distance xy only
        center_distance = torch.norm(tp_boxes[:, :2] - gt_boxes[:, :2], dim=1)
        trans_err = center_distance.sum().item()
        
        # Angle difference
        angle_diff = self.cal_angle_diff(tp_boxes[:, 6], gt_boxes[:, 6])
        assert angle_diff.sum() >= 0
        orient_err = angle_diff.sum().item()
        
        # Scale difference
        aligned_tp_boxes = tp_boxes.detach().clone()
        # shift their center together
        aligned_tp_boxes[:, 0:3] = gt_boxes[:, 0:3]
        # align their angle
        aligned_tp_boxes[:, 6] = gt_boxes[:, 6]
        iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(aligned_tp_boxes[:, 0:7], gt_boxes[:, 0:7])
        max_ious, _ = torch.max(iou_matrix, dim=1)
        scale_err = (1 - max_ious).sum().item()

        return trans_err/tp, orient_err/tp, scale_err/tp


    @staticmethod
    def cor_angle_range(angle):
        """ correct angle range to [-pi, pi]

        Args:
            angle:

        Returns:

        """
        gt_pi_mask = angle > np.pi
        lt_minus_pi_mask = angle < - np.pi
        angle[gt_pi_mask] = angle[gt_pi_mask] - 2 * np.pi
        angle[lt_minus_pi_mask] = angle[lt_minus_pi_mask] + 2 * np.pi

        return angle

    def cal_angle_diff(self, angle1, angle2):
        """ angle is from x to y, anti-clockwise

        """
        angle1 = self.cor_angle_range(angle1)
        angle2 = self.cor_angle_range(angle2)

        diff = torch.abs(angle1 - angle2)
        gt_pi_mask = diff > math.pi
        diff[gt_pi_mask] = 2 * math.pi - diff[gt_pi_mask]

        return diff


class QualityMetricPkl(QualityMetric):
    def update(self, pred_boxes, gt_boxes, iou_thresh=0.7, points=None,
               frame_id=None, idx=None, batch_dict=None):
        tp_boxes, tp_gt_boxes = self.count_tp_fp_fn_gt(
            pred_boxes, gt_boxes, iou_thresh=iou_thresh, points=points
        )
        if tp_boxes is not None and tp_boxes.shape[0] > 0:
            self.cal_tp_metric(tp_boxes, tp_gt_boxes, points=points)


def get_quality_of_single_info(pred_infos, gt_infos, class_name):
    pred_infos = pickle.load(open(pred_infos, 'rb'))
    gt_infos = pickle.load(open(gt_infos, 'rb'))
    gt_annos = [info['annos'] for info in gt_infos]

    ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
        gt_annos, pred_infos, current_classes=['Car']
    )
    print(ap_result_str)

    assert len(pred_infos) == len(gt_annos)
    quality_metric = QualityMetricPkl()
    for pred_info, gt_anno in zip(pred_infos, gt_annos):
        pred_boxes = pred_info['boxes_lidar']
        pred_boxes[:, 2] += pred_boxes[:, 5] / 2

        gt_mask = gt_anno['name'] == class_name

        valid_num = gt_anno['gt_boxes_lidar'].shape[0]
        gt_boxes = gt_anno['gt_boxes_lidar'][gt_mask[:valid_num]]

        assert gt_boxes.shape[0] == gt_mask.sum()
        assert (gt_anno['name'][valid_num:] == 'DontCare').all()
        gt_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes)

        quality_metric.update(pred_boxes, gt_boxes)

    result = quality_metric.statistics_result()
    print(result)


def get_error_of_multiple_infos(info_path_list, gt_info_path, class_name, iou_thresh=0.7):
    pred_info_list = [pickle.load(open(cur_path, 'rb')) for cur_path in info_path_list]
    gt_infos = pickle.load(open(gt_info_path, 'rb'))
    gt_annos = [info['annos'] for info in gt_infos]

    num_infos = len(pred_info_list)
    quality_metric_list = [QualityMetricPkl() for _ in range(num_infos)]

    print(f'------Start to estimate the errors by considering multiple infos (iou_thresh={iou_thresh})..------')
    # import pdb
    # pdb.set_trace()
    for k, gt_anno in enumerate(gt_annos):
        gt_mask = gt_anno['name'] == class_name
        valid_num = gt_anno['gt_boxes_lidar'].shape[0]
        gt_boxes = gt_anno['gt_boxes_lidar'][gt_mask[:valid_num]]
        assert gt_boxes.shape[0] == gt_mask.sum()
        assert (gt_anno['name'][valid_num:] == 'DontCare').all()

        gt_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes)
        gt_boxes, _ = common_utils.check_numpy_to_torch(gt_boxes)

        if gt_boxes.shape[0] == 0:
            continue

        gt_of_tp_mask = np.ones(gt_boxes.shape[0], dtype=np.int)
        for info_idx in range(num_infos):
            pred_boxes = pred_info_list[info_idx][k]['boxes_lidar']
            pred_boxes[:, 2] += pred_boxes[:, 5] / 2
            if pred_boxes.__len__() == 0:
                gt_of_tp_mask[:] = 0
                break

            pred_boxes, _ = common_utils.check_numpy_to_torch(pred_boxes)

            iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes[:, :7].cuda(), gt_boxes[:, :7].cuda())
            max_iou_of_gt, _ = torch.max(iou_matrix, dim=0)
            max_iou_of_gt = max_iou_of_gt.cpu().numpy()
            gt_of_tp_mask[max_iou_of_gt < iou_thresh] = 0

        intersect_gt_boxes = gt_boxes[gt_of_tp_mask > 0]

        for info_idx in range(num_infos):
            pred_boxes = pred_info_list[info_idx][k]['boxes_lidar']
            quality_metric_list[info_idx].update(pred_boxes, intersect_gt_boxes)

    for info_idx in range(num_infos):
        result = quality_metric_list[info_idx].statistics_result()
        print(f'{result} for file: {info_path_list[info_idx]}')

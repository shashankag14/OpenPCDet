import torch
import math
import numpy as np
from pcdet.ops.iou3d_nms import iou3d_nms_utils


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # TODO find avg as per the dtype of metrics. eg. tensors or list
    def update(self, val, n=1):
        ## TODO change it for NaN
        assert np.isscalar(val)
        
        if not np.isnan(val):
            assert val>=0, "Update value should be non negative"
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
            
class Metric(object):
    def __init__(self):
        self.metrics = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 
                        'assignment_err': AverageMeter(), 'cls_err':AverageMeter(),
                        'precision': AverageMeter(), 'recall': AverageMeter(), 'rej_pseudo_lab': AverageMeter()}
    
    def reset(self):
        for key in self.metrics:
            self.metrics[key].reset()

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
        # angle is from x to y, anti-clockwise
        angle1 = self.cor_angle_range(angle1)
        angle2 = self.cor_angle_range(angle2)

        diff = torch.abs(angle1 - angle2)
        gt_pi_mask = diff > math.pi
        diff[gt_pi_mask] = 2 * math.pi - diff[gt_pi_mask]

        return diff    

    def compute_assignment_err(self, tp_boxes, gt_boxes):
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

        return trans_err, orient_err, scale_err 

    # Merged 'cal_scale_diff' and 'cal_tp_metric' of st3d for computing trans_err, orient_err, scale_err
    def update_metrics(self, tp_boxes, gt_boxes):
        tp = self.metrics['tp'].val

        trans_err, orient_err, scale_err = self.compute_assignment_err(tp_boxes, gt_boxes)
        self.metrics['assignment_err'].update(trans_err/tp + orient_err/tp + scale_err/tp)
    
        precision = tp / (tp + self.metrics['fp'].val)
        recall = tp / (tp + self.metrics['fn'].val)
        self.metrics['precision'].update(precision)
        self.metrics['recall'].update(recall)

class MetricRecord(object):
    def __init__(self, num_class):
        self.num_class = num_class
        # self.metric_record = [Metric() for i in range(self.num_class+1)]
        self.metric_record = {'class_agnostic': Metric(), 'car':  Metric(),
                                'cyc': Metric(), 'ped': Metric()}
    
    def reset(self):
        for key in self.metric_record :
               self.metric_record[key].reset()
    
    def update_record(self, pseudo_boxes, gt_boxes, iou_max, fg_thresh, asgn, cls_pseudo):
        # Store tp, fp, fn per batch
        tp_mask = iou_max >= fg_thresh
        gt_labels = gt_boxes[:, 7]
        num_gt_labels_per_class=torch.bincount(gt_labels.type(torch.int64), minlength=4)

        for class_ind, class_key in enumerate(self.metric_record):
            # class agnostic metrics 
            if class_key == 'class_agnostic':
                # update tp, fp, fn
                self.metric_record[class_key].metrics['tp'].update(tp_mask.sum().item())
                self.metric_record[class_key].metrics['fp'].update(iou_max.shape[0] - tp_mask.sum().item())
                self.metric_record[class_key].metrics['fn'].update((num_gt_labels_per_class[1:]).sum().item() - tp_mask.sum().item())
                
                if tp_mask.sum().item() :
                    # get tp boxes and their corresponding gt boxes
                    tp_pseudo_boxes = pseudo_boxes[tp_mask]
                    tp_gt_boxes = gt_boxes[asgn, :][tp_mask]
                    # update assignment error, precision and recall
                    self.metric_record[class_key].update_metrics(tp_pseudo_boxes, tp_gt_boxes)

                # update classification error
                correct_matches = (gt_boxes[:, 7].gather(dim=0, index=asgn) == cls_pseudo).float()
                self.metric_record[class_key].metrics['cls_err'].update((1 - correct_matches.mean()).item())
            
            # update class-wise metrics 
            else :
                # update tp, fp, fn 
                class_mask = cls_pseudo == class_ind
                class_tp_mask = tp_mask & class_mask
                self.metric_record[class_key].metrics['tp'].update(class_tp_mask.sum().item())
                self.metric_record[class_key].metrics['fp'].update(iou_max[class_mask].shape[0] - class_tp_mask.sum().item())
                self.metric_record[class_key].metrics['fn'].update(num_gt_labels_per_class[class_ind].item() - class_tp_mask.sum().item())
                
                if class_tp_mask.sum().item() : 
                    # get tp boxes and their corresponding gt boxes
                    tp_pseudo_boxes_per_class = pseudo_boxes[class_tp_mask]
                    tp_gt_boxes_per_class = gt_boxes[asgn, :][class_tp_mask]
                    # update assignment error, precision and recall   
                    self.metric_record[class_key].update_metrics(tp_pseudo_boxes_per_class, tp_gt_boxes_per_class)

                # update classification error
                cls_err_per_class = (1 - correct_matches[class_mask].mean())  
                self.metric_record[class_key].metrics['cls_err'].update(cls_err_per_class.item())



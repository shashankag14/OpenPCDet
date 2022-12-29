import torch
import os
from pathlib import Path
import shutil
import tqdm
import numpy as np
import torch.distributed as dist

from pcdet.config import cfg
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, commu_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import pickle as pkl

PSEUDO_LABELS_DICT = {}
NEW_PSEUDO_LABELS_DICT = {}

# Adapted from ST3D
def save_pseudo_label_epoch(model, data_loader, rank, leave_pbar, cur_epoch):
    """
    Generate pseudo label with given model.

    Args:
        model: model to predict result for pseudo label
        data_loader: data_loader to predict pseudo label
        rank: process rank
        leave_pbar: tqdm bar controller
        ps_label_dir: dir to save pseudo label
        cur_epoch
        ul_gt_sampler_cfg : config for unlabeled GT Sampler
    """
    pl_dir = cfg.MODEL.UL_GT_SAMPLER['PL_DIR']
    if (Path(pl_dir) / 'pl_database').exists():
        shutil.rmtree(Path(pl_dir) / 'pl_database')

    dataloader_iter = iter(data_loader)
    total_it_each_epoch = int((len(data_loader) / cfg.DATA_CONFIG.REPEAT))

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                        desc='generate_ps_e%d' % cur_epoch, dynamic_ncols=True)
    box_meter = common_utils.AverageMeter()
    
    model.eval()
    for cur_it in range(total_it_each_epoch):
        try:
            batch_dict = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(data_loader)
            batch_dict = next(dataloader_iter)
        
        # Using this as a flag to run pv_rcn_ssl in eval mode
        batch_dict['run_ul_gt_sampler'] = True
        # generate pred boxes for train data
        with torch.no_grad():
            load_data_to_gpu(batch_dict)
            pred_dicts, _, _ = model(batch_dict)

        num_boxes_saved = save_pseudo_label_batch(batch_dict, pred_dicts=pred_dicts)

        # log to console and tensorboard
        box_meter.update(num_boxes_saved)
        disp_dict = {'num_box': "{}".format(box_meter.val), 'total_num_box': "{}".format(box_meter.sum)}
        
        if rank == 0:
            pbar.update()
            pbar.set_postfix(disp_dict)
            pbar.refresh()
    if rank == 0:
        pbar.close()

    batch_dict.pop('run_ul_gt_sampler')
    
    # gather boxes from different processes (for distributed setting)
    gather_and_dump_pseudo_label_result(rank, cur_epoch)

def save_pseudo_label_batch(batch_dict, pred_dicts=None):
    """
    Save pseudo label for give batch.
    If model is given, use model to inference pred_dicts,
    otherwise, directly use given pred_dicts.

    Args:
        input_dict: batch data read from dataloader
        pred_dicts: Dict if not given model.
            predict results to be generated pseudo label and saved
    """
    box_meter = common_utils.AverageMeter()

    labeled_mask = batch_dict['labeled_mask'].view(-1)
    unlabeled_inds = torch.nonzero(1-labeled_mask).squeeze(1).long()
    # NOTE(shashank) : sem_scores not available. Filtering only being done wrt objectness scores
    for b_idx in unlabeled_inds:
        if 'pred_boxes' in pred_dicts[b_idx]:
            # Exist predicted boxes passing self-training score threshold
            pseudo_scores = pred_dicts[b_idx]['pred_scores']
            pseudo_boxes = pred_dicts[b_idx]['pred_boxes']
            pseudo_labels = pred_dicts[b_idx]['pred_labels']

            # filter preds based on objectness scores
            conf_thresh = torch.tensor(cfg.MODEL.UL_GT_SAMPLER.THRESH, device=pseudo_labels.device).unsqueeze(
                                0).repeat(len(pseudo_labels), 1).gather(dim=1, index=(pseudo_labels - 1).unsqueeze(-1))
            valid_inds = pseudo_scores > conf_thresh.squeeze()
            
            pseudo_labels = pseudo_labels[valid_inds].detach().cpu().numpy()
            pseudo_scores = pseudo_scores[valid_inds].detach().cpu().numpy()
            pseudo_boxes = pseudo_boxes[valid_inds].detach().cpu().numpy()

            if valid_inds.shape[0] > 0:
                gt_boxes = np.concatenate((pseudo_boxes,
                                        pseudo_labels.reshape(-1, 1),
                                        pseudo_scores.reshape(-1, 1)), axis=1)
                
                create_pl_database(gt_boxes, batch_dict['frame_id'][b_idx])
                box_meter.update(gt_boxes.shape[0])    
    
    # for k, v in NEW_PSEUDO_LABELS_DICT.items():
    #     print('Database %s: %d' % (k, len(v)))

    return box_meter.sum

def gather_and_dump_pseudo_label_result(rank, cur_epoch):
    commu_utils.synchronize()
    if dist.is_initialized():
        part_pseudo_labels_list = commu_utils.all_gather(NEW_PSEUDO_LABELS_DICT)
        new_pseudo_label_dict = {}
        for pseudo_labels in part_pseudo_labels_list:
            new_pseudo_label_dict.update(pseudo_labels)
        NEW_PSEUDO_LABELS_DICT.update(new_pseudo_label_dict)

    # dump new pseudo label to given dir
    if rank == 0:
        ps_path = os.path.join(cfg.MODEL.UL_GT_SAMPLER['PL_DIR'], "ps_label_e{}.pkl".format(cur_epoch))
        with open(ps_path, 'wb') as f:
            pkl.dump(NEW_PSEUDO_LABELS_DICT, f)

    commu_utils.synchronize()
    PSEUDO_LABELS_DICT.clear()
    PSEUDO_LABELS_DICT.update(NEW_PSEUDO_LABELS_DICT)
    NEW_PSEUDO_LABELS_DICT.clear()

# Adapted from create_groundtruth_database in kitti_dataset_ssl.py        
def create_pl_database(gt_boxes, frame_id):
    # Saving points inside boxes as .bin inside 'output/pl_database' folder
    database_save_path = Path(cfg.MODEL.UL_GT_SAMPLER['PL_DIR']) / 'pl_database'
    database_save_path.mkdir(parents=True, exist_ok=True)

    class_to_name = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}
    num_obj = gt_boxes.shape[0]

    points = get_lidar(frame_id)
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
        torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes[:,:7])
        ).numpy()  # (nboxes, npoints)

    # iterate over each bbox, save its corresponding points as .bin and append its infos in dictionary
    for gt_idx in range(num_obj):
        cls_name = class_to_name[gt_boxes[gt_idx, 7]]
        bbox = np.zeros((4))    # represents 2D bbox info which is not required here
        score = gt_boxes[gt_idx, -1]
        # gt_boxes = annos['gt_boxes_lidar']
    
        # Dump points inside the box in .bin
        filename = '%s_%s_%d.bin' % (frame_id, cls_name, gt_idx)
        filepath = database_save_path / filename
        gt_points = points[point_indices[gt_idx] > 0]
        gt_points[:, :3] -= gt_boxes[gt_idx, :3]
        with open(filepath, 'w') as f:
            gt_points.tofile(f)

        db_path = str(filepath.relative_to(cfg.MODEL.UL_GT_SAMPLER['PL_DIR']))  # gt_database/xxxxx.bin
        db_info = {'name': cls_name, 'path': db_path, 'image_idx': frame_id, 'gt_idx': gt_idx,
                    'box3d_lidar': gt_boxes[gt_idx], 'num_points_in_gt': gt_points.shape[0],
                    'difficulty': -1, 'bbox': bbox, 'score': score}
        
        if cls_name in NEW_PSEUDO_LABELS_DICT:
            NEW_PSEUDO_LABELS_DICT[cls_name].append(db_info)
        else:
            NEW_PSEUDO_LABELS_DICT[cls_name] = [db_info]

def get_lidar(frame_id):
    lidar_file = cfg.ROOT_DIR / 'data/kitti/training/velodyne' / ('%s.bin' % frame_id)
    assert lidar_file.exists()
    return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
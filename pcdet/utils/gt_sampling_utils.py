import torch
import os
import glob
import tqdm
import numpy as np
import torch.distributed as dist

from pcdet.config import cfg
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, commu_utils
import pickle as pkl

PSEUDO_LABELS_DICT = {}
NEW_PSEUDO_LABELS_DICT = {}

def save_pseudo_label_epoch(model, data_loader, rank, leave_pbar, cur_epoch, ul_gt_sampler_cfg):
    """
    Generate pseudo label with given model.

    Args:
        model: model to predict result for pseudo label
        val_loader: data_loader to predict pseudo label
        rank: process rank
        leave_pbar: tqdm bar controller
        ps_label_dir: dir to save pseudo label
        cur_epoch
    """
    #TODO: check for pl_dir not none

    # TODO : only use unlabeled data
    dataloader_iter = iter(data_loader)
    total_it_each_epoch = len(data_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                        desc='generate_ps_e%d' % cur_epoch, dynamic_ncols=True)
    ps_meter = common_utils.AverageMeter()
    
    model.eval()
    for cur_it in range(total_it_each_epoch):
        try:
            batch_dict = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(data_loader)
            batch_dict = next(dataloader_iter)
        
        batch_dict['run_ul_gt_sampler'] = True
        # generate gt_boxes for train data
        with torch.no_grad():
            load_data_to_gpu(batch_dict)
            pred_dicts, ret_dict, _ = model(batch_dict)

        ps_batch = save_pseudo_label_batch(
            batch_dict, ul_gt_sampler_cfg, 
            pred_dicts=pred_dicts
        )
        # log to console and tensorboard
        ps_meter.update(ps_batch)
        disp_dict = {'ps_box': "{:.3f}({:.3f})".format(ps_meter.val, ps_meter.sum)}
        
        if rank == 0:
            pbar.update()
            pbar.set_postfix(disp_dict)
            pbar.refresh()
    if rank == 0:
        pbar.close()
    batch_dict.pop('run_ul_gt_sampler')
    gather_and_dump_pseudo_label_result(rank, ul_gt_sampler_cfg.PL_DIR, cur_epoch)


def save_pseudo_label_batch(batch_dict,
                            ul_gt_sampler_cfg,
                            pred_dicts=None):
    """
    Save pseudo label for give batch.
    If model is given, use model to inference pred_dicts,
    otherwise, directly use given pred_dicts.

    Args:
        input_dict: batch data read from dataloader
        pred_dicts: Dict if not given model.
            predict results to be generated pseudo label and saved
    """
    ps_meter = common_utils.AverageMeter()

    labeled_mask = batch_dict['labeled_mask'].view(-1)
    unlabeled_inds = torch.nonzero(1-labeled_mask).squeeze(1).long()
    # TODO : sem_scores not available. Filtering only being done wrt objectness scores
    for b_idx in unlabeled_inds:
        if 'pred_boxes' in pred_dicts[b_idx]:
            # Exist predicted boxes passing self-training score threshold
            pseudo_scores = pred_dicts[b_idx]['pred_scores']#.detach().cpu().numpy()
            pseudo_boxes = pred_dicts[b_idx]['pred_boxes']#.detach().cpu().numpy()
            pseudo_labels = pred_dicts[b_idx]['pred_labels']#.detach().cpu().numpy()
            # pseudo_sem_scores = pred_dicts[b_idx]['pred_sem_scores'].detach().cpu().numpy()

            # filter preds based on objectness and semantic thresholding
            conf_thresh = torch.tensor(ul_gt_sampler_cfg.THRESH, device=pseudo_labels.device).unsqueeze(
                                0).repeat(len(pseudo_labels), 1).gather(dim=1, index=(pseudo_labels - 1).unsqueeze(-1))
            valid_inds = pseudo_scores > conf_thresh.squeeze()
            # valid_inds = valid_inds * (pseudo_sem_scores > ul_gt_sampler_cfg.sem_thresh[0])
            
            pseudo_labels = pseudo_labels[valid_inds].detach().cpu().numpy()
            pseudo_scores = pseudo_scores[valid_inds].detach().cpu().numpy()
            pseudo_boxes = pseudo_boxes[valid_inds].detach().cpu().numpy()

            if valid_inds.shape[0] > 0:
                gt_box = np.concatenate((pseudo_boxes,
                                        pseudo_labels.reshape(-1, 1),
                                        pseudo_scores.reshape(-1, 1)), axis=1)
            else:
                gt_box = np.zeros((0, 9), dtype=np.float32)
        else:
            gt_box = np.zeros((0, 9), dtype=np.float32)

        gt_infos = {'gt_boxes': gt_box}
        ps_meter.update(gt_infos['gt_boxes'].shape[0])

        NEW_PSEUDO_LABELS_DICT[batch_dict['frame_id'][b_idx]] = gt_infos

    return ps_meter.sum

def gather_and_dump_pseudo_label_result(rank, ps_label_dir, cur_epoch):
    commu_utils.synchronize()
    if dist.is_initialized():
        part_pseudo_labels_list = commu_utils.all_gather(NEW_PSEUDO_LABELS_DICT)
        new_pseudo_label_dict = {}
        for pseudo_labels in part_pseudo_labels_list:
            new_pseudo_label_dict.update(pseudo_labels)
        NEW_PSEUDO_LABELS_DICT.update(new_pseudo_label_dict)

    # dump new pseudo label to given dir
    if rank == 0:
        ps_path = os.path.join(ps_label_dir, "ps_label_e{}.pkl".format(cur_epoch))
        with open(ps_path, 'wb') as f:
            pkl.dump(NEW_PSEUDO_LABELS_DICT, f)

    commu_utils.synchronize()
    PSEUDO_LABELS_DICT.clear()
    PSEUDO_LABELS_DICT.update(NEW_PSEUDO_LABELS_DICT)
    NEW_PSEUDO_LABELS_DICT.clear()

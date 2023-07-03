import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


class PVRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg,
                         predict_boxes_when_training=predict_boxes_when_training)
        self.model_cfg = model_cfg
        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

        self.print_loss_when_eval = False
        self.class_dict = {1:'Car', 2 :'Ped', 3:'Cyc'}
        self.prototype = {'Car': None, 'Ped' : None, 'Cyc' : None}
        self.momentum = self.model_cfg.PROTOTYPE.MOMENTUM
        self.start_iter = self.model_cfg.PROTOTYPE.START_ITER

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['gt_boxes'] if 'create_prototype' in batch_dict else batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)
        batch_dict["weighted_point_features"] = point_features
        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict, disable_gt_roi_when_pseudo_labeling=False):
        """
        :param input_data: input dict
        :return:
        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training and not disable_gt_roi_when_pseudo_labeling else 'TEST']
        )

        # should not use gt_roi for pseudo label generation
        if (self.training or self.print_loss_when_eval) and not disable_gt_roi_when_pseudo_labeling:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
        
        if self.training and self.model_cfg.PROTO_INTER_LOSS.ENABLE:
            
            batch_dict['pooled_features'] =  pooled_features.view(batch_dict['batch_size'],batch_dict['roi_labels'].shape[1],-1, grid_size, grid_size, grid_size)
            batch_dict['pooled_features_lbl'] = batch_dict['pooled_features'][batch_dict['labeled_inds']]
            batch_dict['pooled_features_ulb'] =  batch_dict['pooled_features'][batch_dict['unlabeled_inds']]

            
            if batch_dict['module_type'] == 'WeakAug':
                self.prototype = self.calc_prototype(batch_dict)
                output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                file_path = os.path.join(output_dir, 'prototypes.pkl')
                pickle.dump(self.prototype, open(file_path, 'wb'))

                features_weak_ulb = self.get_features(batch_dict,labeled=False)
                batch_dict["prototype"] = self.prototype
                batch_dict["features_weak_ulb"] = features_weak_ulb

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        if self.training or self.print_loss_when_eval:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict



    def get_strong_ulb_features(self,batch_dict):
        assert batch_dict['module_type'] == "StrongAug"
        features_strong_ulb = self.get_features(batch_dict,labeled=False) 
        batch_dict["features_strong_ulb"] = features_strong_ulb
        return batch_dict

    #Compute current pooled_features. Set labeled on to get current labeled features, else off for unlabeled
    def get_features(self,batch_dict,labeled):
        current_features = {'Car': None, 'Ped' : None, 'Cyc' : None}
        
        # get features for ROIs of unlabeled data (viewB)
        if batch_dict['module_type'] == "StrongAug":    
            batch_dict = self._get_pooled_features(batch_dict)

        for cls_idx, cls_name in enumerate(list(self.class_dict.values())):
            # get features for GTs of labeled data (for prototypes)
            if labeled==True and batch_dict['module_type'] == "WeakAug":
                feature_type ='pooled_features_lbl'
                cls_mask = (batch_dict['gt_boxes'][batch_dict['labeled_inds'], :, -1] == (cls_idx+1)).flatten()
            # get features for ROIs of unlabeled data (viewA)
            else:
                feature_type = 'pooled_features_ulb'
                cls_mask = (batch_dict['roi_labels'][batch_dict['unlabeled_inds'], :] == (cls_idx+1)).flatten()

            # B - batch size, N - No. of roi/gt
            # C - No. of channels in features (128), G - grid size (6)
            (B,N,C,G,G,G) = batch_dict[feature_type].size()         #shape - (B,N,128,6,6,6))
            features = batch_dict[feature_type].view(B*N, C*G*G*G)  #shape - (B*N, 27648 (128*6*6*6))
            
            # Fetch classwise features 
            current_features[cls_name] = features[cls_mask, ...] 

            # TODO(shashank) : If ft. are aritficially filled like this way then 
            # later there should be a check for such a scenario while computing cos_sim
            if current_features[cls_name].numel() == 0: 
                current_features[cls_name] = torch.zeros((1, C*G*G*G)).to(device=batch_dict['pooled_features'].device)
        
        return current_features

    # Call current_features for labeled  and update labeled prototype 
    # NOTE : Prototype created using GT boxes and labels for labeled entries.
    def calc_prototype(self, batch_dict):
        # get pooled features from viewA
        batch_dict['create_prototype'] = True
        batch_dict = self._get_pooled_features(batch_dict)  
        batch_dict.pop('create_prototype')
        
        # get features for GTs using above pooled features
        current_features = self.get_features(batch_dict, labeled=True)
        
        # classwise mean of features across all GTs 
        for cls_name in list(self.class_dict.values()):
            current_features[cls_name] =  (current_features[cls_name].mean(dim=0)).detach().cpu()  

        # update prototype with current labeled features using EMA
        if batch_dict['cur_iteration']< self.start_iter: 
            self.prototype = current_features
        else :
            for cls_name in list(self.class_dict.values()):
                self.prototype[cls_name] = self.momentum * self.prototype[cls_name] + (1 - self.momentum) * current_features[cls_name]        
        return self.prototype

    # Called when computing unlabeled_features for strong_aug. Pass through ROI_Grid_pool and return to get_features() 
    def _get_pooled_features(self,batch_dict):

        pooled_features = self.roi_grid_pool(batch_dict) 
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
    
        
        if 'create_prototype' in batch_dict:
            batch_dict['pooled_features'] =  pooled_features.view(batch_dict['batch_size'],batch_dict['gt_boxes'].shape[1],-1, grid_size, grid_size, grid_size)
            batch_dict['pooled_features_lbl'] =  batch_dict['pooled_features'][batch_dict['labeled_inds']]
        else:
            batch_dict['pooled_features'] =  pooled_features.view(batch_dict['batch_size'],batch_dict['roi_labels'].shape[1],-1, grid_size, grid_size, grid_size)
            batch_dict['pooled_features_ulb'] =  batch_dict['pooled_features'][batch_dict['unlabeled_inds']]

        return batch_dict

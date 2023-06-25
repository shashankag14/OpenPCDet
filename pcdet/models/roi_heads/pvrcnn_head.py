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
        self.src_prototypeViewB = {'Car': None, 'Ped' : None, 'Cyc' : None}
        self.target_prototypeViewB = {'Car': None, 'Ped' : None, 'Cyc' : None}
        self.src_prototypeViewA = {'Car': None, 'Ped' : None, 'Cyc' : None}
        self.target_prototypeViewA = {'Car': None, 'Ped' : None, 'Cyc' : None}
        self.source_prototypeViewBA = {'Car': None, 'Ped' : None, 'Cyc' : None} 
        self.target_prototypeViewBA = {'Car': None, 'Ped' : None, 'Cyc' : None} #Corresponding prototype 
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
        rois = batch_dict['rois']
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
        '''
        if feature_augBA flag is set as true, we are already going to provide rois, point_features,
        '''
        # if batch_dict['module_type'] != "StudentViewB": 
        # use test-time nms for pseudo label generation
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
        
        batch_dict['pooled_features'] =  pooled_features.view(batch_dict['batch_size'],batch_dict['roi_labels'].shape[1],-1, grid_size, grid_size, grid_size)
        batch_dict['pooled_features_lbl'] = batch_dict['pooled_features'][batch_dict['labeled_inds']]
        batch_dict['pooled_features_ulb'] =  batch_dict['pooled_features'][batch_dict['unlabeled_inds']]


        if not batch_dict['module_type'] == 'Teacher':
            if batch_dict['module_type'] == 'StudentViewA':
                self.src_prototypeViewA,self.target_prototypeViewA = self.calc_prototype(batch_dict)
            elif batch_dict['module_type'] == 'Student': 
                self.src_prototypeB,self.target_prototypeViewB = self.calc_prototype(batch_dict)
            else:
                print(batch_dict['module_type'])
                raise ValueError("Incorrect prototype calculation!")

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

#TODO - keep source meaned to classwise, unlabeled non-mean. Ablation  mean
    def calc_prototype(self,batch_dict):
        if batch_dict['module_type'] == "Student": # Strong augmentation through Student
            src_prototype = self.src_prototypeViewB # Currently unused, TODO : ablation
            tar_prototype = self.target_prototypeViewB #strong target prototype

        elif batch_dict['module_type']=="StudentViewA": # Weak augmentation through Student
            src_prototype = self.src_prototypeViewA # source prototype
            tar_prototype = self.target_prototypeViewA

        elif batch_dict['module_type']=="StudentViewB": # ViewB proposals, weak augmented, passed through Student. Maintains correspondence
            src_prototype = self.source_prototypeViewBA
            tar_prototype = self.target_prototypeViewBA   # weak target prototype
        else :
            raise ValueError("Incorrect prototype calculation!")

        # Generate Source classwise prototype ()
        for i in range(1, len(self.class_dict)+1):
            cls_mask = batch_dict['roi_labels'][batch_dict['labeled_inds']] == i
            key = self.class_dict[i]
            cur_proto = (batch_dict['pooled_features'][batch_dict['labeled_inds']][cls_mask]).mean(dim=0)
            if batch_dict['cur_iteration']< self.start_iter: 
                src_prototype[key] = cur_proto
            else :
                src_prototype[key] = self.momentum * src_prototype[key] + (1 - self.momentum) * cur_proto
        
        #Generate Target classwise prototype ()
        for i in range(1, len(self.class_dict)+1):
            cls_mask = batch_dict['roi_labels'][batch_dict['unlabeled_inds']] == i
            key = self.class_dict[i]
            cur_proto = (batch_dict['pooled_features'][batch_dict['unlabeled_inds']][cls_mask]).mean(dim=0)
            if batch_dict['cur_iteration']< self.start_iter:
               tar_prototype[key] = cur_proto
            else :
               tar_prototype[key] = self.momentum *tar_prototype[key] + (1 - self.momentum) * cur_proto   
        return src_prototype,tar_prototype

    def proto_WeakB(self,batch_dict):
        assert batch_dict['module_type'] == "StudentViewB"
        pooled_features = self.roi_grid_pool(batch_dict) 
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
    
        batch_dict['pooled_features'] =  pooled_features.view(batch_dict['batch_size'],batch_dict['roi_labels'].shape[1],-1, grid_size, grid_size, grid_size)
        batch_dict['pooled_features_lbl'] = batch_dict['pooled_features'][batch_dict['labeled_inds']]
        batch_dict['pooled_features_ulb'] =  batch_dict['pooled_features'][batch_dict['unlabeled_inds']]

        self.source_prototypeViewBA, self.target_prototypeViewBA = self.calc_prototype(batch_dict) # (BxN, 6x6x6, C)

        return batch_dict


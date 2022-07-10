import torch
import torch.nn as nn

import sys
import time
from utils import group_points_2, group_points_fast,KNN_interpolate


def make_mlp_layers(mlp_num, reg=False, reg_output=None):
    layers = []
    for i in range(len(mlp_num) - 1):
        layers.append(
            nn.Conv2d(mlp_num[i], mlp_num[i + 1], kernel_size=(1, 1)),
        )
        layers.append(
            nn.BatchNorm2d(mlp_num[i + 1])
        )
        layers.append(
            nn.ReLU(inplace=True)
        )
    if reg:
        layers.append(
            nn.Conv2d(mlp_num[-1], reg_output, kernel_size=(1, 1)),
        )
    return nn.Sequential(*layers)


class SA_BasicBlock(nn.Module):
    def __init__(self, mlp, knn_K, end_block=False, sample_num=None):
        super(SA_BasicBlock, self).__init__()
        self.mlp_layers = make_mlp_layers(mlp)
        self.max_pool = nn.MaxPool2d((1, knn_K), stride=1)
        if end_block:
            self.max_pool = nn.MaxPool2d((sample_num, 1), stride=1)

    def forward(self, x):
        # # x: B * in_feature_size * centroid_num * knn
        # out = self.layer1(x)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # # out: B * mlp[3] * centroid_num * knn
        # out = self.max_pool(out)
        # # out: B * mlp[3] * centroid_num * 1
        out = self.mlp_layers(x)
        out = self.max_pool(out)

        return out

class FP_BasicBlock(nn.Module):
    def __init__(self, mlp, k):
        super(FP_BasicBlock, self).__init__()
        self.mlp_layers = make_mlp_layers(mlp)
        self.k = k

    def forward(self, unknown_coords, known_coords, unknown_feats, known_feats):
        '''
        :param unknown_coords: B * 3 * n, the coordinates whose features need to be interpolated
        :param known_coords: B * 3 * m, the coordinates of known feature points
        :param unknown_feats: B * C1 * n, the feature that need to be interpolated
        :param known_feats: B * C2 * m, the known features
        :return: new_feats: B * C * n, C is the required number of the feature
        '''
        if known_coords is None:
            inter_feats = known_feats.expand(known_feats.shape[0], known_feats.shape[1], unknown_coords.shape[2])
        else:
            inter_feats = KNN_interpolate(unknown_coords, known_coords, known_feats, self.k)

        if unknown_feats is None:
            new_feats = inter_feats
        else:
            new_feats = torch.cat((inter_feats, unknown_feats), dim=1)

        new_feats = new_feats.unsqueeze(-1)     # B * (C1+C2) * n * 1, match the dims of mlp
        new_feats = self.mlp_layers(new_feats)

        return new_feats.squeeze(-1) 
        

class net_G(nn.Module):
    def __init__(self, cfg, activation=True, vec=False):
        super(net_G, self).__init__()
        self.mlps = cfg.MODEL.POINTNET.EXTRA.NETG_MLP
        self.activation = activation
        self.joint_num = cfg.MODEL.NUM_JOINTS

        layers = [make_mlp_layers(self.mlps),
                  nn.Dropout(p=0.5),
                  ]
        if vec:
            layers.append(nn.Conv2d(self.mlps[-1], self.joint_num*3, kernel_size=(1, 1)))
        else:
            layers.append(nn.Conv2d(self.mlps[-1], self.joint_num, kernel_size=(1, 1)))

        if self.activation:
            layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class pose_net_v2(nn.Module):
    def __init__(self, cfg, sa_block, fp_block):
        super(pose_net_v2, self).__init__()
        sa_mlps = cfg.MODEL.POINTNET.EXTRA.SA_MLPS
        fp_mlps = cfg.MODEL.POINTNET.EXTRA.FP_MLPS

        self.joint_num = cfg.MODEL.NUM_JOINTS
        self.knn_K = cfg.POINT_CLOUD.KNN
        self.insert_k = cfg.POINT_CLOUD.INSERT_K
        self.ball_radius = cfg.POINT_CLOUD.BALL_RADIUS
        self.level_sample_nums = [cfg.MODEL.POINTNET.INPUT_NUM] + cfg.POINT_CLOUD.GROUP_NUM
        self.INPUT_FEAT_NUM_2 = 6+128+self.joint_num*4

        self.SA11 = sa_block([6] + sa_mlps[0], self.knn_K)
        self.SA12 = sa_block([131] + sa_mlps[1], self.knn_K)
        self.SA13 = sa_block([259] + sa_mlps[2], self.knn_K, end_block=True, sample_num=self.level_sample_nums[2])

        self.FP11 = fp_block([1280] + fp_mlps[0], self.insert_k)
        self.FP12 = fp_block([384] + fp_mlps[1], self.insert_k)
        self.FP13 = fp_block([128 + 6] + fp_mlps[2], self.insert_k)

        self.netG11 = net_G(cfg)
        self.netG12 = net_G(cfg, activation=False, vec=True)

        self.SA21 = sa_block([self.INPUT_FEAT_NUM_2] + sa_mlps[0], self.knn_K)
        self.SA22 = sa_block([131] + sa_mlps[1], self.knn_K)
        self.SA23 = sa_block([259] + sa_mlps[2], self.knn_K, end_block=True, sample_num=self.level_sample_nums[2])

        self.FP21 = fp_block([1280] + fp_mlps[0], self.insert_k)
        self.FP22 = fp_block([384] + fp_mlps[1], self.insert_k)
        self.FP23 = fp_block([128+self.INPUT_FEAT_NUM_2] + fp_mlps[2], self.insert_k)

        self.netG21 = net_G(cfg)
        self.netG22 = net_G(cfg, activation=False, vec=True)

    def forward(self, x, idx1 = None, idx2 = None):
        # SA forward
        # x: B * 6 * 1024
        if idx1 is not None:
            inputs_level1_idx = idx1
            points_l1, centroid1 = group_points_fast(x, self.level_sample_nums[1], self.knn_K, inputs_level1_idx)

        else:
            points_l1, centroid1, inputs_level1_idx = \
                group_points_2(x, self.level_sample_nums[0], self.level_sample_nums[1], self.knn_K, self.ball_radius[0]) 
                 # x: B * 6 * 512 * Knn_K, y: B * 3 * 512 * 1# points_l1 : B*6*512*knn  centorid1 : B*3*512*1 input_level_idx : B*512*knn  knn:64
            

        points_l1 = self.SA11(points_l1)  # x: B * 128 * 512 * 1

        points_l2 = torch.cat((centroid1, points_l1), dim=1).squeeze(-1)  # x: B * 3+128 * 512

        if idx2 is not None:
            inputs_level2_idx = idx2
            points_l2, centroid2 = group_points_fast(points_l2, self.level_sample_nums[2], self.knn_K, inputs_level2_idx)
        else: 
            points_l2, centroid2, inputs_level2_idx = \
                group_points_2(points_l2, self.level_sample_nums[1], self.level_sample_nums[2], self.knn_K, self.ball_radius[1])  
            # input_level2: B * 3+128  * 128 * 64, sa2_centroid: B * 3 * 128 * 1
            
        points_l2 = self.SA12(points_l2)  # x: B * 256 * 128 * 1

        points_l3 = torch.cat((centroid2, points_l2), dim=1)  # x: B * 3+256 * 128 * 1
        points_l3 = self.SA13(points_l3)  # out: B * 1024 * 1 * 1

        # FP forward
        centroid0 = x[:, :3, :]
        centroid1 = centroid1.squeeze(-1)
        centroid2 = centroid2.squeeze(-1)
        centroid3 = None
        points_l0 = x
        points_l1 = points_l1.squeeze(-1)
        points_l2 = points_l2.squeeze(-1)
        points_l3 = points_l3.squeeze(-1)

        points_l2 = self.FP11(centroid2, centroid3, points_l2, points_l3)
        points_l1 = self.FP12(centroid1, centroid2, points_l1, points_l2)
        points_l0 = self.FP13(centroid0, centroid1, points_l0, points_l1)  # B * 128 * 1024

        heat_map = self.netG11(points_l0.unsqueeze(-1))
        heat_map = heat_map.squeeze(-1).unsqueeze(2)  # B * JOINT_NUM * 1 * 1024

        vec_map = self.netG12(points_l0.unsqueeze(-1))
        vec_map = vec_map.squeeze(-1).view(-1, self.joint_num, 3, vec_map.size(2))  # B * JOINT_NUM * 3 * 1024

        hm1 = torch.cat((heat_map, vec_map), 2)

        x = torch.cat((x, points_l0, hm1.view(-1, self.joint_num*4, hm1.size(3))), 1)
        # B * (128+6+4*JOINT_NUM) * 1024

        points_l1, centroid1 = \
            group_points_fast(
                x, self.level_sample_nums[1], self.knn_K, inputs_level1_idx
            )  # x: B * (6+128+joint_num*4) * 512 * Knn_K, y: B * 3 * 512 * 1
        points_l1 = self.SA21(points_l1)  # x: B * 128 * 512

        points_l2 = torch.cat((centroid1, points_l1), dim=1).squeeze(-1)  # x: B * 3+128 * 512
        points_l2, centroid2 = \
            group_points_fast(
                points_l2, self.level_sample_nums[2], self.knn_K, inputs_level2_idx
            )  # input_level2: B * 3+128  * 128 * 64, sa2_centroid: B * 3 * 128 * 1
        points_l2 = self.SA22(points_l2)  # x: B * 256 * 128 * 1

        points_l3 = torch.cat((centroid2, points_l2), dim=1)  # x: B * 3+256 * 128 * 1
        points_l3 = self.SA23(points_l3)  # out: B * 1024 * 1 * 1

        # FP forward
        centroid0 = x[:, :3, :]
        centroid1 = centroid1.squeeze(-1)
        centroid2 = centroid2.squeeze(-1)
        centroid3 = None
        points_l0 = x
        points_l1 = points_l1.squeeze(-1)
        points_l2 = points_l2.squeeze(-1)
        points_l3 = points_l3.squeeze(-1)

        points_l2 = self.FP21(centroid2, centroid3, points_l2, points_l3)
        points_l1 = self.FP22(centroid1, centroid2, points_l1, points_l2)
        points_l0 = self.FP23(centroid0, centroid1, points_l0, points_l1)  # B * 128 * 1024

        heat_map = self.netG21(points_l0.unsqueeze(-1))
        heat_map = heat_map.squeeze(-1).unsqueeze(2)  # B * JOINT_NUM * 1 * 1024

        vec_map = self.netG22(points_l0.unsqueeze(-1))
        vec_map = vec_map.squeeze(-1).view(-1, self.joint_num, 3, vec_map.size(2))  # B * JOINT_NUM * 3 * 1024

        hm2 = torch.cat((heat_map, vec_map), 2)
        return [hm1, hm2]


def get_pose_net(cfg):
    model = pose_net_v2(cfg, sa_block=SA_BasicBlock, fp_block=FP_BasicBlock)

    return model
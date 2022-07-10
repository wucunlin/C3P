import numpy as np
import torch

def mpjpe(pred_3d,gt_3d):
    #pred_3d=pred_3d.reshape((-1,n_joints,3))
    #gt_3d=gt_3d.reshape((-1,n_joints,3))
    cnt=pred_3d.shape[0]
    pjpe=0
    for i in range(pred_3d.shape[0]):
        err=(((gt_3d[i] - pred_3d[i]) ** 2).sum(axis=1) ** 0.5).mean()
        pjpe+=err
    mean_pjpe=pjpe/cnt
    return mean_pjpe#,cnt

def ap10cm(pred_3d,gt_3d):
    cnt = pred_3d.shape[0]
    joint_num=pred_3d.shape[1]
    right_num=0
    wrong_num=0
    for i in range(cnt):
        err=(((gt_3d[i] - pred_3d[i]) ** 2).sum(axis=1) ** 0.5)
        for j in range(err.shape[0]):
            if err[j]>0.1:
                wrong_num+=1
            else :
                right_num+=1
    ap=right_num/cnt/joint_num
    return ap

def obb_coord2cs_coord(batch_coords, rotation, offset, max_obb_len):
    batch_size = batch_coords.shape[0]
    num_joints = batch_coords.shape[1]
    batch_coords_cs = (batch_coords + offset.view(-1, 1, 3)) * max_obb_len.view(-1, 1, 1) 
    batch_coords_cs = torch.matmul(batch_coords_cs, rotation.permute(0, 2, 1))
    return batch_coords_cs

def xyz_form_heatmap_ball_truelen(est_heatmap,points,true_r,obb_len,candidate_num):
    xyz_differ_with_truer = (1-est_heatmap[:,:,0,:])*true_r
    xyz_diff = torch.div(xyz_differ_with_truer, obb_len.unsqueeze(-1).expand(-1,xyz_differ_with_truer.size(1)).unsqueeze(-1).expand(-1,-1,xyz_differ_with_truer.size(-1)))
    xyz_diff = xyz_diff.unsqueeze(2).expand(-1,-1,3,-1)
    vec_point_joint=torch.mul(est_heatmap[:,:,1:,:],xyz_diff)

    est_xyz_all = points.unsqueeze(1).expand(-1,est_heatmap.size(1),-1,-1) + vec_point_joint
    return weighted_mean_fusion(est_xyz_all, est_heatmap[:,:,0,:], candidate_num)

def weighted_mean_fusion(est_xyz_all, heatmap, candidate_num):
    ## select candidate points
    weights, nn_idx = torch.topk(heatmap, candidate_num, 2, largest=True, sorted=False) # weights: B * JOINT_NUM * candidate_num, nn_idx: B * JOINT_NUM * candidate_num
    nn_idx = nn_idx.unsqueeze(2).expand(-1, -1, 3, -1)  # B * JOINT_NUM * 3 * candidate_num
    est_xyz = est_xyz_all.gather(-1,nn_idx)             # B * JOINT_NUM * 3 * candidate_num
    
    # fusion method: weighted mean
    weights_norm = torch.sum(weights, -1, keepdim=True).expand(-1, -1, candidate_num)
    weights = torch.div(weights, weights_norm)          # B * JOINT_NUM * candidate_num
    weights = weights.unsqueeze(2).expand(-1, -1, 3, -1)# B * JOINT_NUM * 3 * candidate_num
    
    est_xyz = torch.mul(est_xyz, weights)               # B * JOINT_NUM * 3 * candidate_num
    est_xyz = torch.sum(est_xyz, -1)                    # B * JOINT_NUM * 3
    #est_xyz = torch.mean(est_xyz, -1)                    # B * JOINT_NUM * 3 (unweighted mean)
    return est_xyz
import numpy as np
import torch
import torch.nn as nn

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

##losses

class v_loss(nn.Module):
    def __init__(self,device='cuda:0'):
        super(v_loss,self).__init__()
        self.device=device

    def get_mean_coo(self,n_joints):
        batch_size=n_joints.shape[0]
        mean_coo=torch.zeros(batch_size-2,17,3).to(self.device)
        for i in range(batch_size-2):
            mean_coo[i]=(n_joints[i]+n_joints[i+2])/2
        return mean_coo

    def forward(self,input):
        n_joints=input.view(-1,17,3)
        get_mean_coo=self.get_mean_coo(n_joints)
        differ=nn.MSELoss()
        loss=differ(get_mean_coo,input[1:-1])
        return loss

class v_len_loss(nn.Module):
    def __init__(self,device='cuda:0'):
        super(v_len_loss,self).__init__()
        self.device=device

    def get_bone(self,n_joints):
        bone_p_index=torch.tensor([ 0, 0, 1, 2, 5, 5, 6, 7, 8, 5, 6,11,12,13,14]).to(self.device)
        bone_s_index=torch.tensor([ 1, 2, 3, 4, 6, 7, 8, 9,10,11,12,13,14,15,16]).to(self.device)

        bone_p_points=n_joints.index_select(1,bone_p_index)
        bone_s_points=n_joints.index_select(1,bone_s_index)
        bone_vecter=bone_p_points-bone_s_points
        bone_lenth=torch.norm(bone_vecter,2,2)
        return bone_lenth

    def forward(self,input):
        n_joints=input.view(-1,17,3)
        n_bone_lenth=self.get_bone(n_joints)
        bone_mean=torch.mean(n_bone_lenth,dim=1,keepdim=True).expand(n_joints.size(0),-1).to(self.device)
        l_n2m=n_bone_lenth/bone_mean
        differ=nn.MSELoss()
        loss=differ(l_n2m,torch.ones(l_n2m.size()).to(self.device))
        return loss

class symloss(nn.Module):
    def __init__(self,device='cuda:0'):
        super(symloss,self).__init__()
        self.device = device

    def get_left_bone(self,n_joints):
        left_p_index=torch.tensor([5,7,11,13]).to(self.device)
        left_s_index=torch.tensor([7,9,13,15]).to(self.device)    #left_bone:0-1,1-3,5-7,7-9,11-13,13-15,5,11,0-5

        left_p_points=n_joints.index_select(1,left_p_index)
        left_s_points=n_joints.index_select(1,left_s_index)
        left_bone_vecter=left_p_points-left_s_points
        left_bone_lenth=torch.norm(left_bone_vecter,2,2)
        return left_bone_lenth

    def get_right_bone(self,n_joints):
        right_p_index=torch.tensor([6,8,12,14]).to(self.device)
        right_s_index=torch.tensor([8,10,14,16]).to(self.device)    #right_bone:0-1,1-3,5-7,7-9,11-13,13-15,5,11,0-5

        right_p_points=n_joints.index_select(1,right_p_index)
        right_s_points=n_joints.index_select(1,right_s_index)
        right_bone_vecter=right_p_points-right_s_points
        right_bone_lenth=torch.norm(right_bone_vecter,2,2)
        return right_bone_lenth

    def forward(self,input):
        n_joints=input.view(-1,17,3)
        l_bone=self.get_left_bone(n_joints)       
        r_bone=self.get_right_bone(n_joints)   
        l2r=l_bone/r_bone
        r2l=r_bone/l_bone
        total=torch.cat((l2r,r2l),1)
        differ=nn.MSELoss()
        loss=differ(total,torch.ones(total.size()).to(self.device))
        return loss

class unprojectloss(nn.Module):
    def __init__(self,device='cuda:0'):
        super(unprojectloss,self).__init__()
        self.device=device

    def forward(self,batch_coords,gt_2d,M,intrinsic):
        batch_size=batch_coords.shape[0]
        joints_num=batch_coords.shape[1]
        pred_3d=torch.matmul(batch_coords,M[:3,:3].permute(1,0))
        pred_3d=pred_3d+M[:3,3]

        gt_2d_vic=torch.ones(pred_3d.shape).to(self.device)
        gt_2d_vic[:,:,0]=(gt_2d[:,:,0]-intrinsic[0,2])/intrinsic[0,0]
        gt_2d_vic[:,:,1]=(gt_2d[:,:,1]-intrinsic[1,2])/intrinsic[1,1]
        gt_2d_norm=torch.norm(gt_2d_vic,dim=2)
        chainplan=torch.zeros(gt_2d.shape).to(self.device)
        chainplan[:,:,0]=gt_2d_vic[:,:,0]*pred_3d[:,:,2]-pred_3d[:,:,0]
        chainplan[:,:,1]=gt_2d_vic[:,:,1]*pred_3d[:,:,2]-pred_3d[:,:,1]
        D_inplan=torch.norm(chainplan,dim=2)
        joints_len=torch.norm(pred_3d,dim=2)
        l_every=D_inplan/(gt_2d_norm*joints_len)
        loss=torch.mean(l_every)
        return loss

class lenth_loss(nn.Module):
    def __init__(self, use_weights=True, device='cuda:0'):
        super(lenth_loss,self).__init__()
        self.use_weights = use_weights
        self.device = device

    def get_bone(self,n_joints):
        bone_p_index=torch.tensor([5,7,11,13,6,8,12,14]).to(self.device)
        bone_s_index=torch.tensor([7,9,13,15,8,10,14,16]).to(self.device)

        bone_p_points=n_joints.index_select(1,bone_p_index)
        bone_s_points=n_joints.index_select(1,bone_s_index)
        bone_vecter=bone_p_points-bone_s_points
        bone_lenth=torch.norm(bone_vecter,2,2)
        return bone_lenth

    def forward(self,input,bone_lenth_gt):
        n_joints=input.view(-1,17,3)
        bone_lenth = self.get_bone(n_joints)
        #statistics_bone_lenth=torch.tensor([0.3051759644470673,0.3111776801758084,0.47747875673948087,0.46071092593822954,0.3013643357323193 ,0.30836866865507406,0.47840432260401033,0.46032841724681783], dtype=torch.float32)
        aim_bone_lenth=bone_lenth_gt[:8].unsqueeze(0).expand(bone_lenth.size(0),-1).to(self.device)
        differ=nn.MSELoss()
        loss=differ(bone_lenth,aim_bone_lenth)
        return loss

class imgandbone_lossnew(nn.Module):
    def __init__(self,device='cuda:0'):
        super(imgandbone_lossnew,self).__init__()
        self.len_loss = lenth_loss( device= device)
        self.unprojectloss = unprojectloss(device=device)
        self.symloss = symloss(device=device)
        self.v_loss=v_loss(device=device)
        self.v_len_loss=v_len_loss(device=device)
        self.device = device

    

    def forward(self,pred_3d_cs,gt_2d,intrinsic,M,bone_lenth):
        loss_ss=self.len_loss(pred_3d_cs,bone_lenth)
        loss_ws=self.unprojectloss(pred_3d_cs,gt_2d,M,intrinsic)
        loss_sym=self.symloss(pred_3d_cs)
        loss_v=self.v_loss(pred_3d_cs)
        loss_v_len=self.v_len_loss(pred_3d_cs)

        loss=loss_ss+10*loss_ws+0.002*loss_sym+0.1*loss_v+0.002*loss_v_len
        return loss
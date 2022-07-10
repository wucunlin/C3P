import os
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import load_cache


class CMUnewDataset(Dataset):
    def __init__(self, cfg, root, is_train):
        super(CMUnewDataset, self).__init__()
        self.root = root
        self.joints_change2coco=[1,15,17,16,18,3,9,4,10,5,11,6,12,7,13,8,14]
        self.kinnect_ind='KINECTNODE2'
        self.is_train=is_train
        if is_train==True:
            self.files=['171026_pose1', '171026_pose2',
                        '171204_pose1','171204_pose2','171204_pose3','171204_pose4','171204_pose5']
            self.pcb_files=self.get_pcbfilesname()
            self.sel_fea_files=self.get_seljointfilename()
            self.batch_size=6


        else:
            self.files=['171026_pose3','171204_pose6']
            self.pcb_files=self.get_pcbfilesname()
            self.sel_fea_files=self.get_seljointfilename()
            self.batch_size=8

    def get_pcbfilesname(self):
        list=[]
        for file in self.files:
            now_path=self.root+'/'+file+'/pcbs'+'/'+self.kinnect_ind
        
            pcb_files = os.listdir(now_path)
            pcb_files.sort()
            for pcb_file in pcb_files:
                now_pcb_path=now_path+'/'+pcb_file
                list.append(now_pcb_path)
        return list

    def get_seljointfilename(self):
        list=[]
        for file in self.files:
            now_path=self.root+'/'+file+'/features'+'/'+self.kinnect_ind
            feature_files = os.listdir(now_path)
            feature_files.sort()
            for feature_file in feature_files:
                now_feature_path=now_path+'/'+feature_file
                list.append(now_feature_path)
        return list


    def getdata(self, pcb_file,batch_size):
        dic_all=load_cache(pcb_file)
        pcb=np.array(dic_all['input_pcb'][:batch_size],dtype=np.float32) #B*2048*6
        obb_rotation_b=np.array(dic_all['rotation_matb'][:batch_size],dtype=np.float32)
        max_obb_len_b=np.array(dic_all['max_len_b'][:batch_size],dtype=np.float32)
        offset_b=np.array(dic_all['offset_b'][:batch_size],dtype=np.float32)
        gt_b=np.array(dic_all['joints_3d_block'][:batch_size],dtype=np.float32)
        gt_dep_2d=np.array(dic_all['joints_dep_2d_block'][:batch_size],dtype=np.float32)
        gt_rgb_2d=np.array(dic_all['joints_col_2d_block'][:batch_size],dtype=np.float32)
        idx1 = np.array(dic_all['idx1'][:batch_size],dtype=np.int16)
        idx2 = np.array(dic_all['idx2'][:batch_size],dtype=np.int16)
        id=dic_all['id'][0]
        #coco17=[]
        gt_b=gt_b[:,self.joints_change2coco,:]
        gt_dep_2d=gt_dep_2d[:,self.joints_change2coco,:2]
        gt_rgb_2d=gt_rgb_2d[:,self.joints_change2coco,:2]

        return pcb,obb_rotation_b,max_obb_len_b,offset_b,gt_b,gt_dep_2d,gt_rgb_2d,id,idx1,idx2

    def get_col_joints_data(self,joints_file,batch_size):
        dic_all=load_cache(joints_file)
        rgb_joints_sel=np.array(dic_all['rgb_pred_label'][:batch_size],dtype=np.float32) #B*17*2
        return rgb_joints_sel

    def get_rgb_came(self,file):
        file_path=self.root+'/'+file
        came_dic=load_cache(file_path+'/came_para.json')
        K_dep=np.array(came_dic[self.kinnect_ind]['color_para']['K_col'],dtype=np.float32)
        M_dep=np.array(came_dic[self.kinnect_ind]['color_para']['M_col'],dtype=np.float32)
        D_dep=np.array(came_dic[self.kinnect_ind]['color_para']['D_col'],dtype=np.float32)


        
        return K_dep,M_dep,D_dep

    def get_bone(self,file,id):
        file_path=self.root+'/'+file+'/mean_bones.json'
        dic=load_cache(file_path)
        bone_lenth=np.array(dic['{}'.format(id)],dtype=np.float32)
        return bone_lenth/100



    def __len__(self):
        return len(self.pcb_files)

    def __getitem__(self, index):
        now_file=self.pcb_files[index]
        now_name=now_file[-42:-30]
        #now_file_idx=int(now_file[-9:-5])
        pcb,obb_rotation_b,max_obb_len_b,offset_b,gt_b,gt_dep_2d,rgb_gt,id,idx1,idx2=self.getdata(now_file,self.batch_size)
        pcb=torch.from_numpy(pcb)
        rgb_joints_sel_file=self.sel_fea_files[index]
        rgb_joints_sel=self.get_col_joints_data(rgb_joints_sel_file,self.batch_size)
        #rgb_video_path, color_frame_block=self.get_rgb_video_and_index(rgb_joints_sel_file,self.batch_size)
        rgb_joints_sel=torch.from_numpy(rgb_joints_sel)

        K_col,M_col,D_col=self.get_rgb_came(now_name)
        bone_lenth=self.get_bone(now_name,id)
        #rgb_gt=self.get_rgb_gt(now_name,now_file_idx,self.batch_size)


        meta = {
            'gt': gt_b,
            'obb_rot_mat': obb_rotation_b,
            'obb_max_len': max_obb_len_b,
            'offset': offset_b,
            'rgb_pred':rgb_joints_sel,
            'rgb_gt':rgb_gt,
            'dep_gt_2d':gt_dep_2d,
            'K_col': K_col,
            'M_col':M_col,
            'D_col': D_col,
            'bone_lenth':bone_lenth,
            'idx1':idx1,
            'idx2':idx2
        }

        return pcb, meta
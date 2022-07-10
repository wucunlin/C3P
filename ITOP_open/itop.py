import os
import h5py
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import sys

class ITOPDataset(Dataset):
    def __init__(self, root, is_train):
        super(ITOPDataset, self).__init__()
        self.root = root
        if is_train==True:
            self.pcfilename='ITOP_side_train_washedobbpc.h5'
            self.allfilename='ITOP_side_train_all.h5'
            self.obb_rotations = self.geth5data(os.path.join(root,self.pcfilename),'rotations')
            self.obb_max_len = self.geth5data(os.path.join(root,self.pcfilename),'max_obb_3d_lens')
            self.obb_offset = self.geth5data(os.path.join(root,self.pcfilename),'offsets')
            self.obb_pc = self.geth5data(os.path.join(root,self.pcfilename),'pointcloud')
            self.id = self.geth5data(os.path.join(root,self.allfilename),'id')
            self.lable = self.geth5data(os.path.join(root,self.allfilename),'real_world_coordinates')
            self.img_label=self.geth5data(os.path.join(root,self.allfilename),'image_coordinates')
            self.bbox = self.geth5data(os.path.join(root,self.allfilename),'bbox')
        else :
            self.pcfilename='ITOP_side_testopen.h5'
            self.obb_rotations = self.geth5data(os.path.join(root,self.pcfilename),'rotations')
            self.obb_max_len = self.geth5data(os.path.join(root,self.pcfilename),'max_obb_3d_lens')
            self.obb_offset = self.geth5data(os.path.join(root,self.pcfilename),'offsets')
            self.obb_pc = self.geth5data(os.path.join(root,self.pcfilename),'pointcloud')
            self.lable = self.geth5data(os.path.join(root,self.pcfilename),'real_world_coordinates')
            self.idx1 = self.geth5data(os.path.join(root,self.pcfilename),'idx1')
            self.idx2 = self.geth5data(os.path.join(root,self.pcfilename),'idx2')

    def __len__(self):
        return self.obb_pc.shape[0]

    def __getitem__(self, index):

        input_pc = self.obb_pc[index]
        input_pc = input_pc.astype(np.float)
        input_pc = torch.from_numpy(input_pc)
        obb_rot = self.obb_rotations[index]
        obb_off = self.obb_offset[index]
        obb_len = self.obb_max_len[index]
        gt = self.lable[index]
        gt[:,1]=-gt[:,1]
        obb_gt=gt.dot(obb_rot)/obb_len -obb_off
        idx1 = self.idx1[index]
        idx2 = self.idx2[index]

        meta = {
            'gt': gt,
            'obb_gt':obb_gt,
            'obb_rot_mat': obb_rot,
            'obb_max_len': obb_len,
            'offset': obb_off,
            'idx1': idx1,
            'idx2': idx2
        }

        return input_pc, meta

    def geth5data(self,path,key):

        h5file=h5py.File(path,'r')
        data=h5file[key][:]

        h5file.close()

        return data
    


import sys
import os
import pprint
import time
import logging

import torch
from config import config
import pose_pointnet
import itop
from core import xyz_form_heatmap_ball_truelen, obb_coord2cs_coord, mpjpe, ap10cm
from utils import create_logger


def main():

    logger = create_logger(config)

    model_dep = pose_pointnet.get_pose_net(config)
    checkpoint = torch.load(os.path.join('model', 'checkpointbest.pth.tar'))
    gpus = [int(i) for i in config.GPUS.split(',')]
    model_dep.load_state_dict(checkpoint['state_dict'])
    model_dep = torch.nn.DataParallel(model_dep, device_ids=gpus).to('cuda:{}'.format(gpus[0]))

    test_dataset = itop.ITOPDataset(config.DATASET.ROOT,False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,#config.TRAIN.BATCH_SIZE , # len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    validate(config, test_loader, model_dep)

def validate(config, validate_loader, model_dep):
    logger = logging.getLogger(__name__)
    acc = AverageMeter()
    ap = AverageMeter()
    gpus = [int(i) for i in config.GPUS.split(',')]
    joints_num = config.MODEL.NUM_JOINTS
    model_dep.eval()
    with torch.no_grad():
        for i, (input_dep,meta) in enumerate(validate_loader):
            input_dep = input_dep.permute(0, 2, 1)
            input_dep = input_dep.type(torch.FloatTensor)
            input_dep = input_dep.to('cuda:{}'.format(gpus[0]))

            rotation = meta['obb_rot_mat'].view(-1, 3, 3).to('cuda:{}'.format(gpus[0]))
            offset = meta['offset'].view(-1, 3).to('cuda:{}'.format(gpus[0]))
            max_obb_len = meta['obb_max_len'].view(-1).to('cuda:{}'.format(gpus[0]))
            gt_3d=meta['gt'].view(-1,15,3).to('cuda:{}'.format(gpus[0]))

            idx1 = meta['idx1'].long().to('cuda:{}'.format(gpus[0]))
            idx2 = meta['idx2'].long().to('cuda:{}'.format(gpus[0]))

            out_d = model_dep(input_dep,idx1,idx2)

            pred_obb_xyz = xyz_form_heatmap_ball_truelen(out_d[-1],input_dep[:,:3,:],0.80,max_obb_len,25)
            pred_gt_xyz = obb_coord2cs_coord(pred_obb_xyz,rotation,offset,max_obb_len)

            acc.update(mpjpe(pred_gt_xyz,gt_3d).item())
            ap.update(ap10cm(pred_gt_xyz,gt_3d))

            if i%50==0:
                #print('Accuracy {acc.val:.3f} ({acc.avg:.3f})\t AP {ap.val:.3f}({ap.avg:.3f})'.format(acc=acc,ap=ap))
                msg = 'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t AP {ap.val:.3f}({ap.avg:.3f})'.format(acc=acc,ap=ap)

                logger.info(msg)


        

    msg = 'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t AP {ap.val:.3f}({ap.avg:.3f})'.format(acc=acc,ap=ap)
    logger.info(msg)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count #if self.count != 0 else 0

if __name__=='__main__':
    main()
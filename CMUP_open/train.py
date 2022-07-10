import os
import logging
import torch
from config import config
import pose_pointnet
import PT
import cmu
from core import xyz_form_heatmap_ball_truelen, obb_coord2cs_coord, mpjpe, ap10cm, imgandbone_lossnew
from utils import get_optimizer, save_checkpoint, create_logger

def main():

    logger = create_logger(config)

    model_dep = pose_pointnet.get_pose_net(config)
    gpus = [int(i) for i in config.GPUS.split(',')]
    model_dep = torch.nn.DataParallel(model_dep, device_ids=gpus).to('cuda:{}'.format(gpus[0]))

    criterion = imgandbone_lossnew(device='cuda:{}'.format(gpus[0])).to('cuda:{}'.format(gpus[0]))
    optimizer = get_optimizer(config, model_dep)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    train_dataset = cmu.CMUnewDataset(
        config,
        config.DATASET.ROOT,
        True
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    test_dataset = cmu.CMUnewDataset(
        config,
        config.DATASET.ROOT,
        False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    best_pref = 0.0

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        train(config, train_loader, model_dep, criterion, optimizer, epoch)
        lr_scheduler.step()

        if epoch%5==0:
            now_pref = validate(config, test_loader, model_dep)
            if now_pref>best_pref:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_dep.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, False, config.OUTPUT_DIR,'checkpoint_best.pth.tar')

def train(config, train_loader, model_dep, criterion, optimizer, epoch):
    logger = logging.getLogger(__name__)
    losses = AverageMeter()
    acc = AverageMeter()
    ap = AverageMeter()
    gpus = [int(i) for i in config.GPUS.split(',')]

    model_dep.train()
    for i, (input_dep, meta) in enumerate(train_loader):

        input_dep = input_dep.reshape(-1, config.MODEL.POINTNET.INPUT_NUM, 6).permute(0, 2, 1)
        input_dep = input_dep.type(torch.FloatTensor)
        input_dep = input_dep.to('cuda:{}'.format(gpus[0]))

        rotation = meta['obb_rot_mat'].view(-1, 3, 3).to('cuda:{}'.format(gpus[0]))
        offset = meta['offset'].view(-1, 3).to('cuda:{}'.format(gpus[0]))
        max_obb_len = meta['obb_max_len'].view(-1).to('cuda:{}'.format(gpus[0]))
        gt_3d=meta['gt'].view(-1,17,3).to('cuda:{}'.format(gpus[0]))
        idx1 = meta['idx1'].long().to('cuda:{}'.format(gpus[0]))
        idx2 = meta['idx2'].long().to('cuda:{}'.format(gpus[0]))

        out_d = model_dep(input_dep,idx1,idx2)

        rgb_pred=meta['rgb_pred'].view(-1,17,2).to('cuda:{}'.format(gpus[0]))
        K_dep=meta['K_col'].view(3,3).to('cuda:{}'.format(gpus[0]))
        M_dep=meta['M_col'].view(4,4).to('cuda:{}'.format(gpus[0]))
        bone_lenth=meta['bone_lenth'].view(-1).to('cuda:{}'.format(gpus[0]))
        optimizer.zero_grad()
        loss =0.0

        for out_every in out_d:
            
            pred_obb_xyz = xyz_form_heatmap_ball_truelen(out_every,input_dep[:,:3,:],0.80,max_obb_len,64)
            pred_gt_xyz = obb_coord2cs_coord(pred_obb_xyz,rotation,offset,max_obb_len)
            loss_all=criterion(pred_gt_xyz,rgb_pred,K_dep,M_dep,bone_lenth)
            loss=loss+loss_all

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), input_dep.size(0))

        pred_obb_xyz = xyz_form_heatmap_ball_truelen(out_d[-1],input_dep[:,:3,:],0.80,max_obb_len,64)
        pred_gt_xyz = obb_coord2cs_coord(pred_obb_xyz,rotation,offset,max_obb_len)
        acc.update(mpjpe(pred_gt_xyz,gt_3d).item())
        ap.update(ap10cm(pred_gt_xyz,gt_3d))

        if i%20==0:

            msg = 'Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                ' Accuracy {acc.val:.3f} ({acc.avg:.3f})\t AP {ap.val:.3f}({ap.avg:.3f})'.format(
                    epoch, i, len(train_loader),loss=losses, acc=acc,ap=ap)
            logger.info(msg)

    msg = 'Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                ' Accuracy {acc.val:.3f} ({acc.avg:.3f})\t AP {ap.val:.3f}({ap.avg:.3f})'.format(
                    epoch, i, len(train_loader),loss=losses, acc=acc,ap=ap)
    logger.info(msg)
        


def validate(config, validate_loader, model_dep):
    logger = logging.getLogger(__name__)
    acc = AverageMeter()
    ap = AverageMeter()
    gpus = [int(i) for i in config.GPUS.split(',')]
    model_dep.eval()
    with torch.no_grad():
        for i, (input_dep,meta) in enumerate(validate_loader):
            input_dep = input_dep.reshape(-1, config.MODEL.POINTNET.INPUT_NUM, 6).permute(0, 2, 1)
            input_dep = input_dep.type(torch.FloatTensor)
            input_dep = input_dep.to('cuda:{}'.format(gpus[0]))
            rotation = meta['obb_rot_mat'].view(-1, 3, 3).to('cuda:{}'.format(gpus[0]))
            offset = meta['offset'].view(-1, 3).to('cuda:{}'.format(gpus[0]))
            max_obb_len = meta['obb_max_len'].view(-1).to('cuda:{}'.format(gpus[0]))
            gt_3d=meta['gt'].view(-1,17,3).to('cuda:{}'.format(gpus[0]))

            idx1 = meta['idx1'].long().to('cuda:{}'.format(gpus[0]))
            idx2 = meta['idx2'].long().to('cuda:{}'.format(gpus[0]))

            out_d = model_dep(input_dep,idx1,idx2)

            pred_obb_xyz = xyz_form_heatmap_ball_truelen(out_d[-1],input_dep[:,:3,:],0.80,max_obb_len,64)
            pred_gt_xyz = obb_coord2cs_coord(pred_obb_xyz,rotation,offset,max_obb_len)

            acc.update(mpjpe(pred_gt_xyz,gt_3d).item())
            ap.update(ap10cm(pred_gt_xyz,gt_3d))

            if i%20==0:
                msg = 'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t AP {ap.val:.3f}({ap.avg:.3f})'.format(acc=acc,ap=ap)
                logger.info(msg)

    msg = 'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t AP {ap.val:.3f}({ap.avg:.3f})'.format(acc=acc,ap=ap)
    logger.info(msg)
    
    return acc.avg







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
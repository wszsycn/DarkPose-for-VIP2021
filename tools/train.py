# ------------------------------------------------------------------------------
# WANG Shaozhi PolyU shao-zhi.wang@connect.polyu.hk
# Adjusted from https://github.com/ilovepose/DarkPose
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
#from tensorboardX import SummaryWriter

import _init_paths

# Depending on the working environment, you could delete the "lib." accordingly.

from lib.config import cfg
from lib.config import update_config
from lib.core.loss import JointsMSELoss
from lib.core.evaluate import accuracy
from lib.core.function import train
from lib.core.function import validate
from lib.core.function import AverageMeter
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.utils.utils import get_model_summary
import dataset
import models
from lib.dataset.vip_dataset import vip
from lib.models.pose_hrnet import get_pose_net
from lib.models.pose_hrnet import PoseHighResolutionNet
import yaml
import time
#from torch.optim.lr_scheduler import CosineAnnealingLR

def model_parameters(_structure, _parameterDir):
    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint
    model_state_dict = _structure.state_dict()
    # 1. filter out unnecessary keys
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
    pretrained_state_dict.pop('final_layer.weight', None)
    pretrained_state_dict.pop('final_layer.bias', None)

    # 2. overwrite entries in the existing state dict
    model_state_dict.update(pretrained_state_dict)
    _structure.load_state_dict(model_state_dict)


def main():

    # initialize dataloader
    # These two paths are the given database
    VIP_Train = vip(root="/home/vip2021/NewDarkDataset/SLP_VIPCup_database", mode="train")
    VIP_Valid = vip(root="/home/vip2021/NewDarkDataset/SLP_VIPCup_database", mode="valid")
    train_loader = torch.utils.data.DataLoader(
        VIP_Train,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        #pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        VIP_Valid,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        #pin_memory=cfg.PIN_MEMORY
    )



    # Network
    # update_config(cfg, args)

    with open('/home/vip2021/NewDark/experiments/mpii/hrnet/w48_256x192_adam_lr1e-3.yaml', 'r') as file:
        cfg = yaml.full_load(file)

    # load the pretrained model
    model = get_pose_net(cfg, is_train = False).cuda()
    model_parameters(model, "/home/vip2021/NewDark/model/w48_256x192.pth")
    # #model = torch.nn.DataParallel(model, [0, 1])
    model.cuda()
    #Note that because I only have one GPU thus I could not use the torch.nn.DataParallel method.
    # Users could adjust the code accordingly.



    # loss function

    criterion = JointsMSELoss(
        use_target_weight=True
    ).cuda()

    # Optimizer
    # Here we use Adam

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay=5e-4)


    # Some parameters that could be used soon
    best_perf = 0.0
    best_model = False
    last_epoch = -1
    final_output_dir = '/home/vip2021/NewDark/output/vip_dataset/pose_hrnet'
    #begin_epoch = cfg['TRAIN']['BEGIN_EPOCH'] # 0
    # checkpoint_file = os.path.join(
    #     final_output_dir, 'checkpoint.pth'
    # )
    # # copy model file
    # this_dir = os.path.dirname(__file__)

    #lr_scheduler = CosineAnnealingLR(optimizer, T_max=iter, eta_min=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg['TRAIN']['LR_STEP'], cfg['TRAIN']['LR_FACTOR'],
        last_epoch=last_epoch
    )
    tbatch_time = AverageMeter()
    tdata_time = AverageMeter()
    tlosses = AverageMeter()
    tacc = AverageMeter()
    vbatch_time = AverageMeter()
    vlosses = AverageMeter()
    vacc = AverageMeter()

    for epoch in range(0, 500): # we run for 500 epoch

        tend = time.time()
        # begin to train
        best_perf = 0

        model.train()
        #train process
        for i, (imgs, targets, targets_weights) in enumerate(train_loader):
            # measure data loading time
            tdata_time.update(time.time() - tend)
            # compute output
            imgs = imgs.cuda()
            targets = targets.cuda()
            targets_weights = targets_weights.cuda()
            print(np.shape(imgs))
            outputs = model(imgs)
            result_loss = criterion(outputs, targets, targets_weights)
            optimizer.zero_grad()
            result_loss.backward()
            optimizer.step()
            #print(imgs.size())
            print(i)
            print(result_loss)

            tlosses.update(result_loss.item(), imgs.size(0))

            _, t_avg_acc, cnt, pred = accuracy(outputs.detach().cpu().numpy(),
                                             targets.detach().cpu().numpy())
            tacc.update(t_avg_acc, cnt)

            # measure elapsed time
            tbatch_time.update(time.time() - tend)
            tend = time.time()
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=tbatch_time,
                      speed=imgs.size(0)/tbatch_time.val,
                      data_time=tdata_time, loss=tlosses, acc=tacc)
                #logger.info(msg)
            print(msg)
            # if i == 80:
            #     break

                #writer = writer_dict['writer']
                #global_steps = writer_dict['train_global_steps']
                #writer.add_scalar('train_loss', losses.val, global_steps)
                #writer.add_scalar('train_acc', acc.val, global_steps)
                #writer_dict['train_global_steps'] = global_steps + 1

                #prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
                #save_debug_images(config, input, meta, target, pred*4, output,
                #                  prefix)

         # End of train

        lr_scheduler.step()

        # Begin to validate
        # validate process
        model.eval()
        for i, (imgs, targets, targets_weights) in enumerate(valid_loader):
            # measure data loading time
            #data_time.update(time.time() - end)
            # compute output
            imgs = imgs.cuda()
            targets = targets.cuda()
            targets_weights = targets_weights.cuda()
            # outputs = model(imgs)
            # valid_loss = criterion(outputs, targets, targets_weights)
            with torch.no_grad():
                outputs = model(imgs)
                valid_loss = criterion(outputs, targets, targets_weights)
            print(i)
            print(valid_loss)
            # print('=> saving checkpoint to {}'.format(final_output_dir))

            vlosses.update(valid_loss.item(), imgs.size(0))

            _, v_avg_acc, cnt, pred = accuracy(outputs.detach().cpu().numpy(),
                                             targets.detach().cpu().numpy())
            vacc.update(v_avg_acc, cnt)


            msg = 'Test: [{0}/{1}]\t' \
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                i, len(valid_loader), batch_time=vbatch_time,
                loss=vlosses, acc=vacc)
            # logger.info(msg)
            print(msg)



        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     #'model': cfg.MODEL.NAME,
        #     'state_dict': model.state_dict(),
        #     #'best_state_dict': model.module.state_dict(),
        #     #'perf': perf_indicator,
        #     'optimizer': optimizer.state_dict(),
        # }, best_model, final_output_dir)


    final_model_state_file = os.path.join(
                 final_output_dir, 'final_state.pth'
            )
    torch.save(model.state_dict(), final_model_state_file)
    print("the pth file has been saved to: {}".format(final_model_state_file))


if __name__ == '__main__':
    main()

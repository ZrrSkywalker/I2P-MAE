import torch
import torch.nn as nn
import numpy as np
import os
import json
import math
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from sklearn.svm import SVC
from torchvision import transforms
from datasets import data_transforms

from .rotate import rotate_point_clouds, rotate_point_clouds_batch, rotate_theta_phi


train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)

    # build dataset for pre-training
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    
    # build dataset for linear SVM
    train_dataloader_svm, test_dataloader_svm = builder.dataset_builder_svm(config.dataset.svm)
    
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # training
    base_model.zero_grad()
    best_accuracy = 0.0
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['losses'])
        losses_2d = AverageMeter(['Loss_2D'])
        losses_3d = AverageMeter(['Loss_3D'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)   
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints  

            # random rotate the point cloud with a random init angle
            angle = torch.stack([torch.rand(points.size(0)) * 1.9 + 0.04,                       # 0.04 ~ 1.94pi
                                (torch.rand(points.size(0)) * 0.2 - 0.4)], dim=-1) * math.pi    # -0.4 ~ -0.2 pi
            rotation_matrix = rotate_theta_phi(angle)
            input_pc = rotate_point_clouds_batch(points, rotation_matrix, use_normals=False).contiguous()   

            points = train_transforms(points)
            loss_3d, loss_2d = base_model(points)

            loss = loss_3d + loss_2d

            try:
                loss.backward()
            except:
                loss = loss.mean()
                loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()*1000])
                losses_2d.update([loss_2d.item()*1000])
                losses_3d.update([loss_3d.item()*1000])
            else:
                losses.update([loss.item()*1000])
                losses_3d.update([loss_3d.item()*1000])
                losses_2d.update([loss_2d.item()*1000])


            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_2d', loss_2d.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_3d', loss_3d.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses_2d = %s Losses_3d = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses_2d.val()], ['%.4f' % l for l in losses_3d.val()], optimizer.param_groups[0]['lr']), logger = logger)
            
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_2d', losses_2d.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_3d', losses_3d.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses_2d = %s Losses_3d = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses_2d.avg()], ['%.4f' % l for l in losses_3d.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)

        ### eval with SVM ###
        feats_train = []
        labels_train = []
        base_model.eval()

        for i, (data, label) in enumerate(train_dataloader_svm):
            labels = list(map(lambda x: x[0],label.numpy().tolist()))
            data = data.cuda().contiguous()
            with torch.no_grad():
                feats = base_model(data, eval=True)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels

        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        feats_test = []
        labels_test = []

        for i, (data, label) in enumerate(test_dataloader_svm):
            labels = list(map(lambda x: x[0],label.numpy().tolist()))
            data = data.cuda().contiguous()
            with torch.no_grad():
                feats = base_model(data, eval=True)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels

        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)
        
        model_tl = SVC(C = 0.01, kernel ='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy = model_tl.score(feats_test, labels_test)
        
        print_log(f"Linear Accuracy : {test_accuracy}", logger=logger)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print_log(f"Saving best...", logger=logger)
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 25 ==0 and epoch >=250:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)
    
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

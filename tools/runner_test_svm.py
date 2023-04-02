import torch
import torch.nn as nn
import numpy as np
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from sklearn.svm import SVC



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


def run_net_svm_modelnet40(args, config):
    logger = get_logger(args.log_name)
    
    # build for linear SVM
    train_dataloader_svm, test_dataloader_svm = builder.dataset_builder_svm(config.dataset)

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # load ckpts
    builder.load_model(base_model, args.ckpts, logger = logger)
    
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
        
    model_tl = SVC(C = 0.0093, kernel ='linear')
    model_tl.fit(feats_train, labels_train)
    test_accuracy = model_tl.score(feats_test, labels_test)
    print_log(f"Linear Accuracy : {test_accuracy}", logger=logger)


def run_net_svm_scan(args, config):
    logger = get_logger(args.log_name)
    
    # build for linear SVM
    train_dataloader_svm, test_dataloader_svm = builder.dataset_builder_svm(config.dataset)

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # load ckpts
    builder.load_model(base_model, args.ckpts, logger = logger)
    
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
    
    ### eval with SVM ###
    feats_train = []
    labels_train = []
    base_model.eval()

    for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader_svm):
        points = data[0]
        label = data[1]

        labels = list(map(lambda x: x, label.numpy().tolist()))
        # .detach().cpu().numpy().tolist()
        data = points.cuda().contiguous()
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

    for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader_svm):
        points = data[0]
        label = data[1]

        labels = list(map(lambda x: x, label.numpy().tolist()))
        data = points.cuda().contiguous()
        with torch.no_grad():
            feats = base_model(data, eval=True)
        feats = feats.detach().cpu().numpy()
        for feat in feats:
            feats_test.append(feat)
        labels_test += labels

    feats_test = np.array(feats_test)
    labels_test = np.array(labels_test)
        
    model_tl = SVC(C = 0.018, kernel ='linear')
    model_tl.fit(feats_train, labels_train)
    test_accuracy = model_tl.score(feats_test, labels_test)
    print_log(f"Linear Accuracy : {test_accuracy}", logger=logger)

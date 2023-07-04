import os
import cv2
import sys
import time
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from Loss import *
from Utils import *

def Train(args, model, train_source_dataloader, dist_source_train, init_source_train,\
    train_target_dataloader, dist_target_train, init_target_train, \
    train_target_data_set, optimizer, epoch, writer, Best_Metrics):
    """Train."""
    
    model.train()

    #! ----------------------------- 1. Prepairation for training -----------------------------------
    #@  Define the preformance metric
    acc_source, prec_source, recall_source = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    ce_loss, ce_loss_target, ce_loss_source, loss, data_time, batch_time = [AverageMeter() for i in range(7)],\
         [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], AverageMeter(), AverageMeter(), AverageMeter()
    six_probilities_source, six_confidences_source = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]

    #@ Different learning rate iteration strategies are set for different backbone networks
    if args.Backbone in ['ResNet18', 'ResNet50']:
        if epoch <= 10:
            args.lr = 1e-4
        elif epoch <= 20:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    elif args.Backbone == 'MobileNet':
        if epoch <= 20:
            args.lr = 1e-3
        elif epoch <= 40:
            args.lr = 1e-4
        elif epoch <= 60:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    elif args.Backbone == 'VGGNet':
        if epoch <= 30:
            args.lr = 1e-3
        elif epoch <= 60:
            args.lr = 1e-4
        elif epoch <= 70:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    #@ Get Source/Target Dataloader iterator
    iter_source_dataloader = iter(train_source_dataloader)
    iter_target_dataloader = iter(train_target_dataloader)

    #@ Initializes a pseudo label for all target domain data of seven classifiers
    selected_labels = [torch.ones((len(train_target_data_set),), dtype=torch.long, ) * -1 for i in range(7)]
    selected_labels = [selected_labels[i].cuda() for i in range(7)]

    #@ Define the learning effect of seven categories of seven classifiers 
    classwise_accs = [torch.zeros((args.class_num,)).cuda() for i in range(7)]
    #! -------------------------------------------------------------------------------------------------

    #! ----------------------------- 2. Start training and calculating  -------------------------------
    # num_iter = len(train_source_dataloader) if (len(train_source_dataloader) < len(train_target_dataloader)) else len(train_target_dataloader)
    num_iter = len(train_source_dataloader)
    end = time.time()
    train_bar = tqdm(range(num_iter))
    for step, batch_index in enumerate(train_bar):

        #@ Obtain the input from source and target domain
        try:
            img_index_source, data_source, landmark_source, label_source = next(iter_source_dataloader)
        except:
            iter_source_dataloader = iter(train_source_dataloader)
            img_index_source, data_source, landmark_source, label_source = next(iter_source_dataloader)
        try:
            img_index_target, data_target, data_target_s, landmark_target, label_target = next(iter_target_dataloader)
        except:
            iter_target_dataloader = iter(train_target_dataloader)
            img_index_target, data_target, data_target_s, landmark_target, label_target = next(iter_target_dataloader)
        data_time.update(time.time() - end)

        img_index_source, data_source, landmark_source, label_source = \
            img_index_source.cuda(), data_source.cuda(), landmark_source.cuda(), label_source.cuda()
        img_index_target, data_target, data_target_s, landmark_target, label_target = \
            img_index_target.cuda(), data_target.cuda(), data_target_s.cuda(), landmark_target.cuda(), label_target.cuda()

        #@ Forward propagation
        end = time.time()
        features, preds = model(data_source, landmark_source)
        batch_time.update(time.time() - end)
        loss_ = 0

        #@ Loss1: Compute Cross Entropy Loss(Source Domain)
        classifiers_loss_ratio = [5, 1, 1, 1, 1, 1, 5]
        #@@ common cross entropy
        criteria = nn.CrossEntropyLoss()
        #@@ weight softmax
        # weight = torch.FloatTensor([1 / p for p in dist_source_train]).cuda()
        # criteria = nn.CrossEntropyLoss(weight=weight)
        #@@ focal loss
        # alpha = [1, 1, 1, 1, 1, 1, 1]
        # criteria = FocalLoss(alpha=0.5, gamma=1, num_classes=7)
        #@@ LADM Loss
        # criteria = LDAMLoss(cls_num_list=dist_source_train, max_m=0.5, s=30, weight=None)
        for i in range(7):
            tmp = criteria(preds[i], label_source)
            ce_loss[i].update(float(tmp.cpu().data.item()))
            ce_loss_source[i].update(float(tmp.cpu().data.item()))
            loss_ += classifiers_loss_ratio[i] * tmp

        # #@ Loss2: Compute Cross Entropy Loss(Target Domain) by Pseudo Labels
        # ratio = 1.0
        # classifiers_loss_ratio = [1, 1, 1, 1, 1, 1, 1]
        # feat_w, logits_w, feat_s, logits_s =  model(data_target, landmark_target, data_target_s, use_flexmatch=True)
        # pseudo_counters = [Counter(selected_labels[i].tolist()) for i in range(7)] # is a dict of list, each dict = {0: number of class 0, 1: number of class1...}
        # for i in range(7): # Traverse the seven classifer
        #     #@@ Calculate the learning effect of each category under each classifier
        #     if max(pseudo_counters[i].values()) < len(train_target_data_set):  # not all(5w) -1
        #         for class_index in range(args.class_num):
        #             classwise_accs[i][class_index] = pseudo_counters[i][class_index] / max(pseudo_counters[i].values())
        #     #@@ Calculate the loss``
        #     tmp, _, _, select, pseudo_lb, _ = ConsistencyLoss(logits_s[i], logits_w[i], classwise_accs[i])
        #     ce_loss[i].update(float(tmp.cpu().data.item()))
        #     ce_loss_target[i].update(float(tmp.cpu().data.item()))
        #     loss_ += ratio * classifiers_loss_ratio[i] * tmp
        #     #@@ update the pseudo labels set of target domain
        #     if img_index_target[select == 1].nelement() != 0:
        #         selected_labels[i][img_index_target[select == 1]] = pseudo_lb[select == 1]
        
        loss.update(float(loss_.cpu().data.item()))

        #@ Back Propagation
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()

        #@ Decay Learn Rate
        optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr, weight_decay=args.weight_decay) # optimizer = lr_scheduler(optimizer, num_iter*(epoch-1)+step, 0.001, 0.75, lr=args.lr, weight_decay=args.weight_decay)
        
        #@ 🔺 Compute accuracy, recall and loss
        for classifier_id in range(7):
            Compute_Accuracy(args, preds[classifier_id].narrow(0, 0, data_source.size(0)),\
                 label_source, acc_source[classifier_id], prec_source[classifier_id], recall_source[classifier_id])
        
        #@ Compute the proportion and confidence of pseudo labels produced by different number of classifiers predicting the same time are calculated
        Count_Probility_Accuracy(six_probilities_source, six_confidences_source,\
             [pred.narrow(0, 0, data_source.size(0)) for pred in preds], label_source)

        #@ Visualize the input imgs
        if step % 200 == -1:
            Visualize_Input_Image(args, writer, 'train', 'source', data_source, data_raw_path=init_source_train['imgs_list'])
            Visualize_Input_Image(args, writer, 'train', 'target', data_target, data_s_trans=data_target_s, data_raw_path=init_target_train['imgs_list'],)
                
        #@ Ending
        end = time.time()
        # train_bar.desc = f'[Train (Source Domain) Epoch {epoch}/{args.epochs}] cls loss: {ce_loss_source[0].avg:.3f}, {ce_loss_source[1].avg:.3f}, {ce_loss_source[2].avg:.3f}, {ce_loss_source[3].avg:.3f}, {ce_loss_source[4].avg:.3f}, {ce_loss_source[5].avg:.3f}, {ce_loss_source[6].avg:.3f}'
        train_bar.desc = f'[Train (Source Domain) Epoch {epoch}/{args.epochs}]({ce_loss_source[0].avg:.3f},{ce_loss_target[0].avg:.3f})'\
            f'({ce_loss_source[1].avg:.3f},{ce_loss_target[1].avg:.3f})({ce_loss_source[2].avg:.3f},{ce_loss_target[2].avg:.3f})'\
            f'({ce_loss_source[3].avg:.3f},{ce_loss_target[3].avg:.3f})({ce_loss_source[4].avg:.3f},{ce_loss_target[4].avg:.3f})'\
            f'({ce_loss_source[5].avg:.3f},{ce_loss_target[5].avg:.3f})({ce_loss_source[6].avg:.3f},{ce_loss_target[6].avg:.3f})'
    
    #@ 🔺 Compute the seven classifiers' overral accuracy, precision, recall and F1-score
    accs_source = Show_OnlyAccuracy(acc_source)
    precs_source, recalls_source, f1s_source = Show_OtherMetrics(prec_source, recall_source, [i for i in range(7)])
    #! --------------------------------------------------------------------------------------------------
    
    #! ----------------------------- 3. Record the results  ---------------------------------------------
    #@ Visualize the prediction effect of each classifier for each category(precision, recall and F1-score)
    fig = Draw_Category_Metrics_Bar(prec_source, recall_source)
    writer.add_figure('Category_Metrics/Train_Source', fig, global_step=epoch, close=True)

    #@ Visualize the prediction performance curve of each classifier
    Visualize_Transfer_Common_Result(epoch, writer, accs_source, precs_source, recalls_source, f1s_source, ce_loss_source,\
    six_confidences_source, six_probilities_source, state='Train', domain='Source')

    #@ Update the best metrics
    modified_metrics = Update_Best_Metrics(Best_Metrics, accs_source, precs_source, recalls_source, f1s_source, epoch, state='Train', domain='Source')

    #@ Write the log information
    EpochInfo, HistoryInfo = Write_Results_Log(Best_Metrics, accs_source, [], precs_source, [], \
        recalls_source, [], f1s_source, [], epoch, args, mode='Train')
    if modified_metrics > 0:
        LoggerInfo = EpochInfo + HistoryInfo
    else:
        LoggerInfo = EpochInfo
    with open(args.OutputPath + "/preTransfer_train_result.md", "a") as f:
        f.writelines(LoggerInfo)
    #! -------------------------------------------------------------------------------------------------

    return classwise_accs


def Test(args, model, test_source_dataloader, test_target_dataloader, dist_source_test, \
    dist_target_test, init_source_test, init_target_test, Best_Metrics, epoch, confidence_target, writer):
    """Test."""
    model.eval()

    #! ----------------------------- 1. Test On Target Domain  -----------------------------------------
    #@  Define the preformance metric
    acc_target, prec_target, recall_target = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    ce_loss, loss, data_time, batch_time = [AverageMeter() for i in range(7)], AverageMeter(), AverageMeter(), AverageMeter()
    six_probilities, six_confidences = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]

    end = time.time()
    test_target_bar = tqdm(test_target_dataloader)
    for batch_index, (_, input, landmark, target) in enumerate(test_target_bar):
        data_time.update(time.time() - end)

        input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()

        #@@ Forward Propagation
        with torch.no_grad():
            end = time.time()
            features, preds = model(input, landmark)
            batch_time.update(time.time() - end)

        #@@ Calculate the cross entropy of source data
        loss_ = 0
        #@@ weighted softmax
        # weight = torch.FloatTensor([1 / p for p in dist_target_test]).cuda()
        # criteria = nn.CrossEntropyLoss(weight=weight)
        #@@ focal loss
        # alpha = [1, 1, 1, 1, 1, 1, 1]
        # criteria = FocalLoss(alpha=0.5, gamma=1, num_classes=7)
        #@@ Cross Entropy
        criteria = nn.CrossEntropyLoss()
        for i in range(7):
            tmp = criteria(preds[i], target)
            ce_loss[i].update(tmp.cpu().data.item())
            loss_ += tmp
        loss.update(float(loss_.cpu().data.numpy()))

        #@@ 🔺 Compute the seven classifiers' accuracy, precision, recall and F1-score
        for classifier_id in range(7):
            Compute_Accuracy(args, preds[classifier_id], target, acc_target[classifier_id], prec_target[classifier_id], recall_target[classifier_id])

        #@@ Compute the proportion and confidence of pseudo labels produced by different number of classifiers predicting the same time are calculated
        Count_Probility_Accuracy(six_probilities, six_confidences, preds, target)

        #@ Visualize the input imgs
        if batch_index % 200 == -1:
            Visualize_Input_Image(args, writer, 'test', 'source', input, data_raw_path=init_source_test['imgs_list'])

        #@@ Ending
        end = time.time()
        test_target_bar.desc = f'[Test (Target Domain) Epoch {epoch}/{args.epochs}] ce loss: {ce_loss[0].avg:.3f}, {ce_loss[1].avg:.3f}, {ce_loss[2].avg:.3f}, {ce_loss[3].avg:.3f}, {ce_loss[4].avg:.3f}, {ce_loss[5].avg:.3f}, {ce_loss[6].avg:.3f}'
    
    #@ 🔺 Compute the seven classifiers' overal accuracy, precision, recall and F1-score
    accs_target = Show_OnlyAccuracy(acc_target)
    precs_target, recalls_target, f1s_target = Show_OtherMetrics(prec_target, recall_target, [i for i in range(7)])

    #@ Record the result
    #@@ Visualize the prediction effect of each classifier for each category
    fig = Draw_Category_Metrics_Bar(prec_target, recall_target)
    writer.add_figure('Category_Metrics/Test_Target', fig, global_step=epoch, close=True)

    #@@ Visualize the prediction performance curve of each classifier
    Visualize_Transfer_Common_Result(epoch, writer, accs_target, precs_target, recalls_target, f1s_target, ce_loss,\
         six_confidences, six_probilities, state='Test', domain='Target')

    #@@ Update the best metrics
    modified_metrics_target = Update_Best_Metrics(Best_Metrics,  accs_target, precs_target, recalls_target, f1s_target, epoch, state='Test', domain='Target')
    #! -------------------------------------------------------------------------------------------------

    #! ----------------------------- 2. Test On Source Domain  -----------------------------------------
    #@  Define the preformance metric
    acc_source, prec_source, recall_source = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    ce_loss, loss, data_time, batch_time = [AverageMeter() for i in range(7)], AverageMeter(), AverageMeter(), AverageMeter()
    six_probilities, six_confidences = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]

    #@ Start trainning and calculating
    end = time.time()
    test_source_bar = tqdm(test_source_dataloader)
    for batch_index, (_, input, landmark, target) in enumerate(test_source_bar):
        input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()
        data_time.update(time.time() - end)

        #@@ Forward Propagation
        with torch.no_grad():
            end = time.time()
            features, preds = model(input, landmark)
            batch_time.update(time.time() - end)

        #@@ Calculate the cross entropy of source data
        loss_ = 0
        #@@ weighted softmax
        # weight = torch.FloatTensor([1 / p for p in dist_source_test]).cuda()
        # criteria = nn.CrossEntropyLoss(weight=weight)
        #@@ focal loss
        # alpha = [1, 1, 1, 1, 1, 1, 1]
        # criteria = FocalLoss(alpha=0.5, gamma=1, num_classes=7)
        #@@ Cross Entropy
        criteria = nn.CrossEntropyLoss()
        for i in range(7):
            tmp = criteria(preds[i], target)
            ce_loss[i].update(tmp.cpu().data.item())
            loss_ += tmp
        loss.update(float(loss_.cpu().data.numpy()))

        #@@ 🔺 Compute the seven classifiers' accuracy, precision, recall and F1-score
        for classifier_id in range(7):
            Compute_Accuracy(args, preds[classifier_id], target, acc_source[classifier_id], prec_source[classifier_id], recall_source[classifier_id])
        
        #@@ Compute the proportion and confidence of pseudo labels produced by different number of classifiers predicting the same time are calculated
        Count_Probility_Accuracy(six_probilities, six_confidences, preds, target)

        #@ Visualize the input imgs
        if batch_index % 200 == -1:
            Visualize_Input_Image(args, writer, 'test', 'target', input, data_raw_path=init_target_test['imgs_list'])

        #@@ Ending
        end = time.time()
        test_source_bar.desc = f'[Test (Source Domain) Epoch {epoch}/{args.epochs}] ce loss: {ce_loss[0].avg:.3f}, {ce_loss[1].avg:.3f}, {ce_loss[2].avg:.3f}, {ce_loss[3].avg:.3f}, {ce_loss[4].avg:.3f}, {ce_loss[5].avg:.3f}, {ce_loss[6].avg:.3f}'

    #@ 🔺 Compute the seven classifiers' accuracy, precision, recall and F1-scores
    accs_source = Show_OnlyAccuracy(acc_source)
    precs_source, recalls_source, f1s_source = Show_OtherMetrics(prec_source, recall_source, [i for i in range(7)]) # 第三个参数就是随便凑个长度为7的list而已
    
    #@ Record the result
    #@@ Visualize the prediction effect of each classifier for each category
    fig = Draw_Category_Metrics_Bar(prec_source, recall_source)
    writer.add_figure('Category_Metrics/Test_Source', fig, global_step=epoch, close=True)

    #@@ Visualize the prediction performance curve of each classifier
    Visualize_Transfer_Common_Result(epoch, writer, accs_source, precs_source, recalls_source, f1s_source, ce_loss,\
         six_confidences, six_probilities, state='Test', domain='Source')

    #@ Save Checkpoints
    metric = args.judge_criteria
    dictionary = {'acc': accs_source, 'prec': precs_source, 'recall': recalls_source, 'f1': f1s_source}
    best_name, best_value = Get_Best_Name_Value(dictionary[metric], mode='decimal')
    alone = {'name': best_name, 'value': best_value}

    #@ Update the best metrics
    modified_metrics_source = Update_Best_Metrics(Best_Metrics, accs_source, precs_source,\
         recalls_source, f1s_source, epoch, state='Test', domain='Source')
    #! -------------------------------------------------------------------------------------------------

    #! ----------------------------- 3. write the log file  --------------------------------------------
    EpochInfo, HistoryInfo = Write_Results_Log(Best_Metrics, accs_source, accs_target, precs_source, precs_target, \
    recalls_source, recalls_target, f1s_source, f1s_target, epoch, args, mode='Test')
    if modified_metrics_source + modified_metrics_target > 0:
        LoggerInfo = EpochInfo + HistoryInfo
    else:
        LoggerInfo = EpochInfo
    with open(args.OutputPath + "/preTransfer_test_result.md","a") as f:
        f.writelines(LoggerInfo)
    #! -------------------------------------------------------------------------------------------------

    return alone

def Main():
    """Main."""
    #! ----------------------------- 1. Set up some log files -----------------------------------
    #@ Parse Argument
    args = construct_args_preTransfer()
    torch.manual_seed(args.seed)
    print(f"args.seed = {args.seed}")
    folder = str(int(time.time()))
    print(f"Timestamp is {folder}")
    args.OutputPath = os.path.join(args.OutputPath, folder)
    makeFolder(args)
    Output_Configuration_preTransfer(args)
    writer = SummaryWriter(os.path.join(args.OutputPath, 'visual_board'))
    #! -------------------------------------------------------------------------------------------

    #! ----------------------------- 2. Build modules --------------------------------------------
    print('================================================')
    #@ Bulid Model
    print('Building Model...')
    model = BulidModel(args)
    print('Done!')
    print('================================================')

    #@ Bulid Dataloder
    print("Building Train and Test Dataloader...")
    train_source_dataloader, _, init_source_train, dist_source_train_quantity = \
        BulidDataloader(args, flag1='train', flag2='source')
    train_target_dataloader, train_target_data_set, init_target_train, dist_target_train_quantity = \
        BulidDataloader(args, flag1='train', flag2='target', need_strong_trnasform=True)
    test_source_dataloader, _, init_source_test, dist_source_test_quantity =\
        BulidDataloader(args, flag1='test', flag2='source')
    test_target_dataloader, _, init_target_test, dist_target_test_quantity =\
        BulidDataloader(args, flag1='test', flag2='target')
    
    #@ Visulize the Dataset distribution
    proportion_fig, dist_source_train_proportion, dist_target_train_proportion, dist_source_test_proportion,\
         dist_target_test_proportion = Draw_Dataset_Distribution_Bar(args.sourceDataset, dist_source_train_quantity, \
            dist_source_test_quantity, args.targetDataset, dist_target_train_quantity, dist_target_test_quantity, mode='proportion')
    writer.add_figure('Dataset_Distribution/Proportion', proportion_fig, close=True)
    quantity_fig, _, _, _, _ = Draw_Dataset_Distribution_Bar(args.sourceDataset, dist_source_train_quantity, \
            dist_source_test_quantity, args.targetDataset, dist_target_train_quantity, dist_target_test_quantity, mode='quantity')
    writer.add_figure('Dataset_Distribution/Quantity', quantity_fig, close=True)
    print('Done!')
    print('================================================')

    #@  Build Optimizer
    print('Building Optimizer...')
    param_optim = Set_Param_Optim(args, model) # 待优化的参数
    optimizer = Set_Optimizer(args, param_optim, args.lr, args.weight_decay, args.momentum)
    print('Done!')
    print('================================================')

    #@ Loading Checkpoint
    if args.Resume_Model != 'None':
        print(f"Loading Checkpoint... ")
        epoch, classwise, model, optimizer, _, _ = \
            Load_Checkpoint(args, model, optimizer, From='init', To='first')
        print(f"Done!")

    #@ Initialize the best metrics
    Best_Metrics = {
        'acc':{'Train_Source': (0., 0), 'Train_Target': (0., 0), 'Test_Source': (0., 0),'Test_Target': (0., 0)},
        'prec':{'Train_Source': (0., 0), 'Train_Target': (0., 0), 'Test_Source': (0., 0),'Test_Target': (0., 0)},
        'recall':{'Train_Source': (0., 0), 'Train_Target': (0., 0), 'Test_Source': (0., 0),'Test_Target': (0., 0)},
        'f1':{'Train_Source': (0., 0), 'Train_Target': (0., 0), 'Test_Source': (0., 0),'Test_Target': (0., 0)},
    } # (value, epoch)
    confidence = 0
    #! -------------------------------------------------------------------------------------------

    #! ----------------------------- 3. Running Experiments --------------------------------------
    # Running Experiment
    print("Run Experiment...")
    for epoch in range(1, args.epochs + 1):
        if args.showFeature and epoch % 5 == 1:
            print(f"=================\ndraw the tSNE graph...")
            Visualization(args.OutputPath + '/result_pics/train/source/{}_Source.jpg'.format(epoch), model, dataloader=train_source_dataloader, useClassify=True, domain='Source')
            Visualization(args.OutputPath + '/result_pics/train/target/{}_Target.jpg'.format(epoch), model, train_target_dataloader, useClassify=True, domain='Target')

            VisualizationForTwoDomain(args.OutputPath + '/result_pics/train_tow_domain/{}_train'.format(epoch), model, train_source_dataloader, train_target_dataloader, useClassify=True, showClusterCenter=False)
            VisualizationForTwoDomain(args.OutputPath + '/result_pics/test_tow_domain/{}_test'.format(epoch), model, test_source_dataloader, test_target_dataloader, useClassify=True, showClusterCenter=False)
            print(f"finish drawing!\n=================")

        if not args.isTest:
            classwise = Train(args, model, train_source_dataloader, dist_source_train_proportion, init_source_train, \
                train_target_dataloader, dist_target_train_proportion, init_target_train, train_target_data_set, optimizer, epoch, writer, Best_Metrics)
            
        alone = Test(args, model, test_source_dataloader, test_target_dataloader, dist_source_test_proportion, dist_target_test_proportion, init_source_test, init_target_test, Best_Metrics, epoch, confidence, writer)
        
        if alone['value'] >= Best_Metrics[args.judge_criteria]['Test_Source'][0]:
            Save_Checkpoint(args, alone, Best_Metrics, epoch, classwise, model, optimizer)    
        torch.cuda.empty_cache()

    writer.close()
    #! -------------------------------------------------------------------------------------------

if __name__ == '__main__':
    Main()

import os
import sys
import time
import tqdm
import shutil
import argparse
import subprocess
import numpy as np
import pickle
import pandas as pd
from logging import Logger
from sklearn import metrics
from collections import Counter
from selectors import EpollSelector

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from Loss import *
from Utils import *

def Train(args, model, ad_nets, random_layers, train_source_dataloader, train_target_dataloader,\
            labeled_train_target_dataloader, dist_source_train, dist_target_train, train_target_data_set,\
            init_source_train, init_target_train, optimizer, optimizer_ad, epoch, writer, Best_Metrics, feat_pool):
    model.train()

    #! ----------------------------- 1. Define the preformance metric -----------------------------------
    acc_source, prec_source, recall_source = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    acc_target, prec_target, recall_target = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    ce_loss, dan_loss, loss, data_time, batch_time = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], AverageMeter(), AverageMeter(), AverageMeter()
    pl_conf_list, pl_prop_list = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]
    data_time, batch_time = AverageMeter(), AverageMeter()
    six_probilities_source, six_confidences_source = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]
    six_probilities_target, six_confidences_target = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]

    #@ 在七个分类器的预测值都相同的情况下再增加多一个entropy的condition（这些统计参数只用于target domain）
    # delta = 1.9459 / args.num_divided / args.num_divided
    delta = 0.19459 / args.num_divided
    entropy_thresholds = np.arange(delta, 0.19459 + delta, delta)
    probilities_entropy, accuracys_entropy = [AverageMeter() for i in range(args.num_divided)], [AverageMeter() for i in range(args.num_divided)]
    
    #@ Cross entropy
    ce_loss_source, ce_loss_target = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]

    #@ Hinge Loss
    hin_loss = AverageMeter()
    #! -------------------------------------------------------------------------------------------------

    #! ----------------------------- 2. Preparation before trainning -----------------------------------
    #@ Initialize the Adversarial Network
    num_ADNets = [0 for i in range(7)]
    if args.useDAN:
        [ad_nets[i].train() for i in range(len(ad_nets))]

    #@ Decay Learn Rate per Epoch
    if epoch <= 10:
        args.lr, args.lr_ad = 1e-5, 0.0001
    elif epoch <= 20:
        args.lr, args.lr_ad = 5e-6, 0.0001
    else:
        args.lr, args.lr_ad = 1e-6, 0.00001
    

    #@ Define the opimizer
    optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr)
    if args.useDAN:
        optimizer_ad, lr_ad = lr_scheduler_withoutDecay(optimizer_ad, lr=args.lr_ad)

    #@ Get Source/Target Dataloader iterator
    iter_source_dataloader = iter(train_source_dataloader)
    iter_target_dataloader = iter(train_target_dataloader)
    if labeled_train_target_dataloader != None:
        iter_labeled_target_dataloader = iter(labeled_train_target_dataloader)

    #@ Initializes a pseudo label for all target domain data of seven classifier
    selected_labels = [torch.ones((len(train_target_data_set),), dtype=torch.long, ) * -1 for i in range(7)]
    selected_labels = [selected_labels[i].cuda() for i in range(7)]

    #@ Define the learning effect of seven categories of seven classifiers 
    classwise = [torch.zeros((args.class_num,)).cuda() for i in range(7)]
    #! -------------------------------------------------------------------------------------------------

    #! ----------------------------- 3. Start trainning and calculating loss  --------------------------
    num_iter = len(train_source_dataloader) if (len(train_source_dataloader) > len(train_target_dataloader)) else len(train_target_dataloader)
    end = time.time()
    train_bar = tqdm(range(num_iter))
    for step, batch_index in enumerate(train_bar):
        #@ Obtain the input from source and target domain
        try: 
            _, data_source, landmark_source, label_source = next(iter_source_dataloader)
        except:
            iter_source_dataloader = iter(train_source_dataloader)
            _, data_source, landmark_source, label_source = next(iter_source_dataloader)
        try:
            _, data_target, landmark_target, label_target = next(iter_target_dataloader)
        except:
            iter_target_dataloader = iter(train_target_dataloader)
            _, data_target, landmark_target, label_target = next(iter_target_dataloader)
        
        data_time.update(time.time() - end)

        data_source, landmark_source, label_source = data_source.cuda(), landmark_source.cuda(), label_source.cuda()
        data_target, landmark_target, label_target = data_target.cuda(), landmark_target.cuda(), label_target.cuda()

        #@ Forward Propagation
        end = time.time()
        features, preds, layer_features = model(torch.cat((data_source, data_target), 0),\
             torch.cat((landmark_source, landmark_target), 0), return_layer_features=True)
        batch_time.update(time.time() - end)
        loss_ = 0

        #@ Loss 1: Compute Cross Entropy Loss(Source Domain)
        ratio_source = 1.0
        classifiers_ratio_source = [7, 1, 1, 1, 1, 1, 7]
        #@@ cross entropy
        criteria = nn.CrossEntropyLoss()
        #@@ weighted softmax
        # weight = torch.FloatTensor([1 / p for p in dist_source_train]).cuda()
        # criteria = nn.CrossEntropyLoss(weight=weight)
        #@@ focal loss
        # alpha = [1, 1, 1, 1, 1, 1, 1]
        # criteria = FocalLoss(alpha=0.5, gamma=2, num_classes=7)
        for i in range(7):
            tmp = criteria(preds[i].narrow(0, 0, data_source.size(0)), label_source)
            ce_loss[i].update(float(tmp.cpu().data.item()))
            ce_loss_source[i].update(float(tmp.cpu().data.item()))
            loss_ += ratio_source * classifiers_ratio_source[i] * tmp

        #@ Loss 2: Using the pseudo label of target domain to train the classifier
        ratio_flex = 1.0
        classifiers_ratio_flex = [7, 1, 1, 1, 1, 1, 7]
        if labeled_train_target_dataloader != None:
            try:
                img_index_target, data_labeled_target, data_labeled_target_s, landmark_labeled_target, label_labeled_target = next(iter_labeled_target_dataloader)
            except:
                iter_labeled_target_dataloader = iter(labeled_train_target_dataloader)
                img_index_target, data_labeled_target, data_labeled_target_s, landmark_labeled_target, label_labeled_target = next(iter_labeled_target_dataloader)
            
            img_index_target, data_labeled_target, data_labeled_target_s, landmark_labeled_target, label_labeled_target = \
            img_index_target.cuda(), data_labeled_target.cuda(), data_labeled_target_s.cuda(), landmark_labeled_target.cuda(), label_labeled_target.cuda()
            
            feat_w, logits_w, feat_s, logits_s =  model(data_labeled_target, landmark_labeled_target, data_labeled_target_s, use_flexmatch=True)
            pseudo_counters = [Counter(selected_labels[i].tolist()) for i in range(7)] # is a dict of list, each dict = {0: number of class 0, 1: number of class1...}
            for i in range(7): # Traverse the seven classifer
                #@@ Calculate the learning effect of each category under each classifier
                if max(pseudo_counters[i].values()) < len(train_target_data_set):  # not all(5w) -1
                    for class_index in range(args.class_num):
                        classwise[i][class_index] = pseudo_counters[i][class_index] / max(pseudo_counters[i].values())
                #@@ Calculate the loswos
                tmp, _, mask, select, pseudo_lb, _ = ConsistencyLoss(logits_s[i], logits_w[i], classwise[i])
                pl_prop_list[i].update(float(torch.sum(mask).cpu().data.item()), float(mask.size(0)))
                pl_conf_list[i].update(float(torch.sum(pseudo_lb[mask.ge(0.5)] == label_labeled_target[mask.ge(0.5)]).cpu().data.item()), float(torch.sum(mask).cpu().data.item()))
                ce_loss[i].update(float(tmp.cpu().data.item()))
                ce_loss_target[i].update(float(tmp.cpu().data.item()))
                loss_ += ratio_flex * classifiers_ratio_flex[i] * tmp
                #@@ update the pseudo labels set of target domain
                if img_index_target[select == 1].nelement() != 0:
                    selected_labels[i][img_index_target[select == 1]] = pseudo_lb[select == 1]

        #@ Loss 3: Compute the DAN Loss of seven discriminators
        ratio_ad = 0.5
        dan_loss_ = 0
        dan_idx = [0, 1, 1, 1, 1, 1, 2]
        classifiers_ratio_ad = [1, 1, 1, 1, 1, 1, 1]
        softmax = nn.Softmax(dim=1)
        if args.useDAN: # unsupervised learning (binary classification problems)
            for classifier_id in range(7):
                tmp = 0
                softmax_output = softmax(preds[classifier_id])
                if args.methodOfDAN == 'CDAN-E':
                    entropy = Entropy(softmax_output)
                    tmp = CDAN([features[classifier_id], softmax_output], ad_nets[dan_idx[classifier_id]], entropy, calc_coeff(num_iter * (epoch - 1) + batch_index), random_layers[dan_idx[classifier_id]])
                    dan_loss_ += ratio_ad * classifiers_ratio_ad[classifier_id] * tmp
                    dan_loss[classifier_id].update(float(tmp.cpu().data.item()))
                elif args.methodOfDAN == 'CDAN':
                    dan_loss_ = CDAN([features[classifier_id], softmax_output], ad_nets[dan_idx[classifier_id]], None, None, random_layers)
                elif args.methodOfDAN == 'DANN':
                    dan_loss_ = DANN(features[classifier_id], ad_nets[dan_idx[classifier_id]])
        else:
            dan_loss_ = 0
        if args.useDAN:
            loss_ += dan_loss_

        # #@ Loss 4: Using the features pool to calculate the similarity and hinge loss
        # hin_ratio = 0.5
        # tmp = HingeLoss(args, layer_features[4].narrow(0, 0, data_source.size(0)), label_source, feat_pool)
        # hin_loss.update(float(tmp.cpu().data.item()))
        # loss_ += tmp
        # if labeled_train_target_dataloader != None:
        #     features_faked, preds_faked, layer_features_faked = model(data_labeled_target, landmark_labeled_target, return_layer_features=True)
        #     tmp = HingeLoss(args, layer_features_faked[4], label_labeled_target, feat_pool)
        #     hin_loss.update(float(tmp.cpu().data.item()))
        #     loss_ += hin_ratio * tmp

        #@ Back Propagation
        optimizer.zero_grad()
        if args.useDAN:
            optimizer_ad.zero_grad()
        loss_.backward()
        optimizer.step()
        if args.useDAN:
            optimizer_ad.step()
        loss.update(float(loss_.cpu().data.item()))

        #@ Log Adversarial Network Accuracy
        for classifier_id in range(7):
            if args.useDAN:
                if args.methodOfDAN == 'CDAN' or args.methodOfDAN == 'CDAN-E':
                    softmax_output = nn.Softmax(dim=1)(preds[classifier_id])
                    if args.randomLayer:
                        random_out = random_layers[dan_idx[classifier_id]].forward([features[classifier_id], softmax_output])
                        adnet_output = ad_nets[dan_idx[classifier_id]](random_out.view(-1, random_out.size(1)))
                    else:
                        op_out = torch.bmm(softmax_output.unsqueeze(2), features[classifier_id].unsqueeze(1)) # softmax_output's shape is (batchSize, 7, 1) feature's shape is (batchSize, 1, 384)
                        adnet_output = ad_nets[dan_idx[classifier_id]](op_out.view(-1, softmax_output.size(1) * features[classifier_id].size(1)))
                elif args.methodOfDAN == 'DANN':
                    adnet_output = ad_nets[dan_idx](features[classifier_id])

                adnet_output = adnet_output.cpu().data.numpy()
                adnet_output[adnet_output > 0.5] = 1
                adnet_output[adnet_output <= 0.5] = 0
                num_ADNets[classifier_id] += np.sum(adnet_output[:args.train_batch_size]) + (args.train_batch_size - np.sum(adnet_output[args.train_batch_size:]))
        if args.useDAN:
            dan_accs = []
        for classifier_id in range(7):
            dan_accs.append(num_ADNets[classifier_id] / (2.0 * args.train_batch_size * num_iter))

        #@ 🔺Compute the seven classifiers' accuracy, precision, recall and F1-score
        for classifier_id in range(7):
            Compute_Accuracy(args, preds[classifier_id].narrow(0, 0, data_source.size(0)), label_source, acc_source[classifier_id], prec_source[classifier_id], recall_source[classifier_id])
        for classifier_id in range(7):
            Compute_Accuracy(args, preds[classifier_id].narrow(0, data_target.size(0), data_target.size(0)), label_target, acc_target[classifier_id], prec_target[classifier_id], recall_target[classifier_id])

        #@ Compute the proportion and confidence of pseudo labels produced by different number of classifiers predicting the same time are calculated
        Count_Probility_Accuracy(six_probilities_source, six_confidences_source, [pred.narrow(0, 0, data_source.size(0)) for pred in preds], label_source)
        Count_Probility_Accuracy(six_probilities_target, six_confidences_target, [pred.narrow(0, data_target.size(0), data_target.size(0)) for pred in preds], label_target)

        #@ ⭕Compute the confidence of the pseudo label in the target domain is calculated under the restriction of different entropies 
        Count_Probility_Accuracy_Entropy(entropy_thresholds, probilities_entropy, accuracys_entropy, [pred.narrow(0, data_target.size(0), data_target.size(0)) for pred in preds], label_target)
        end = time.time()
        
        #@ Visualize the input imgs
        if step % 200 == -1:
            Visualize_Input_Image(args, writer, 'train', 'source', data_source, data_raw_path=init_source_train['imgs_list'])
            Visualize_Input_Image(args, writer, 'train', 'target', data_labeled_target, data_s_trans=data_labeled_target_s, data_raw_path=init_target_train['imgs_list'])

        train_bar.desc = f'[Train (Source Domain) Epoch {epoch}/{args.epochs}] '\
            f'G:({ce_loss_source[0].avg:.3f},{ce_loss_target[0].avg:.3f},{dan_loss[0].avg:.3f},{hin_loss.avg:.3f}) '\
            f'L:({ce_loss_source[3].avg:.3f},{ce_loss_target[3].avg:.3f},{dan_loss[3].avg:.3f},{hin_loss.avg:.3f}) '\
            f'GL:({ce_loss_source[6].avg:.3f},{ce_loss_target[6].avg:.3f},{dan_loss[6].avg:.3f},{hin_loss.avg:.3f}) '\
    
    #@ 🔺Compute the seven classifiers' overral accuracy, precision, recall and F1-score
    accs_source = Show_OnlyAccuracy(acc_source)
    precs_source, recalls_source, f1s_source = Show_OtherMetrics(prec_source, recall_source, [i for i in range(7)]) # 第三个参数就是随便凑个长度为7的list而已
    accs_target = Show_OnlyAccuracy(acc_target) # 因为这里用到了target的伪标签去训练target数据, 所以可以可视化看一下target的效果有没有上升
    precs_target, recalls_target, f1s_target = Show_OtherMetrics(prec_target, recall_target, [i for i in range(7)]) # 第三个参数就是随便凑个长度为7的list而已

    #@ ⭕Compute the overal confidence of the pseudo label in the target domain is calculated under the restriction of different entropies 
    acc_dic, pro_dic = {}, {}
    for i in range(args.num_divided):
        acc_dic.update({'entropy_' + str(entropy_thresholds[i]): accuracys_entropy[i].avg})
        pro_dic.update({'entropy_' + str(entropy_thresholds[i]): probilities_entropy[i].avg})
    #! --------------------------------------------------------------------------------------------------

    #! --------------------------------- 4. Record the results ------------------------------------------
    #@ Visualize the prediction effect of each classifier for each category
    fig = Draw_Category_Metrics_Bar(prec_source, recall_source)
    writer.add_figure('Category_Metrics/Train_Source', fig, global_step=epoch, close=True)
    fig = Draw_Category_Metrics_Bar(prec_target, recall_target)
    writer.add_figure('Category_Metrics/Train_Target', fig, global_step=epoch, close=True)
    
    #@ Visualize the prediction performance curve of each classifier
    Visualize_Transfer_Common_Result(epoch, writer, accs_source, precs_source, recalls_source, f1s_source, ce_loss_source,\
    six_confidences_source, six_probilities_source, state='Train', domain='Source')
    Visualize_Transfer_Common_Result(epoch, writer, accs_target, precs_target, recalls_target, f1s_target, ce_loss_target,\
    six_confidences_target, six_probilities_target, state='Train', domain='Target')
    Visualize_Transfer_OnlyinTraining_Result(epoch, writer, dan_accs, dan_loss, acc_dic, pro_dic, hin_loss)

    #@ Visualize the collection of pseudo labels
    fig = Visualize_Collection_Pseudo_Labels_Category(pseudo_counters, mode='proportion')
    writer.add_figure('Category_Pseudo_Labels/PropBar-Category', fig, global_step=epoch, close=True)
    fig = Visualize_Collection_Pseudo_Labels_Category(pseudo_counters, mode='quantity')
    writer.add_figure('Category_Pseudo_Labels/QuantityBar-Category', fig, global_step=epoch, close=True)
    fig = Visualize_Collection_Pseudo_Labels_Category_ClassWise(classwise)
    writer.add_figure('Category_Pseudo_Labels/ClassWise-Category', fig, global_step=epoch, close=True)
    fig = Visualize_Collection_Pseudo_Labels_General(pl_conf_list, pl_prop_list)
    writer.add_figure('Category_Pseudo_Labels/Prop&Conf-General', fig, global_step=epoch, close=True)
    Visualize_Collection_Difference_PL_between_Classifiers(writer, selected_labels)

    #@ Update the best metrics and get the number of modified metrics
    modified_metrics = Update_Best_Metrics(Best_Metrics, accs_source, precs_source, recalls_source, f1s_source, epoch, state='Train', domain='Source')
    modified_metrics += Update_Best_Metrics(Best_Metrics, accs_target, precs_target, recalls_target, f1s_target, epoch, state='Train', domain='Target')
    
    #@ Write the log information
    EpochInfo, HistoryInfo = Write_Results_Log(Best_Metrics, accs_source, accs_target, precs_source, precs_target, \
    recalls_source, recalls_target, f1s_source, f1s_target, epoch, args, mode='Train')
    if modified_metrics > 0:
        LoggerInfo = EpochInfo + HistoryInfo
    else:
        LoggerInfo = EpochInfo
    with open(args.OutputPath + "/transfer_train_result.md", "a") as f:
        f.writelines(LoggerInfo)
    
    #@ Record some memo information on the tensorboard
    memo =  f"**Save**: {os.getcwd() + args.OutputPath.replace('.', '')}  \n"\
            f"**Loss Ratio**:  \n"\
            f">Source\t= {ratio_source} x {classifiers_ratio_source}  \n"\
            f">Flex\t= {ratio_flex} x {classifiers_ratio_flex}  \n"\
            f">Adverserial\t= {ratio_ad} * {classifiers_ratio_ad}  \n"
    writer.add_text('MarkNote', memo, 1)
    #! -------------------------------------------------------------------------------------------------

    return classwise

def Test(args, model, test_source_dataloader, test_target_dataloader, dist_source_test, dist_target_test,\
        init_source_test, init_target_test, Best_Metrics, classwise, epoch, writer):
    """Test."""

    model.eval()
    #! ----------------------------- 1. Test On Source Domain  -----------------------------------------
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
        weight = torch.FloatTensor([1 / p for p in dist_source_test]).cuda()
        criteria = nn.CrossEntropyLoss(weight=weight)
        #@@ focal loss
        # alpha = [1, 1, 1, 1, 1, 1, 1]
        # criteria = FocalLoss(alpha=0.5, gamma=2, num_classes=7)
        for i in range(7):
            tmp = criteria(preds[i], target)
            ce_loss[i].update(float(tmp.cpu().data.item()))
            loss_ += tmp
        loss.update(float(loss_.cpu().data.numpy()))

        #@@ 🔺 Compute the seven classifiers' accuracy, precision, recall and F1-score
        for classifier_id in range(7):
            Compute_Accuracy(args, preds[classifier_id], target, acc_source[classifier_id], prec_source[classifier_id], recall_source[classifier_id])

        #@@ Compute the proportion and confidence of pseudo labels produced by different number of classifiers predicting the same time are calculated
        Count_Probility_Accuracy(six_probilities, six_confidences, preds, target)

        #@ Visualize the input imgs
        if batch_index % 200 == -1:
            Visualize_Input_Image(args, writer, 'test', 'source', input, data_raw_path=init_source_test['imgs_list'])

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
    #@@ Update the best metrics
    modified_metrics_source = Update_Best_Metrics(Best_Metrics, accs_source, precs_source,\
         recalls_source, f1s_source, epoch, state='Test', domain='Source')
    #! -------------------------------------------------------------------------------------------------

    #! ----------------------------- 2. Test On Target Domain  -----------------------------------------
    #@  Define the preformance metric
    acc_target, prec_target, recall_target = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    ce_loss, loss, data_time, batch_time = [AverageMeter() for i in range(7)], AverageMeter(), AverageMeter(), AverageMeter()
    six_probilities, six_confidences = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]

    alone_acc, alone_recall, alone_prec = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]

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
        weight = torch.FloatTensor([1 / p for p in dist_target_test]).cuda()
        criteria = nn.CrossEntropyLoss(weight=weight)
        #@@ focal loss
        # alpha = [1, 1, 1, 1, 1, 1, 1]
        # criteria = FocalLoss(alpha=0.5, gamma=2, num_classes=7)
        for i in range(7):
            tmp = criteria(preds[i], target)
            ce_loss[i].update(float(tmp.cpu().data.item()))
            loss_ += tmp
        loss.update(float(loss_.cpu().data.numpy()))

        #@@ 🔺 Compute the seven classifiers' accuracy, precision, recall and F1-score
        for classifier_id in range(7):
            Compute_Accuracy(args, preds[classifier_id], target, acc_target[classifier_id], prec_target[classifier_id], recall_target[classifier_id])

        #@ ⚪ Compute new acc alone
        if classwise != None:
            Compute_Alone_Accuracy(args, preds, target, alone_acc, alone_recall, alone_prec, classwise)

        #@@ Compute the proportion and confidence of pseudo labels produced by different number of classifiers predicting the same time are calculated
        Count_Probility_Accuracy(six_probilities, six_confidences, preds, target)

        #@ Visualize the input imgs
        if batch_index % 200 == -1:
            Visualize_Input_Image(args, writer, 'test', 'target', input, data_raw_path=init_target_test['imgs_list'])

        #@@ Ending
        end = time.time()
        test_target_bar.desc = f'[Test (Target Domain) Epoch {epoch}/{args.epochs}] ce loss: {ce_loss[0].avg:.3f}, {ce_loss[1].avg:.3f}, {ce_loss[2].avg:.3f}, {ce_loss[3].avg:.3f}, {ce_loss[4].avg:.3f}, {ce_loss[5].avg:.3f}, {ce_loss[6].avg:.3f}'
    
    #@ 🔺 Compute the seven classifiers' overal accuracy, precision, recall and F1-score
    accs_target = Show_OnlyAccuracy(acc_target)
    precs_target, recalls_target, f1s_target = Show_OtherMetrics(prec_target, recall_target, [i for i in range(7)])

    #@ ⚪ Compute new acc alone
    if classwise != None:
        accs_alone = Show_OnlyAccuracy([alone_acc])
        precs_alone, recalls_alone, f1s_alone = Show_OtherMetrics([alone_prec], [alone_recall], [i for i in range(7)])
        alone = {'acc': accs_alone[0], 'prec': precs_alone[0], 'recall': recalls_alone[0], 'f1': f1s_alone[0]}
        writer.add_scalars('Alone', {'acc': accs_alone[0], 'recall':recalls_alone[0], 'precision':precs_alone[0], 'f1':f1s_alone[0]}, epoch)

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

    #! ----------------------------- 3. write the log file  --------------------------------------------
    EpochInfo, HistoryInfo = Write_Results_Log(Best_Metrics, accs_source, accs_target, precs_source, precs_target, \
    recalls_source, recalls_target, f1s_source, f1s_target, epoch, args, mode='Test')
    if modified_metrics_source + modified_metrics_target > 0:
        LoggerInfo = EpochInfo + HistoryInfo
    else:
        LoggerInfo = EpochInfo
    with open(args.OutputPath + "/transfer_test_result.md","a") as f:
        f.writelines(LoggerInfo)
    #! -------------------------------------------------------------------------------------------------
    return alone

def Main():
    """Main."""

    #! ----------------------------- 1. Set up some log files -----------------------------------
    #@ Parse Argument
    args = construct_args_transfer()
    torch.manual_seed(args.seed)
    folder = str(int(time.time()))
    print(f"Timestamp is {folder}")
    args.OutputPath = os.path.join(args.OutputPath, folder)
    makeFolder(args)
    Output_Configuration_transfer(args)
    writer = SummaryWriter(os.path.join(args.OutputPath, 'visual_board'))
    #! -------------------------------------------------------------------------------------------

    #! ----------------------------- 2. Build modules --------------------------------------------
    #@ Bulid Model
    print('================================================')
    print('Building Model...')
    try:
        model, classwise = BulidModel(args)
        print(f"==> Got the classwise!")
    except:
        classwise = None
        model = BulidModel(args)
    print('Done!')
    print('================================================')

    #@ Bulid Dataloder
    print("Building Train and Test Dataloader...")
    train_source_dataloader, _, init_source_train, dist_source_train_quantity = \
        BulidDataloader(args, flag1='train', flag2='source')
    train_target_dataloader, train_target_data_set, init_target_train, dist_target_train_quantity = \
        BulidDataloader(args, flag1='train', flag2='target')
    test_source_dataloader, _, init_source_test, dist_source_test_quantity =\
        BulidDataloader(args, flag1='test', flag2='source')
    test_target_dataloader, _, init_target_test, dist_target_test_quantity =\
        BulidDataloader(args, flag1='test', flag2='target')
    labeled_train_target_loader = None

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

    #@ Bulid Adversarial Network
    print('Building Adversarial Network...')
    random_layers, ad_nets = [], []
    for i in range(2):
        random_layer, ad_net = BulidAdversarialNetwork(args, 64, args.class_num) if args.useDAN else (None, None)
        ad_nets.append(ad_net)
        random_layers.append(random_layer)
    random_layer, ad_net = BulidAdversarialNetwork(args, 384, args.class_num) if args.useDAN else (None, None)
    ad_nets.append(ad_net)
    random_layers.append(random_layer)
    print('Done!')
    print('================================================')

    #@ Build Optimizer
    print('Building Optimizer...')
    param_optim = Set_Param_Optim(args, model)
    optimizer = Set_Optimizer(args, param_optim, args.lr, args.weight_decay, args.momentum)

    param_optim_ads = []
    for i in range(len(ad_nets)):
        param_optim_ads += Set_Param_Optim(args, ad_nets[i]) if args.useDAN else None
    optimizer_ad = Set_Optimizer(args, param_optim_ads, args.lr_ad, args.weight_decay,
                                 args.momentum) if args.useDAN else None
    
    #@ Loading Checkpoint
    if args.Resume_Model != 'None':
        print(f"Loading Checkpoint ! ")
        epoch, classwise, model, optimizer, ad_nets, optimizer_ad = \
            Load_Checkpoint(args, model, optimizer, ad_nets, optimizer_ad, From='first', To='second')

    else:
        print('No Resume Model')
    print('Done!')
    print('================================================')

    #@ Initialize the best metrics
    Best_Metrics = {
        'acc':{'Train_Source': (0., 0), 'Train_Target': (0., 0), 'Test_Source': (0., 0),'Test_Target': (0., 0)},
        'prec':{'Train_Source': (0., 0), 'Train_Target': (0., 0), 'Test_Source': (0., 0),'Test_Target': (0., 0)},
        'recall':{'Train_Source': (0., 0), 'Train_Target': (0., 0), 'Test_Source': (0., 0),'Test_Target': (0., 0)},
        'f1':{'Train_Source': (0., 0), 'Train_Target': (0., 0), 'Test_Source': (0., 0),'Test_Target': (0., 0)},
        'alone': {'acc': (0., 0), 'prec': (0., 0), 'recall': (0., 0), 'f1': (0., 0)}
    } # (value, epoch)
    #@ Initialize the feature pool
    feat_pool = {'source': [], 'target': []}
    #! -------------------------------------------------------------------------------------------

    #! ----------------------------- 3. Running Experiments --------------------------------------
    print("Run Experiment...")
    for epoch in range(epoch, args.epochs + 1):

        if not args.isTest:
            if epoch <= 0:
                confidence_threshold = 0.97
                pi, ei, et = [0, 6, 1, 2, 3, 4, 5], [0, 6, 1, 2, 3, 4, 5], 0.19459
            else:
                confidence_threshold = 0.0
                pi, ei, et = [0], [0], 0.19459e2
            #@ Pseuod labels generation
            labeled_train_target_loader_, confidence_, proportion_, category_confidece_, category_proportion_ = \
                BuildLabeledDataloader(args, train_target_dataloader, init_target_train, model, confidence_threshold=confidence_threshold,\
                    prediction_indexs=pi, entropy_indexs=ei, entropy_threshold=et)
            if confidence_ >= confidence_threshold :
                labeled_train_target_loader = labeled_train_target_loader_
                confidence, proportion, category_confidece, category_proportion = confidence_, proportion_, category_confidece_, category_proportion_
            Visualize_Pseudo_Labels_Generator(args, epoch, writer, confidence, proportion, category_confidece, category_proportion)

            # #@ Build features pool
            # source_sim = Cal_Class_Similarity_in_Domain(args, model, train_source_dataloader, 'source', int(min(dist_target_train_quantity)/2))
            # # target_sim = Cal_Class_Similarity_in_Domain(args, model, labeled_train_target_loader, 'source', int(min(dist_target_train_quantity)/2))
            # feat_pool = Build_Features_Pool(args, epoch, model, train_source_dataloader, labeled_train_target_loader,\
            #      source_sim, collect_num=int(min(dist_target_train_quantity)/2))

            classwise = Train(args, model, ad_nets, random_layers, train_source_dataloader, train_target_dataloader,\
                 labeled_train_target_loader, dist_source_train_proportion, dist_target_train_proportion, \
                 train_target_data_set, init_source_train, init_target_train, optimizer, optimizer_ad, epoch, writer, Best_Metrics, feat_pool)

        #@ Calculate the similarity of seven expressions in the same domain and Visulization
        # Calculate_Visualize_Class_Similarity(args, model, train_source_dataloader, test_source_dataloader, train_target_dataloader, test_target_dataloader, writer)
        alone = Test(args, model, test_source_dataloader, test_target_dataloader, dist_source_test_proportion, \
            dist_target_test_proportion, init_source_test, init_target_test, Best_Metrics, classwise, epoch, writer)

        if alone[args.judge_criteria] > Best_Metrics['alone'][args.judge_criteria][0]:
            Save_Checkpoint(args, alone, Best_Metrics, epoch, classwise, model, optimizer, ad_nets, optimizer_ad, From='second')   

    writer.close()
    print(f"==========================\n{args.Backbone + args.Log_Name + args.sourceDataset + 'to' + args.targetDataset} is done, ")
    print(f"saved in: {args.OutputPath}")
    #! -------------------------------------------------------------------------------------------

if __name__ == '__main__':
    Main()

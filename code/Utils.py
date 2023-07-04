import os
import cv2
import sys
import json
import copy
import time
import shutil
import argparse
import subprocess
import random
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from functools import cmp_to_key

import sklearn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from PIL import Image, ImageDraw

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Dataset import MyDataset
from RandAugment import RandAugment
from VGG import VGG, VGG_onlyGlobal
from ResNet import IR, IR_onlyGlobal
from MobileNet import MobileNetV2, MobileNetV2_onlyGlobal
from AdversarialNetwork import RandomLayer, AdversarialNetwork, calc_coeff

class AverageMeter(object):
    '''Computes and stores the sum, count and average'''
    def __init__(self):
        self.reset()

    def reset(self):    
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val 
        self.count += count

        if self.count==0:
            self.avg = 0
        else:
            self.avg = float(self.sum) / self.count

def str2bool(input):
    if isinstance(input, bool):
       return input
    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def Set_Param_Optim(args, model):
    """Set Parameters for optimization."""
    
    if isinstance(model, nn.DataParallel):
        return model.module.get_parameters()

    return model.get_parameters()

def Set_Optimizer(args, parameter_list, lr=0.001, weight_decay=0.0005, momentum=0.9):
    """Set Optimizer."""
    
    return optim.SGD(parameter_list, lr=lr, weight_decay=weight_decay, momentum=momentum)

def lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    lr = lr * (1 + gamma * iter_num) ** (-power)

    for param_group in optimizer.param_groups:
        
        if 'lr_mult' in param_group:
            param_group['lr'] = lr * param_group['lr_mult']
        else:
            param_group['lr'] = lr

        if 'decay_mult' in param_group:    
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        else:
            param_group['weight_decay'] = weight_decay

    return optimizer, lr

def lr_scheduler_withoutDecay(optimizer, lr=0.001, weight_decay=0.0005):
    """Learning rate without Decay."""

    for param_group in optimizer.param_groups:
        
        if 'lr_mult' in param_group:
            param_group['lr'] = lr * param_group['lr_mult']
        else:
            param_group['lr'] = lr

        if 'decay_mult' in param_group:    
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        else:
            param_group['weight_decay'] = weight_decay

    return optimizer, lr

def Compute_Accuracy(args, pred, target, acc, prec, recall):
    '''Compute the accuracy of all samples, the accuracy of positive samples, the recall of positive samples.'''

    pred = pred.cpu().data.numpy()
    pred = np.argmax(pred, axis=1)
    target = target.cpu().data.numpy()

    pred = pred.astype(np.int32).reshape(pred.shape[0],)
    target = target.astype(np.int32).reshape(target.shape[0],)
    
    for i in range(7):
        TP = np.sum((pred == i)*(target == i))
        TN = np.sum((pred != i)*(target != i))

        # Compute Accuracy of All --> TP+TN / All
        acc[i].update(np.sum(pred == target), pred.shape[0])

        # Compute Precision of Positive --> TP/(TP+FP)
        prec[i].update(TP, np.sum(pred==i))

        # Compute Recall of Positive --> TP/(TP+FN)
        recall[i].update(TP, np.sum(target==i))

def Compute_Alone_Accuracy(args, preds, target, acc, recall, prec, classwises):
    #@ Preprocess
    flags = [6, 0]
    pred_final = []
    preds = [torch.softmax(p, dim=-1) for p in preds]
    for i in range(preds[flags[0]].size(0)):
        hit = False
        for flag in flags:
            pred = preds[flag][i].cpu().data.numpy()
            pred_class = np.argmax(pred, axis=0)
            score_pred = float(preds[flag][i][pred_class].data.item())
            score_classwise = float(classwises[flag][pred_class].data.item())
            # if score_pred >= 0.95 * score_classwise :
            if score_pred >= 0.95 * pow((score_classwise + 1)/2, 2):
            # if score_pred >= 0.95 * score_classwise / (2 - score_classwise):
                pred_final.append(pred_class)
                hit = True
                break
        if not hit:
            scores = torch.cat([preds[j][i].unsqueeze(0) for j in range(len(preds))], dim=0)
            threshold = torch.cat([classwises[j].unsqueeze(0) for j in range(len(preds))], dim=0)
            mask = torch.ge(scores, threshold)
            scores = scores * mask
            scores = torch.sum(scores, dim=0)
            pred = scores.cpu().data.numpy()
            pred_class = np.argmax(pred, axis=0)
            pred_final.append(pred_class)
    pred = np.array(pred_final)

    #@ Start Caculating
    target = target.cpu().data.numpy()
    pred = pred.astype(np.int32).reshape(target.shape[0],)
    target = target.astype(np.int32).reshape(target.shape[0],)

    for i in range(7):
        TP = np.sum((pred == i)*(target == i))
        TN = np.sum((pred != i)*(target != i))

        # Compute Accuracy of All --> TP+TN / All
        acc[i].update(np.sum(pred == target), pred.shape[0])

        # Compute Precision of Positive --> TP/(TP+FP)
        prec[i].update(TP, np.sum(pred==i))

        # Compute Recall of Positive --> TP/(TP+FN)
        recall[i].update(TP, np.sum(target==i))

def Count_Probility_Accuracy(six_probilities, six_accuracys, preds, target):
    preds = [np.argmax(pred.cpu().data.numpy(), axis=1) for pred in preds]
    target = target.cpu().data.numpy()
    
    #@ situation_0: global == global_local
    six_probilities[0].update(np.sum(preds[0] == preds[6]), len(preds[0]))
    six_accuracys[0].update(np.sum((preds[0] == preds[6]) * (preds[0] == target)), len(preds[0]))
    boolMatrics = (preds[0] == preds[6])

    #@ situation_1: global == global_local && one local also predict the same
    for img_index in range(len(preds[0])):
        cnt = 0
        if boolMatrics[img_index]: # 前提条件是global 和 global_local分类器预测值是相同的
            if preds[1][img_index] == preds[0][img_index]:
                cnt += 1
            if preds[2][img_index] == preds[0][img_index]:
                cnt += 1
            if preds[3][img_index] == preds[0][img_index]:
                cnt += 1
            if preds[4][img_index] == preds[0][img_index]:
                cnt += 1
            if preds[5][img_index] == preds[0][img_index]:
                cnt += 1
        for statisc in range(1, cnt + 1):
            six_probilities[statisc].update(1, 1) # 正更新
            if preds[0][img_index] == target[img_index]: # 预测正确
                six_accuracys[statisc].update(1, 1)
            else:
                six_accuracys[statisc].update(0, 1)

        for statisc_ in range(cnt+1, 6):
            six_probilities[statisc_].update(0, 1) # 负更新
            # six_accuracys[statisc_].update(0, 1)
def Count_Probility_Accuracy_Entropy(entropy_thresholds, probilities, accuracys,  preds_, target):
    '''
        entropy_thresholds: 信息熵的阈值列表，[0, log2(7)], n等分, 所以这个列表的长度是n
        probilities: 长度为n, probilities[i]代表的是在七个分类器都相同的情况下,且七个分类器的信息熵都小于entropy_thresholds[i]的概率
        accuracys: 长度为n, accuracys[i]代表的是在七个分类器都相同的情况下, 且七个分类器的信息熵都小于entropy_thresholds[i]的情况下, 七个分类器给出的共同的伪标签跟gt对比之下的可信度
        preds: shape是(7, batchSize, 7), 没有经过softmax处理的原始预测值
        target: shape是(batchSize), 每张图片的真是groundtruth
    '''

    preds = [np.argmax(pred.cpu().data.numpy(), axis=1) for pred in preds_]     # shape is (7, batchSize,)
    target = target.cpu().data.numpy()

    softmaxs = [nn.Softmax(dim=1)(preds_[i]) for i in range(7)]    # shape是（7, batchSize,）
    entropys = [torch.sum(-softmaxs[i] * torch.log(softmaxs[i] + 1e-5), dim=1) for i in range(7)] #计算七个分类器的信息熵, shape是(7, batchSize, 1)
    
    num_thresholds = len(entropy_thresholds)
    boolMatrics = (preds[0] == preds[6]) # global 和 global + local 这两个分类器的预测值相同的0-1矩阵

    #@ 七个分类器预测值都相同且七个分类器的信息熵都小于各个阈值的情况的数量统计和给出的伪标签的可信度
    for enp_index in range(num_thresholds):
        for img_index in range(len(preds[0])):
            found = False
            if preds[0][img_index] == preds[1][img_index] == preds[2][img_index] == preds[3][img_index] == preds[4][img_index] == preds[5][img_index]: # 七个分类器的预测值相同
                if entropys[0][img_index] < entropy_thresholds[enp_index] and entropys[1][img_index] < entropy_thresholds[enp_index] and entropys[2][img_index] < entropy_thresholds[enp_index] and\
                    entropys[3][img_index] < entropy_thresholds[enp_index] and entropys[4][img_index] < entropy_thresholds[enp_index] and entropys[5][img_index] < entropy_thresholds[enp_index] and entropys[6][img_index] < entropy_thresholds[enp_index]: # 七个分类器的信息熵
                    probilities[enp_index].update(1, 1)
                    if preds[0][img_index] == target[img_index]:
                        accuracys[enp_index].update(1, 1)
                    else:
                        accuracys[enp_index].update(0, 1)
                    found = True
            if not found:
                probilities[enp_index].update(0, 1)

def Collect_Labeled_Image_Indexs(indexs, preds_, target, prediction_indexs=[0, 6], entropy_indexs=[0, 6], entropy_threshold=0.19459):
    preds = [np.argmax(pred.cpu().data.numpy(), axis=1) for pred in preds_]     # shape is (7, batchSize,)
    target = target.cpu().data.numpy()

    softmaxs = [nn.Softmax(dim=1)(preds_[i]) for i in range(7)]    # update to shape是（8, batchSize,）
    entropys = [torch.sum(-softmaxs[i] * torch.log(softmaxs[i] + 1e-5), dim=1) for i in range(7)] #计算七个分类器的信息熵, shape是(7, batchSize, 1)

    collect_indexs, fake_labels, true_labels = [], [], []

    def check_entropy(img_index, indexs=[0, 6]):
        for i in indexs:
            if entropys[i][img_index] > entropy_threshold:
                return False
        return True
    def check_prediction(img_index, indexs=[0, 6]):
        prediction = preds[indexs[0]][img_index]
        for i in indexs[1:]:
            if preds[i][img_index] != prediction:
                return False
        return True

    #@ different condiction to generate pseudo labels
    for img_index in range(len(preds[0])):
        if check_prediction(img_index, prediction_indexs) and check_entropy(img_index, entropy_indexs):
            collect_indexs.append(indexs[img_index])
            fake_labels.append(preds[0][img_index])
            true_labels.append(target[img_index])

    return collect_indexs, fake_labels, true_labels

def BulidModel(args):
    """Bulid Model."""

    if args.useLocalFeature:
        if args.Backbone == 'ResNet18':
            model = IR(18, args.useIntraGCN, args.useInterGCN, args.useRandomMatrix, args.useAllOneMatrix, args.useCov, args.useCluster)
        elif args.Backbone == 'ResNet50':
            model = IR(50, args.useIntraGCN, args.useInterGCN, args.useRandomMatrix, args.useAllOneMatrix, args.useCov, args.useCluster)
        elif args.Backbone == 'VGGNet':
            model = VGG(args.useIntraGCN, args.useInterGCN, args.useRandomMatrix, args.useAllOneMatrix, args.useCov, args.useCluster)
        elif args.Backbone == 'MobileNet':
            model = MobileNetV2(args.useIntraGCN, args.useInterGCN, args.useRandomMatrix, args.useAllOneMatrix, args.useCov, args.useCluster)
    else:
        if args.Backbone == 'ResNet18':
            model = IR_onlyGlobal(18)
        elif args.Backbone == 'ResNet50':
            model = IR_onlyGlobal(50)
        elif args.Backbone == 'VGGNet':
            model = VGG_onlyGlobal()
        elif args.Backbone == 'MobileNet':
            model = MobileNetV2_onlyGlobal()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.cuda()
    return model

def Load_Checkpoint(args, model, optimizer, ad_nets=None, optimizer_ad=None, From='init', To='first'):
    assert From in ['init', 'first', 'second']

    model_weight_path  = args.Resume_Model
    if From  == 'init':
        pretrained_dict = torch.load(model_weight_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        last_epoch = 1
        classwise = None

    elif From == 'first':
        dic = torch.load(model_weight_path)
        model_param_dict = dic['model']
        model_dict = model.state_dict()
        model_param_dict = {k: v for k, v in model_param_dict.items() if k in model_dict}
        model_dict.update(model_param_dict)
        model.load_state_dict(model_dict)
        optimizer.load_state_dict(dic['optimizer'])
        classwise = dic['classwise']
        if To == 'first':
            last_epoch = dic['epoch']
        elif To == 'second':
            last_epoch = 1
    
    elif From == 'second':
        dic = torch.load(model_weight_path)
        
        model_param_dict = dic['model']
        model_dict = model.state_dict()
        model_param_dict = {k: v for k, v in model_param_dict.items() if k in model_dict}
        model_dict.update(model_param_dict)
        model.load_state_dict(model_dict)
        
        for i in range(len(ad_nets)):
            ad_nets[i].load_state_dict(dic['ad_nets'][i])

        optimizer.load_state_dict(dic['optimizer'])
        optimizer_ad.load_state_dict(dic['optimizer_ad'])

        last_epoch = dic['epoch']
        classwise = dic['classwise']

    return last_epoch, classwise, model, optimizer, ad_nets, optimizer_ad

def Save_Checkpoint(args, alone, Best_Metrics, epoch, classwise, model, optimizer, ad_nets=None, optimizer_ad=None, From='first'):
    assert From in ['first', 'second']
    if From == 'first':
        print("***************")
        best_name, best_value = alone['name'], alone['value']
        print(f'[Save] Best {str.capitalize(args.judge_criteria)}: {best_value:.2%}, the classifier is {best_name}. Save the checkpoint!')
        print("***************")
    elif From == 'second':
        print("***************")
        for k in alone.keys():
            Best_Metrics['alone'][k] = (alone[k], epoch)
        print(f'[Save] Best {str.capitalize(args.judge_criteria)}: {alone[args.judge_criteria]:.2%}, the classifier is alone. Save the checkpoint!')
        print("***************")

    torch.save({
        'epoch': epoch, 
        'classwise': classwise,
        'optimizer': optimizer.state_dict(),
        'optimizer_ad': optimizer_ad.state_dict() if optimizer_ad != None else None,
        'ad_nets': [ad_nets[i].state_dict() for i in range(len(ad_nets))] if ad_nets != None else None,
        'model': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict()},
    os.path.join(args.OutputPath, '{}.pkl'.format(args.Backbone+args.Log_Name+args.sourceDataset+'to'+args.targetDataset)))

def Compute_Confidence_Proportion(args, dataloader, init_dataset_data, model, eps=1e-6,\
    prediction_indexs=[0, 6], entropy_indexs=[0, 6], entropy_threshold=0.19459, require_data=False):
    """Test."""
    model.eval()
    bar = tqdm(dataloader)
    data_imgs, data_labels, data_bboxs, data_landmarks, target_labels = [], [], [], [], []
    collect_num = 0
    for batch_index, (indexs, input, landmark, target) in enumerate(bar):
        input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()

        # Forward Propagation
        with torch.no_grad():
            features, preds = model(input, landmark)
        collect_list_index, fake_labels,  true_labels = Collect_Labeled_Image_Indexs(indexs, preds, target, \
            prediction_indexs=prediction_indexs, entropy_indexs=entropy_indexs, entropy_threshold=entropy_threshold)
        data_labels += fake_labels
        target_labels += true_labels
        collect_num += len(collect_list_index)
        for c_index in collect_list_index:
            data_imgs.append(init_dataset_data['imgs_list'][c_index])
            data_bboxs.append(init_dataset_data['bboxs_list'][c_index])
            data_landmarks.append(init_dataset_data['landmarks_list'][c_index])
        bar.desc = f'[Calculating the confi & prop: ({prediction_indexs}), ({entropy_indexs}), ({entropy_threshold}) ]'

    #@ Calculate the proportion of the target domain data with the generated pseudo-labels in the whole dataset
    proportion = len(data_labels) / (len(dataloader) * args.train_batch_size + eps)
    data_labels_np = np.array(data_labels)
    target_labels_np = np.array(target_labels)
    #@ Calculate the overall confidence of the generated pseudo labels
    confidence = np.sum(data_labels_np == target_labels_np) / (len(data_labels) + eps)
    #@ Calculate the confidence of the generated pseudo labels for each category
    category_confidece = [np.sum(data_labels_np[np.argwhere((data_labels_np == target_labels_np) > 0)] == i) / (np.sum(data_labels_np == i) + eps) for i in range(7)]
    #@ Calculate the distribution proportion of the number of each category in the pseudo label
    category_proportion = [np.sum(data_labels_np == i) / (len(data_labels_np) + eps) for i in range(7)]
    print(f"{proportion:.2%} of target data with {confidence:.2%} confidence have been collected.")
    print(f"----> category confidence: {category_confidece}")
    print(f"----> category proportion: {category_proportion}")
    if require_data:
        return data_imgs, target_labels, data_bboxs, data_landmarks, collect_num, confidence, proportion, category_confidece, category_proportion
    else:
        return confidence, proportion, category_confidece, category_proportion

def Find_Best_Group_toGenerate_PseudoLabels(args, dataloader, init_dataset_data, model, confidence_threshold=0.80):
    
    def cmp(a, b): # -1 means you don't have to switch positions
        if a['proportion'] > b['proportion']:
            return -1
        elif a['proportion'] == b['proportion'] and a['confidence'] > b['confidence']:
            return -1
        return 1
    def get_info(num, data):
        prediction_indexs, entropy_indexs, entropy_threshold = data['group']['prediction_indexs'], data['group']['entropy_indexs'], data['group']['entropy_threshold']
        confidence, proportion, category_confidece, category_proportion =  data['confidence'], data['proportion'], data['category_confidece'], data['category_proportion']

        info = f'**[ {num} ] pred: {prediction_indexs} entropy: {entropy_indexs} threshold: {entropy_threshold}** \n'\
        f'| Confidence | Proportion | Category_Confi | Category_Prop |\n'\
        f'| :----: | :----: | :----: | :----: | \n'\
        f'| {confidence} | {proportion} | {category_confidece} | {category_proportion} |\n\n'
        return info

    def save_to_md(serial_number, data, mode = 'all'):
        assert mode in ['all', 'selected']
        if mode == 'all':
            with open(os.path.join(args.OutputPath, 'all_groups.md'), 'a+') as f:
                f.writelines(get_info(serial_number, data))
        else:
            with open(os.path.join(args.OutputPath, 'selected_groups.md'), 'a+') as f:
                for d in data:
                    serial_number += 1
                    f.writelines(get_info(serial_number, d))
    def save_to_bin(res, mode='all'):
        assert mode in ['all', 'selected']
        with open(os.path.join(args.OutputPath, mode + '_groups.bin'), 'wb') as f:
            pickle.dump(res, f)


    print(f"Finding the best condition group to generate pseudo labels...")
    prediction_indexs_list = [[0], [6], [0, 6], [0, 6, 3], [0, 6, 3, 1], \
        [0, 6, 3, 1, 2], [0, 6, 3, 1, 2, 4], [0, 6, 3, 1, 2, 4, 5]]
    entropy_indexs_list = [[0], [6], [0, 6], [0, 6, 3], [0, 6, 3, 1], \
        [0, 6, 3, 1, 2], [0, 6, 3, 1, 2, 4], [0, 6, 3, 1, 2, 4, 5]]
    theta = 0.19459
    entropy_threshold_list = [theta*1e2, theta*1e1, theta*1e0, theta*1e-1, theta*1e-2, theta*1e-3,\
        theta*1e-4, theta*1e-5, theta*1e-6, theta*1e-7, theta*1e-8, theta*1e-9]

    all_groups = []
    for i in range(len(prediction_indexs_list)):
        all_groups.append([])
        for j in range(len(entropy_indexs_list)):
            all_groups[i].append([])
            for k in range(len(entropy_threshold_list)):
                all_groups[i][j].append({})

    selected_groups = [] # The combination that satisfies the threshold
    serial_number = 0
    for i, pred in enumerate(prediction_indexs_list):
        for j, prop in enumerate(entropy_indexs_list):
            for k, thres in enumerate(entropy_threshold_list):
                confidence, proportion, category_confidece, category_proportion = \
                    Compute_Confidence_Proportion(args, dataloader, init_dataset_data, model, eps=1e-6, \
                        prediction_indexs=pred, entropy_indexs=prop, entropy_threshold=thres, require_data=False)
                all_groups[i][j][k] = {
                    'confidence': confidence,
                    'proportion': proportion,
                    'category_confidece': category_confidece,
                    'category_proportion': category_proportion,
                    'group': {
                        'prediction_indexs': pred,
                        'entropy_indexs': prop,
                        'entropy_threshold': thres
                    }
                }
                if confidence > confidence_threshold:
                    selected_groups.append(all_groups[i][j][k])
                serial_number += 1
                save_to_md(serial_number, all_groups[i][j][k])
                print(f"Collect {proportion:.2%} of target images with {confidence:.2%} confidence.")

    save_to_bin(all_groups)
    if len(selected_groups) == 0:
        print(f"There's no group that fits the confidence threshold.")
        return { 'prediction_indexs': [0, 1, 3, 4, 5, 6], 'entropy_indexs': [0, 1, 2, 3, 4, 5, 6], 'entropy_threshold': 0.19459 }
    else:
        selected_groups.sort(key=cmp_to_key(cmp))
        save_to_md(0, selected_groups, mode='selected')
        save_to_bin(selected_groups, mode='selected')
        best_group = selected_groups[0]['group']
        print(f"The best group: prediction_indexs={best_group['prediction_indexs']}, entropy_indexs={best_group['entropy_indexs']}, entropy_threshold={best_group['entropy_threshold']}")
        return selected_groups[0]['group']
            
def BuildLabeledDataloader(args, dataloader, init_dataset_data, model, flag='train',\
     eps=1e-6, confidence_threshold=0.95, prediction_indexs=[0], entropy_indexs=[0], entropy_threshold=0.19459):
    # best_group = Find_Best_Group_toGenerate_PseudoLabels(args, dataloader, init_dataset_data, model, confidence_threshold=confidence_threshold)
    # prediction_indexs, entropy_indexs, entropy_threshold = best_group['prediction_indexs'], best_group['entropy_indexs'], best_group['entropy_threshold']

    data_imgs, data_labels, data_bboxs, data_landmarks, collect_num, \
        confidence, proportion, category_confidece, category_proportion = \
        Compute_Confidence_Proportion(args, dataloader, init_dataset_data, model, eps=1e-6, prediction_indexs=prediction_indexs, \
            entropy_indexs=entropy_indexs, entropy_threshold=entropy_threshold, require_data=True)
    
    trans = transforms.Compose([
            transforms.Resize((args.faceScale, args.faceScale)), # 112 * 112
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    trans_strong = copy.deepcopy(trans)
    trans_strong.transforms.insert(0,  RandAugment(3, 5))
    # DataSet
    data_set = MyDataset(data_imgs, data_labels, data_bboxs, data_landmarks, flag, trans, target_transform=trans_strong) # trans是tansformer处理，data_set是一个图片集，包括裁剪后的人脸图片，图片的五个关键点，表情标签
    try:
        if flag == 'train':
            data_loader = data.DataLoader(dataset=data_set, batch_size=min(args.train_batch_size, collect_num), shuffle=True, num_workers=8, drop_last=True)
        elif flag == 'test':
            data_loader = data.DataLoader(dataset=data_set, batch_size=min(args.test_batch_size, collect_num), shuffle=False, num_workers=8, drop_last=False)
    except:
        data_loader = None

    return data_loader, confidence, proportion, category_confidece, category_proportion

def BulidAdversarialNetwork(args, model_output_num, class_num=7):
    """Bulid Adversarial Network."""

    if args.randomLayer:
        random_layer = RandomLayer([model_output_num, class_num], 1024)
        ad_net = AdversarialNetwork(1024, 512)
        random_layer.cuda()
        
    else:
        random_layer = None
        if args.methodOfDAN=='DANN':
            ad_net = AdversarialNetwork(model_output_num, 128)
        else:
            ad_net = AdversarialNetwork(model_output_num * class_num, 512)

    ad_net.cuda()

    return random_layer, ad_net

def BulidDataloader(args, flag1='train', flag2='source', need_strong_trnasform=False):
    """Bulid data loader."""
    '''
    assert的用法:assert 条件,"报错信息"
    只有满足了条件，程序才会继续往下运行，否则就报自己设置的那个错误
    '''
    assert flag1 in ['train', 'test'], 'Function BuildDataloader : function parameter flag1 wrong.'
    assert flag2 in ['source', 'target'], 'Function BuildDataloader : function parameter flag2 wrong.'

    # Set Transform
    if flag1 == 'train':
        trans = transforms.Compose([
            transforms.Resize((args.faceScale, args.faceScale)),  # 112*112
            # transforms.transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        trans = transforms.Compose([
                transforms.Resize((args.faceScale, args.faceScale)), # 112 * 112
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    target_trans = None
    if need_strong_trnasform :
        target_trans = copy.deepcopy(trans)
        target_trans.transforms.insert(0,  RandAugment(3, 5))

    # Basic Notes:
    # 0: Surprised
    # 1: Fear
    # 2: Disgust
    # 3: Happy
    # 4: Sad
    # 5: Angry
    # 6: Neutral

    dataPath_prefix = args.datasetPath

    data_imgs, data_labels, data_bboxs, data_landmarks = [], [], [], []
    if flag1 == 'train':
        if flag2 == 'source':
            if args.sourceDataset == 'RAF': # RAF Train Set 用RAF作为源训练集
                list_patition_label = pd.read_csv(os.path.join(dataPath_prefix+'/RAF/basic/EmoLabel/list_patition_label.txt'), header=None, delim_whitespace=True) # delim_whitespace设置为True代表每一行以空格分开
                list_patition_label = np.array(list_patition_label)  # 在这之后的list_patition_label的shpae是15339*2，举个例子每一个单项就是'train_3066.jpg' 2 左边是图片的名字，右边是答案标签
                for index in range(list_patition_label.shape[0]):    # 遍历15339次
                    if list_patition_label[index, 0][:5] == "train": # list_patition_label[index,0][:5]的意思==list_patition_label[index][0][0:5]
                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt'): # 去边框的文件夹中查找， 如果不存在就不用这张图片
                            continue
                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt'): # ! 为什么ladmark_5的数据是五行数字而已，具体要怎么发挥这个的作用（我觉得有可能是五个点的意思）
                            continue
                        bbox = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int) # 把他转成整型,bbox bounding box的缩写
                        landmark = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int) # 注意了，list_patiotion_label的顺序是从训练集开始的，而bounding和Landmarks_5的txt文件都是从test开始排列的

                        data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index,0]) # 把图片存储路径用数组进行保存
                        data_labels.append(list_patition_label[index, 1]-1)  # 把图片对应的表情标签也按序号进行保存
                        data_bboxs.append(bbox)                              # 把每一张图片的人脸的轮廓的矩形的坐标点记录在这个列表里边，序号对应
                        data_landmarks.append(landmark) # 将一个人脸上的五个点的位置进行存储

            elif args.sourceDataset == 'AFED': # AFED Train Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 } #! ↓↓↓
                '''
                因为AFED这个数据集有11表情标签，并且标签的顺序是跟这个代码所规定的的表情表情是不完全一致的，所以，所以需要把train_list里边的错误的标签进行重映射。
                同时因为这个train_list里边并没有2，7，8，10号标枪数据，所以不用做这个映射。
                '''
                list_patition_label = pd.read_csv(dataPath_prefix+'/AFED/Asian_Facial_Expression/AsianMovie_0725_0730/list/train_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label) # list_patition_label的shape是(32757,6)，每一唯的表示的是（图片的名字，人脸左上角的坐标，人脸右下角的坐标，表情的标签）

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys(): # 如果list中存在么有这几种标签的数据说明不符合本代码的结构，直接跳过
                        continue 

                    bbox = list_patition_label[index,1:5].astype(np.int) # 提取除人脸的框框

                    landmark = np.loadtxt(dataPath_prefix+'/AFED/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    '''
                    找到所遍历到的这幅图的五个局部特征点
                    '''
                    
                    data_imgs.append(dataPath_prefix+'/AFED/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0]) # 存图片
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]]) # 存标签
                    data_bboxs.append(bbox)  # 存人脸框框
                    data_landmarks.append(landmark) # 存五个局部特征点

            elif args.sourceDataset == 'MMI': # MMI Dataset
                
                MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                    data_labels.append(MMItoLabel[list_patition_label[index,1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
            
            elif args.sourceDataset=='FER2013': # FER2013 Train Set
                
                FER2013toLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/FER2013/list/train_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    imgPath = dataPath_prefix+'/FER2013/images/'+list_patition_label[index,0]

                    img = Image.open(imgPath).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    if not os.path.exists(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(imgPath)
                    data_labels.append(FER2013toLabel[list_patition_label[index,-1]])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h))
                    data_landmarks.append(landmark)

            if args.useMultiDatasets == 'True':

                if args.targetDataset!='CK+': # CK+ Dataset

                    for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                        Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop',expression))
                        for imgFile in Dirs:
                            imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop',expression,imgFile)
                            img = Image.open(imgPath).convert('RGB')
                            ori_img_w, ori_img_h = img.size
                            
                            if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop/landmark_5/',expression,imgFile[:-3]+'txt')):
                                continue
                            landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop/landmark_5/',expression,imgFile[:-3]+'txt')).astype(np.int)
                            
                            data_imgs.append(imgPath)
                            data_labels.append(index)
                            data_bboxs.append((0,0,ori_img_w,ori_img_h))
                            data_landmarks.append(landmark)

                    for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                        Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression))
                        for imgFile in Dirs:
                            imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression,imgFile)
                            img = Image.open(imgPath).convert('RGB')
                            ori_img_w, ori_img_h = img.size
                            
                            if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')):
                                continue
                            landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')).astype(np.int)
                            
                            data_imgs.append(imgPath)
                            data_labels.append(index)
                            data_bboxs.append((0,0,ori_img_w,ori_img_h))
                            data_landmarks.append(landmark)

                if args.targetDataset!='JAFFE': # JAFFE Dataset

                    list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                    list_patition_label = np.array(list_patition_label)

                    for index in range(list_patition_label.shape[0]):

                        if not os.path.exists(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/JAFFE/images/'+list_patition_label[index,0])
                        data_labels.append(list_patition_label[index,1])
                        data_bboxs.append(bbox) 
                        data_landmarks.append(landmark)

                if args.targetDataset!='MMI': # MMI Dataset

                    MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                    list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                    list_patition_label = np.array(list_patition_label)

                    for index in range(list_patition_label.shape[0]):

                        if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                        data_labels.append(MMItoLabel[list_patition_label[index,1]])
                        data_bboxs.append(bbox) 
                        data_landmarks.append(landmark)

                if args.targetDataset!='Oulu-CASIA': # Oulu-CASIA Dataset

                    list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                    list_patition_label = np.array(list_patition_label)

                    for index in range(list_patition_label.shape[0]):

                        if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                            continue
                        
                        img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                        ori_img_w, ori_img_h = img.size

                        landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                        data_labels.append(list_patition_label[index,1])
                        data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                        data_landmarks.append(landmark)

        elif flag2 == 'target':
            if args.targetDataset == 'CK+': # CK+ Train Set
                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop', expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop', expression, imgFile)
                        img = Image.open(imgPath).convert('RGB')
                        ori_img_w, ori_img_h = img.size # 因为已经裁剪好了，但是为了统一使用bbox参数，所以就用了原图size的尺寸
                        
                        if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop/landmark_5/', expression,imgFile[:-3]+'txt')):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Train/CK+_Train_crop/landmark_5/', expression,imgFile[:-3]+'txt')).astype(np.int)
                        
                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append((0, 0, ori_img_w, ori_img_h))
                        data_landmarks.append(landmark)

            elif args.targetDataset=='JAFFE': # JAFFE Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/JAFFE/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1]) # 存标签
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.targetDataset=='MMI': # MMI Dataset

                MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                    data_labels.append(MMItoLabel[list_patition_label[index,1]])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.targetDataset=='Oulu-CASIA': # Oulu-CASIA Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                        continue
                    
                    img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                    data_landmarks.append(landmark)

            elif args.targetDataset=='SFEW': # SFEW Train Set

                for index, expression in enumerate(['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/SFEW/Train/Annotations/Bboxs/',expression))
                    for bboxName in Dirs:
                        bboxsPath = os.path.join(dataPath_prefix+'/SFEW/Train/Annotations/Bboxs/',expression,bboxName)
                        bboxs = np.loadtxt(bboxsPath).astype(np.int)

                        if not os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Train/Annotations/Landmarks_5/',expression,bboxName)):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/SFEW/Train/Annotations/Landmarks_5/',expression,bboxName)).astype(np.int)

                        if os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Train/imgs/',expression,bboxName[:-3]+'png')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Train/imgs/',expression,bboxName[:-3]+'png')
                        elif os.path.exists(os.path.join(dataPath_prefix+'/SFEW//Train/imgs/',expression,bboxName[:-3]+'jpg')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Train/imgs/',expression,bboxName[:-3]+'jpg')
                        else:
                            print(os.path.join(dataPath_prefix+'/SFEW/Train/imgs/',expression,bboxName[:-3]+'*') + ' no exist')

                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append(bboxs)
                        data_landmarks.append(landmark)

            elif args.targetDataset=='FER2013': # FER2013 Train Set
                
                FER2013toLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/FER2013/list/train_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    imgPath = dataPath_prefix+'/FER2013/images/'+list_patition_label[index,0]

                    img = Image.open(imgPath).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    if not os.path.exists(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(imgPath)
                    data_labels.append(FER2013toLabel[list_patition_label[index,-1]])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h))
                    data_landmarks.append(landmark)

            elif args.targetDataset=='ExpW': # ExpW Train Set
                
                ExpWtoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/ExpW/list/Landmarks_5/train_list_5landmarks.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,2:6].astype(np.int)
                    landmark = np.array(list_patition_label[index,7:]).astype(np.int).reshape(-1,2)
                    
                    data_imgs.append(dataPath_prefix+'/ExpW/data/image/origin/'+list_patition_label[index,0])
                    data_labels.append(ExpWtoLabel[list_patition_label[index,6]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
            
            elif args.targetDataset=='AFED': # AFED Train Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/list/train_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)  
                    data_landmarks.append(landmark)

            elif args.targetDataset=='WFED': # WFED Train Set

                WesternToLabel = { 2:0, 5:1, 4:2, 1:3, 3:4, 6:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Western_Films_Expression_Datasets/list/train_random.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    
                    if not os.path.exists(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt').astype(np.int)
                    
                    imgPath = dataPath_prefix+'/Western_Films_Expression_Datasets/images/'+list_patition_label[index,0]
                    if os.path.exists(imgPath+'.png'):
                        data_imgs.append(imgPath+'.png')
                    elif os.path.exists(imgPath+'.jpg'):
                        data_imgs.append(imgPath+'.jpg')
                    else:
                        continue

                    data_labels.append(WesternToLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.targetDataset == 'RAF': # RAF Train Set

                list_patition_label = pd.read_csv(dataPath_prefix+'/RAF/basic/EmoLabel/list_patition_label.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index,0][:5] == "train":

                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index,0])
                        data_labels.append(list_patition_label[index,1]-1)
                        data_bboxs.append(bbox)
                        data_landmarks.append(landmark)

    elif flag1 == 'test':
        if flag2 =='source':
            if args.sourceDataset=='CK+': # CK+ Val Set
                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop',expression,imgFile)
                        img = Image.open(imgPath).convert('RGB')
                        ori_img_w, ori_img_h = img.size
                        
                        if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')).astype(np.int)
                        
                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append((0,0,ori_img_w,ori_img_h))
                        data_landmarks.append(landmark)

            elif args.sourceDataset=='JAFFE': # JAFFE Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/JAFFE/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='MMI': # MMI Dataset

                MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                    data_labels.append(MMItoLabel[list_patition_label[index,1]])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='Oulu-CASIA': # Oulu-CASIA Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                        continue
                    
                    img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='SFEW': # SFEW 2.0 Val Set

                for index, expression in enumerate(['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Bboxs/',expression))
                    for bboxName in Dirs:
                        bboxsPath = os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Bboxs/',expression,bboxName)
                        bboxs = np.loadtxt(bboxsPath).astype(np.int)

                        if not os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Landmarks_5/',expression,bboxName)):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Landmarks_5/',expression,bboxName)).astype(np.int)

                        if os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'png')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'png')
                        elif os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'jpg')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'jpg')
                        else:
                            print(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'*') + ' no exist')

                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append(bboxs)
                        data_landmarks.append(landmark)

            elif args.sourceDataset=='FER2013': # FER2013 Val Set

                FER2013toLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/FER2013/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    imgPath = dataPath_prefix+'/FER2013/images/'+list_patition_label[index,0]

                    img = Image.open(imgPath).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    if not os.path.exists(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(imgPath)
                    data_labels.append(FER2013toLabel[list_patition_label[index,-1]])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h))
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='ExpW': # ExpW Val Set

                ExpWtoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/ExpW/list/Landmarks_5/val_list_5landmarks.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,2:6].astype(np.int)
                    landmark = np.array(list_patition_label[index,7:]).astype(np.int).reshape(-1,2)
                    
                    data_imgs.append(dataPath_prefix+'/ExpW/data/image/origin/'+list_patition_label[index,0])
                    data_labels.append(ExpWtoLabel[list_patition_label[index,6]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
           
            elif args.sourceDataset=='AFED': # AFED Val Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/AFED/Asian_Facial_Expression/AsianMovie_0725_0730/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/AFED/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/AFED/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='WFED': # WFED Val Set

                WesternToLabel = { 2:0, 5:1, 4:2, 1:3, 3:4, 6:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Western_Films_Expression_Datasets/list/val_random.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    
                    if not os.path.exists(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt').astype(np.int)
                    
                    imgPath = dataPath_prefix+'/Western_Films_Expression_Datasets/images/'+list_patition_label[index,0]
                    if os.path.exists(imgPath+'.png'):
                        data_imgs.append(imgPath+'.png')
                    elif os.path.exists(imgPath+'.jpg'):
                        data_imgs.append(imgPath+'.jpg')
                    else:
                        continue

                    data_labels.append(WesternToLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.sourceDataset=='RAF': # RAF Test Set
                list_patition_label = pd.read_csv(dataPath_prefix+'/RAF/basic/EmoLabel/list_patition_label.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index, 0][:4] == "test":

                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index, 0])
                        data_labels.append(list_patition_label[index, 1] - 1)
                        data_bboxs.append(bbox)
                        data_landmarks.append(landmark)

        elif flag2 == 'target':
            if args.targetDataset == 'CK+': # CK+ Val Set

                for index, expression in enumerate(['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop', expression))
                    for imgFile in Dirs:
                        imgPath = os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop', expression, imgFile)
                        img = Image.open(imgPath).convert('RGB')
                        ori_img_w, ori_img_h = img.size
                        
                        if not os.path.exists(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/CK+_Emotion/Val/CK+_Val_crop/landmark_5/',expression,imgFile[:-3]+'txt')).astype(np.int)
                        
                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append((0, 0, ori_img_w, ori_img_h))
                        data_landmarks.append(landmark)

            elif args.targetDataset=='JAFFE': # JAFFE Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/JAFFE/list/list_putao.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/JAFFE/annos/bbox/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/JAFFE/annos/landmark_5/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/JAFFE/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index, 1])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.targetDataset=='MMI': # MMI Dataset

                MMItoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/MMI/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    if not os.path.exists(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue

                    bbox = np.loadtxt(dataPath_prefix+'/MMI/annos/bbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/MMI/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/MMI/images/'+list_patition_label[index,0])
                    data_labels.append(MMItoLabel[list_patition_label[index,1]])
                    data_bboxs.append(bbox) 
                    data_landmarks.append(landmark)

            elif args.targetDataset=='Oulu-CASIA': # Oulu-CASIA Dataset

                list_patition_label = pd.read_csv(dataPath_prefix+'/Oulu-CASIA/list/list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt'): 
                        continue
                    
                    img = Image.open(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0]).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    landmark = np.loadtxt(dataPath_prefix+'/Oulu-CASIA/annos/landmark_5/VL_Acropped/Strong/'+list_patition_label[index,0][:-4]+'txt').astype(np.int)

                    data_imgs.append(dataPath_prefix+'/Oulu-CASIA/images/'+list_patition_label[index,0])
                    data_labels.append(list_patition_label[index,1])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h)) 
                    data_landmarks.append(landmark)

            elif args.targetDataset=='SFEW': # SFEW 2.0 Val Set

                for index, expression in enumerate(['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral']):
                    Dirs = os.listdir(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Bboxs/',expression))
                    for bboxName in Dirs:
                        bboxsPath = os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Bboxs/',expression,bboxName)
                        bboxs = np.loadtxt(bboxsPath).astype(np.int)

                        if not os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Landmarks_5/',expression,bboxName)):
                            continue
                        landmark = np.loadtxt(os.path.join(dataPath_prefix+'/SFEW/Val/Annotations/Landmarks_5/',expression,bboxName)).astype(np.int)

                        if os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'png')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'png')
                        elif os.path.exists(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'jpg')):
                            imgPath = os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'jpg')
                        else:
                            print(os.path.join(dataPath_prefix+'/SFEW/Val/imgs/',expression,bboxName[:-3]+'*') + ' no exist')

                        data_imgs.append(imgPath)
                        data_labels.append(index)
                        data_bboxs.append(bboxs)
                        data_landmarks.append(landmark)

            elif args.targetDataset=='FER2013': # FER2013 Val Set

                FER2013toLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/FER2013/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    imgPath = dataPath_prefix+'/FER2013/images/'+list_patition_label[index,0]

                    img = Image.open(imgPath).convert('RGB')
                    ori_img_w, ori_img_h = img.size

                    if not os.path.exists(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/FER2013/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(imgPath)
                    data_labels.append(FER2013toLabel[list_patition_label[index,-1]])
                    data_bboxs.append((0,0,ori_img_w,ori_img_h))
                    data_landmarks.append(landmark)

            elif args.targetDataset=='ExpW': # ExpW Val Set

                ExpWtoLabel = { 5:0, 2:1, 1:2, 3:3, 4:4, 0:5, 6:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/ExpW/list/Landmarks_5/val_list_5landmarks.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,2:6].astype(np.int)
                    landmark = np.array(list_patition_label[index,7:]).astype(np.int).reshape(-1,2)
                    
                    data_imgs.append(dataPath_prefix+'/ExpW/data/image/origin/'+list_patition_label[index,0])
                    data_labels.append(ExpWtoLabel[list_patition_label[index,6]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)
           
            elif args.targetDataset=='AFED': # AFED Val Set

                AsiantoLabel = { 3:0, 6:1, 5:2, 1:3, 4:4, 9:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/list/val_list.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if list_patition_label[index,-1] not in AsiantoLabel.keys():
                        continue

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    landmark = np.loadtxt(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/annos/landmark_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                    
                    data_imgs.append(dataPath_prefix+'/Asian_Facial_Expression/AsianMovie_0725_0730/images/'+list_patition_label[index,0])
                    data_labels.append(AsiantoLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.targetDataset=='WFED': # WFED Val Set

                WesternToLabel = { 2:0, 5:1, 4:2, 1:3, 3:4, 6:5, 0:6 }
                list_patition_label = pd.read_csv(dataPath_prefix+'/Western_Films_Expression_Datasets/list/val_random.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    bbox = list_patition_label[index,1:5].astype(np.int)
                    
                    if not os.path.exists(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt'):
                        continue
                    landmark = np.loadtxt(dataPath_prefix+'/Western_Films_Expression_Datasets/annos/5_landmarks/'+list_patition_label[index,0]+'.txt').astype(np.int)
                    
                    imgPath = dataPath_prefix+'/Western_Films_Expression_Datasets/images/'+list_patition_label[index,0]
                    if os.path.exists(imgPath+'.png'):
                        data_imgs.append(imgPath+'.png')
                    elif os.path.exists(imgPath+'.jpg'):
                        data_imgs.append(imgPath+'.jpg')
                    else:
                        continue

                    data_labels.append(WesternToLabel[list_patition_label[index,-1]])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

            elif args.targetDataset=='RAF': # RAF Test Set

                list_patition_label = pd.read_csv(dataPath_prefix+'/RAF/basic/EmoLabel/list_patition_label.txt', header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index,0][:4] == "test":

                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt'):
                            continue
                        if not os.path.exists(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt'):
                            continue

                        bbox = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/boundingbox/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)
                        landmark = np.loadtxt(dataPath_prefix+'/RAF/basic/Annotation/Landmarks_5/'+list_patition_label[index,0][:-3]+'txt').astype(np.int)

                        data_imgs.append(dataPath_prefix+'/RAF/basic/Image/original/'+list_patition_label[index,0])
                        data_labels.append(list_patition_label[index,1]-1)
                        data_bboxs.append(bbox)
                        data_landmarks.append(landmark)

    # DataSet Distribute
    distribute_ = np.array(data_labels) # 存图片的标签的
    print('The %s %s dataset quantity: %d' % (flag1, flag2, len(data_imgs)))
    print('The %s %s dataset distribute: %d, %d, %d, %d, %d, %d, %d' % (flag1, flag2,
           np.sum(distribute_ == 0), np.sum(distribute_== 1), np.sum(distribute_== 2), np.sum(distribute_== 3),
           np.sum(distribute_ == 4), np.sum(distribute_== 5), np.sum(distribute_== 6))) # 这里整个数据集中，7种类别的表情的数量

    # DataSet
    data_set = MyDataset(data_imgs, data_labels, data_bboxs, data_landmarks, flag1, trans, target_trans) # trans是tansformer处理，data_set是一个图片集，包括裁剪后的人脸图片，图片的五个关键点，表情标签
    # DataLoader
    if flag1 == 'train':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.train_batch_size, shuffle=True, num_workers=8, drop_last=True)
    elif flag1 == 'test':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.test_batch_size, shuffle=False, num_workers=8, drop_last=False)

    return data_loader, data_set, {'imgs_list':data_imgs, 'labels_list':data_labels, 'bboxs_list':data_bboxs, 'landmarks_list':data_landmarks, 'transform':trans}, [np.sum(distribute_ == i) for i in range(7)] # 第三个返回值是目标域的数量的distribution


def Build_Features_Pool(args, epoch, model, train_source_dataloader, labeled_train_target_loader, sim_matrics, collect_num=10):
    print(f"========================================================")
    print(f"building the features pool...")

    #@ Build the source domain features pool
    source_features_pool = Collect_Domain_Class_Features(args, model, train_source_dataloader, 'source', collect_num)
    base_index = Find_the_Most_Class(source_features_pool)  # record the most class of the pool constructed once
    for class_index in range(args.class_num):
        if source_features_pool[str(class_index)].size()[0] < collect_num:
            remaining = collect_num - source_features_pool[str(class_index)].size()[0]

            base_features = Collect_Specific_Class_Features(args, model, train_source_dataloader, \
                'source', base_index, remaining)
            aim_features = Collect_Specific_Class_Features(args, model, train_source_dataloader, \
                'source', class_index, remaining)
            
            if base_index == class_index:
                fused_target_features = aim_features
            else:
                fused_target_features =  Fuse_Class_Feature(args, base_features, aim_features, sim_matrics[class_index][base_index]) # aim_features + sim * base_features
            source_features_pool[str(class_index)] = torch.cat((source_features_pool[str(class_index)], fused_target_features), dim=0)
    print(f"source pool finished")

    #@ Build the target domain features pool
    target_features_pool = Collect_Domain_Class_Features(args, model, labeled_train_target_loader, 'labeled-target', collect_num)
    base_index = Find_the_Most_Class(target_features_pool)  # record the most class of the pool constructed once
    for class_index in range(args.class_num):
        if target_features_pool[str(class_index)].size()[0] < collect_num:
            remaining = collect_num - target_features_pool[str(class_index)].size()[0]
            
            #@@ Note that this is different from the source pool generation
            base_features = Collect_Specific_Class_Features(args, model, train_source_dataloader, \
                'source', class_index, remaining)
            aim_features = Collect_Specific_Class_Features(args, model, labeled_train_target_loader,\
                'labeled-target', base_index,  remaining)

            if base_index == class_index:
                fused_target_features = aim_features
            else:
                fused_target_features =  Fuse_Class_Feature(args, base_features, aim_features, sim_matrics[class_index][base_index]) # aim_features + sim * base_features

            target_features_pool[str(class_index)] = torch.cat((target_features_pool[str(class_index)], fused_target_features), dim=0)
    
    print(f"target pool finished")
    print(f"========================================================")
    
    return {'source': source_features_pool, 'target': target_features_pool}

def Show_Accuracy(acc, prec, recall, class_num=7):  # AGRA 计算acc,recall,precision,F1的方法
    """Compute average of accuaracy/precision/recall/f1"""

    # Compute F1 value    
    f1 = [AverageMeter() for i in range(class_num)]
    for i in range(class_num):
        if prec[i].avg == 0 or recall[i].avg == 0:
            f1[i].avg = 0
            continue
        f1[i].avg = 2*prec[i].avg*recall[i].avg/(prec[i].avg+recall[i].avg)
    
    # Compute average of accuaracy/precision/recall/f1
    acc_avg, prec_avg, recall_avg, f1_avg = 0, 0, 0, 0

    for i in range(class_num):
        acc_avg+=acc[i].avg
        prec_avg+=prec[i].avg
        recall_avg+=recall[i].avg
        f1_avg+=f1[i].avg

    acc_avg, prec_avg, recall_avg, f1_avg = acc_avg/class_num, prec_avg/class_num, recall_avg/class_num, f1_avg/class_num

    # Log Accuracy Infomation
    Accuracy_Info = ''
    
    Accuracy_Info+='    Accuracy'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(acc[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    Precision'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(prec[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    Recall'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(recall[i].avg)
    Accuracy_Info+='\n'

    Accuracy_Info+='    F1'
    for i in range(class_num):
        Accuracy_Info+=' {:.4f}'.format(f1[i].avg)
    Accuracy_Info+='\n'

    return Accuracy_Info, acc_avg, prec_avg, recall_avg, f1_avg

def Show_OnlyAccuracy(accuracys):
    accs = [0 for i in range(len(accuracys))] # 预设7个分类器的分类准确率为0
    for classifier_id in range(len(accuracys)):
        for class_id in range(len(accuracys[0])):
            accs[classifier_id] += accuracys[classifier_id][class_id].avg
    return [acc / len(accuracys[0]) for acc in accs]

def Show_OtherMetrics(prec, recall, dist): # 计算 weighted precision, weighted recall, F1 Score
    '''
    #@ prec: 七个分类器, 每个分类器存储七个类别的precision
    #@ recall: 七个分类器, 每个分类器存储七个类别的precision
    #@ dist: 七个表情类别数据的数量
    '''
    total = sum(dist)
    dist_norm = [1 / 7 for i in range(len(dist))]
    # dist_norm = [dist[i] / total for i in range(len(dist))]
    weighted_precs, weighted_recalls, f1s = [0 for i in range(len(prec))], [0 for i in range(len(recall))], [0 for i in range(len(recall))]
    for classifier_id in range(len(prec)):
        tmp_prec, tmp_recall, tmp_f1 = 0, 0, 0
        for class_id in range(len(prec[0])):
            tmp_prec += dist_norm[class_id] * prec[classifier_id][class_id].avg
            tmp_recall += dist_norm[class_id] * recall[classifier_id][class_id].avg
            if prec[classifier_id][class_id].avg == 0 or recall[classifier_id][class_id].avg == 0:
                tmp_f1 += 0
            else:
                tmp_f1 += dist_norm[class_id] * (2 * prec[classifier_id][class_id].avg * recall[classifier_id][class_id].avg / (prec[classifier_id][class_id].avg + recall[classifier_id][class_id].avg))
        weighted_precs[classifier_id] = tmp_prec
        weighted_recalls[classifier_id] = tmp_recall
        f1s[classifier_id] = tmp_f1
    return weighted_precs, weighted_recalls, f1s

def Draw_Category_Metrics_Bar(prec, recall):
    prec_data = [['Surprised'], ['Fear'], ['Disgust'], ['Happy'], ['Sad'], ['Angry'], ['Neutral']]
    recall_data = [['Surprised'], ['Fear'], ['Disgust'], ['Happy'], ['Sad'], ['Angry'], ['Neutral']]
    f1_data = [['Surprised'], ['Fear'], ['Disgust'], ['Happy'], ['Sad'], ['Angry'], ['Neutral']]
    for class_id in range(len(prec[0])):
        for classifier_id in range(len(prec)):
            prec_data[class_id].append(prec[classifier_id][class_id].avg)
            recall_data[class_id].append(recall[classifier_id][class_id].avg)
            if prec[classifier_id][class_id].avg == 0 or recall[classifier_id][class_id].avg == 0:
                f1_data[class_id].append(0)
            else:    
                f1_data[class_id].append(2 * prec[classifier_id][class_id].avg * recall[classifier_id][class_id].avg / ( prec[classifier_id][class_id].avg + recall[classifier_id][class_id].avg))
    
    fig = plt.figure()
    ax_prec = fig.add_subplot(3, 1, 1)
    ax_recall = fig.add_subplot(3, 1, 2)
    ax_f1 = fig.add_subplot(3, 1, 3)

    df=pd.DataFrame(prec_data,columns=["Expression", "Global","L-Eye","R-Eye","Nose", "L-Mouse", "R-Mouse", "G-Local"])
    df.plot(x="Expression", y=["Global","L-Eye","R-Eye","Nose", "L-Mouse", "R-Mouse", "G-Local"], kind="bar",figsize=(9,8), ax=ax_prec, title='precision', grid=True, sharex=True)
    
    df=pd.DataFrame(recall_data,columns=["Expression", "Global","L-Eye","R-Eye","Nose", "L-Mouse", "R-Mouse", "G-Local"])
    ax_recall = df.plot(x="Expression", y=["Global","L-Eye","R-Eye","Nose", "L-Mouse", "R-Mouse", "G-Local"], kind="bar",figsize=(9,8), ax=ax_recall, title='recall', grid=True)
    
    df=pd.DataFrame(f1_data,columns=["Expression", "Global","L-Eye","R-Eye","Nose", "L-Mouse", "R-Mouse", "G-Local"])
    ax_f1 = df.plot(x="Expression", y=["Global","L-Eye","R-Eye","Nose", "L-Mouse", "R-Mouse", "G-Local"], kind="bar",figsize=(9,8), ax=ax_f1, title='f1-score', grid=True, rot=360)
    
    return fig

def Draw_Dataset_Distribution_Bar(source_name, train_source, test_source, target_name, train_target, test_target, mode='prop'):
    assert mode in ['proportion', 'quantity']
    if mode == 'proportion':
        train_source_ = [train_source[i]/sum(train_source) for i in range(len(train_source))]
        test_source_ = [test_source[i]/sum(test_source) for i in range(len(test_source))]
        train_target_ = [train_target[i]/sum(train_target) for i in range(len(train_target))]
        test_target_ = [test_target[i]/sum(test_target) for i in range(len(test_target))]
    elif mode == 'quantity':
        train_source_ = train_source
        test_source_ = test_source
        train_target_ = train_target
        test_target_ = test_target

    source_data = [['Surprised'], ['Fear'], ['Disgust'], ['Happy'], ['Sad'], ['Angry'], ['Neutral']]
    target_data = [['Surprised'], ['Fear'], ['Disgust'], ['Happy'], ['Sad'], ['Angry'], ['Neutral']]
    for class_id in range(len(train_source)):
        source_data[class_id].append(train_source_[class_id])
        source_data[class_id].append(test_source_[class_id])
        target_data[class_id].append(train_target_[class_id])
        target_data[class_id].append(test_target_[class_id])

    fig = plt.figure()
    ax_source = fig.add_subplot(2, 1, 1)
    ax_target = fig.add_subplot(2, 1, 2)

    df = pd.DataFrame(source_data,columns=["Expression", "Train","Test"])
    df.plot(x="Expression", y=["Train", "Test"], kind="bar",figsize=(9,8), ax=ax_source, title='source distribution'+f'({source_name})', grid=True, sharex=True)
    df = pd.DataFrame(target_data,columns=["Expression", "Train","Test"])
    df.plot(x="Expression", y=["Train", "Test"], kind="bar",figsize=(9,8), ax=ax_target, title='target distribution'+f'({target_name})', grid=True, sharex=True, rot=360)

    return fig, train_source_, test_source_, train_target_, test_target_

def Draw_PL_Distribute_Bar(category_confidence, category_proportion, domain, xaxis=['Surprised', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']):
    fig, ax = plt.subplots()
    
    data = []
    for index in range(len(category_confidence)):
        data.append([])
        data[index].append(xaxis[index])
        data[index].append(category_confidence[index])
        data[index].append(category_proportion[index])

    df = pd.DataFrame(data, columns=["Expression", "Confidence", "Proportion"])
    df.plot(x="Expression", y=["Confidence", "Proportion"], kind="bar",figsize=(9,8), ax=ax, title='Category Info'+f'({domain})', grid=True, sharex=True, rot=360)

    return fig

def Momentum_Update(encoder_q, encoder_k, keep_ratio): # 用encoder_q的参数去更新encoder_k
    """
    update momentum encoder
    """
    for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
        param_k.data = param_k.data * keep_ratio + param_q.data * (1. - keep_ratio)

def Get_Best_Name_Value(metric, mode='percent', name_dic={0:'global', 1:'left_eye', 2:'right_eye', 3:'nose', 4:'left_mouth', 5:'right_mouth', 6:'global_local'}):
    assert mode in ['decimal', 'percent']
    if len(metric) > 0:
        best_index = metric.index(max(metric))
        best_name = name_dic[best_index]
        if mode == 'percent':
            best_value = f'{metric[best_index]:.2%}'
        else:
            best_value = metric[best_index]
        return (best_name, best_value)
    else:
        return '/'

def Update_Best_Metrics(Best_Metrics, accs, precs, recalls, f1s, epoch, state='Train', domain='Source'):
    assert state in ['Train', 'Test']
    assert domain in ['Source', 'Target']

    acc_best_name, acc_best_value = Get_Best_Name_Value(accs, 'decimal')
    prec_best_name, prec_best_value = Get_Best_Name_Value(precs, 'decimal')
    recall_best_name, recall_best_value = Get_Best_Name_Value(recalls, 'decimal')
    f1_best_name, f1_best_value = Get_Best_Name_Value(f1s, 'decimal')

    modified_metrics = 0
    key = state + '_' + domain
    if Best_Metrics["acc"][key][0] < acc_best_value:
        modified_metrics += 1
        Best_Metrics['acc'][key] = (acc_best_value, epoch)
    if Best_Metrics["prec"][key][0] < prec_best_value:
        modified_metrics += 1
        Best_Metrics['prec'][key] = (prec_best_value, epoch)
    if Best_Metrics["recall"][key][0] < recall_best_value:
        modified_metrics += 1
        Best_Metrics['recall'][key] = (recall_best_value, epoch)
    if Best_Metrics["f1"][key][0] < f1_best_value:
        modified_metrics += 1
        Best_Metrics['f1'][key] = (f1_best_value, epoch)

    return modified_metrics

def Find_the_Most_Class(features_pool):
    most_class_index = 0
    most_class_number = features_pool[str(0)].size()[0]
    for class_index in range(1, 7):
        if features_pool[str(class_index)].size()[0] > most_class_number:
            most_class_index = class_index
            most_class_number = features_pool[str(class_index)].size()[0]
    return most_class_index

def PCA_svd(X, k, center=True):
  n = X.size()[0]
  ones = torch.ones(n).view([n,1])
  h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
  H = torch.eye(n) - h
  H = H.cuda()
  X_center =  torch.mm(H.double(), X.double())
  u, s, v = torch.svd(X_center)
  components  = v[:k].t()
  #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
  return components

def Collect_Domain_Class_Features(args, model, dataloader, domain, collect_num=100, class_dic=['Surprised', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']):
    model.eval()
    f_type = args.f_type
    reach_num = 0 # the number of category whose features have been collected enough
    current_class_number = { '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0 }
    current_class_features = { '0': torch.tensor([]), '1': torch.tensor([]), '2': torch.tensor([]), '3': torch.tensor([]), '4': torch.tensor([]), '5': torch.tensor([]), '6': torch.tensor([]) }
    iter_dataloader = iter(dataloader)
    num_iter = len(dataloader)
    bar = tqdm(range(num_iter))
    domain_ = args.sourceDataset if domain == 'source' else args.targetDataset
    for step, batch_index in enumerate(bar):
        if reach_num == 7:  # Successful collection completed
            break
        if domain == 'source':
            _, data, landmark, label = next(iter_dataloader)
        elif domain == 'labeled-target':
            _, data, _, landmark, label = next(iter_dataloader)

        label_np = label.cpu().numpy()
        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()
        
        for class_index in range(7):
            indexs = np.argwhere(label_np == class_index).squeeze(axis=1)
            if current_class_number[str(class_index)] == collect_num:
                continue
            elif len(indexs) == 0:
                continue
            elif current_class_number[str(class_index)] + len(indexs) >= collect_num:
                remain_space = collect_num - current_class_number[str(class_index)]
                indexs = indexs[:remain_space]
                reach_num += 1

            indexs = torch.tensor(indexs).cuda()
            data_class = data[indexs]
            if f_type == 'feature':
                with torch.no_grad():
                    features_class_, preds = model(data_class, landmark) # forward propagation
                    features_class = features_class_[6]
            elif f_type == 'raw_img':
                features_class = torch.reshape(data_class, (data_class.size(0), -1))
                # @ Dimensionality reduction using pca
                # features_class = PCA_svd(features_class, 37632)
            elif f_type == 'layer_feature':
                with torch.no_grad():
                    _, _, layer_features = model(data_class, landmark, return_layer_features=True) # forward propagation
                    features_class = torch.reshape(layer_features[4], (layer_features[4].size(0), -1))
            
            current_class_features[str(class_index)] = \
                torch.cat((current_class_features[str(class_index)], features_class.cpu()), dim=0)
            current_class_number[str(class_index)] += len(indexs)

        bar.desc = f'[Collecting {domain_} features] ({class_dic[0]}, {current_class_number[str(0)]})({class_dic[1]}, {current_class_number[str(1)]})'\
            f'({class_dic[2]}, {current_class_number[str(2)]})({class_dic[3]}, {current_class_number[str(3)]})({class_dic[4]}, {current_class_number[str(4)]})'\
            f'({class_dic[5]}, {current_class_number[str(5)]})({class_dic[6]}, {current_class_number[str(6)]})'           
    
    return current_class_features

def Collect_Specific_Class_Features(args, model, dataloader, domain, aim_class, collect_num):
    '''
        dataloader: the data generator of source / target
        base_features: the base feature group used to fuse to generate the target category
        target_class: The class of the generated target feature set to be fused
        collect_num: the number of target category features that need to be generated
    '''
    domain_ = args.sourceDataset if domain == 'source' else args.targetDataset
    print(f'Collecting {domain_} features of Class {aim_class}...')
    model.eval()
    f_type = args.f_type
    current_features = torch.tensor([])
    iter_dataloader = iter(dataloader)
    while(current_features.size()[0] < collect_num):
        try:
            if domain == 'source':
                _, data, landmark, label = next(iter_dataloader)
            elif domain == 'labeled-target':
                _, data, _, landmark, label = next(iter_dataloader)
        except:
            iter_dataloader = iter(dataloader)
            if domain == 'source':
                _, data, landmark, label = next(iter_dataloader)
            elif domain == 'labeled-target':
                _, data, _, landmark, label = next(iter_dataloader)

        label_np = label.cpu().numpy()
        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()
        indexs = np.argwhere(label_np == aim_class).squeeze(axis=1)
        if len(indexs) == 0:
            continue
        if current_features.size()[0] + len(indexs) >= collect_num:
            remain_space = collect_num - current_features.size()[0]
            indexs = indexs[:remain_space]
        indexs = torch.tensor(indexs).cuda()
        data_class = data[indexs]

        if f_type == 'feature':
            with torch.no_grad():
                features_class_, preds = model(data_class, landmark) # forward propagation
                features_class = features_class_[6]
        elif f_type == 'raw_img':
            features_class = torch.reshape(data_class, (data_class.size(0), -1))
        elif f_type == 'layer_feature':
            with torch.no_grad():
                _, _, layer_features = model(data_class, landmark, return_layer_features=True) # forward propagation
                features_class = torch.reshape(layer_features[4], (layer_features[4].size(0), -1))        

        current_features = torch.cat((current_features, features_class.cpu()), dim=0)

    print(f"done!")
    return current_features
    
def Fuse_Class_Feature(args, base_features, aim_features, domain_distancs, cross_distance=1, fuse_type='in-domain'):
    assert fuse_type in ['in-domain', 'cross-domain']
    if fuse_type == 'in-domain':
        return aim_features + domain_distancs * base_features
    else:
        fuse_type == 'cross-domain'
        return aim_features + domain_distancs * cross_distance * base_features

def Cal_Class_Similarity_in_Domain(args, model, dataloader, domain, collect_num):
    class_features = Collect_Domain_Class_Features(args, model, dataloader, domain, collect_num)
    Similarity_Matrics = np.ones((7,7))
    for i in range(7):
        for j in range(i+1, 7):
            sum_sim = 0.
            for feature_index in range(min(len(class_features[str(i)]), len(class_features[str(j)]))):
                try:
                    sum_sim += Calculate_2Features_Similarity(class_features[str(i)][feature_index], class_features[str(j)][feature_index])
                except:
                    print(f"feature_index = {feature_index}")
                    print(f"len(class_features[str(i)]) = {len(class_features[str(i)])}")
                    print(f"len(class_features[str(j)]) = {len(class_features[str(j)])}")
            sum_sim /= len(class_features[str(i)])
            Similarity_Matrics[i][j] = Similarity_Matrics[j][i] = sum_sim
    return Similarity_Matrics

def Calculate_2Features_Similarity(feat_a, feat_b):

    # The dimension of the expansion vector is at least two dimensions
    if len(feat_a.shape) == 1:
        feat_a = feat_a.unsqueeze(0)
    if len(feat_b.shape) == 1:
        feat_b = feat_b.unsqueeze(0)
    
    feat_a_norm =F.normalize(feat_a, dim=-1)
    feat_b_norm =F.normalize(feat_b, dim=-1)
    return feat_a_norm.mm(feat_b_norm.t())

def Calculate_Focal_Loss(predictions, labels):
    '''
        predictions: N × C
        labels: N
        fomulation: loss = -(1-p_y)log(p_y)
    '''
    
    loss_ = 0
    sum_weight = 0
    for idx in range(len(labels)):

        # loss of each instance
        temp = torch.exp(predictions[idx]).sum()
        temp = -predictions[idx][labels[idx]] + torch.log(temp)
        print("loss{}".format(idx), temp)

        loss_ += temp * weight_CE[labels[idx]]
        sum_weight += weight_CE[labels[idx]]

    loss_ /= sum_weight
    print("sum loss", loss_)

    weight_CE = torch.FloatTensor([1 / i for i in weight_CE])
    print(f"weight_CE = {weight_CE}")

def Write_Results_Log(Best_Metrics, accs_source, accs_target, precs_source, precs_target, \
    recalls_source, recalls_target, f1s_source, f1s_target, epoch, args, mode='Train'):

    assert mode in ['Train', 'Test']
    if mode == 'Train':
        source = 'Train_Source'
        target = 'Train_Target'
    else:
        source = 'Test_Source'
        target = 'Test_Target'

    EpochInfo = f'**{mode} Epoch {epoch}:** Learning Rate {args.lr}  DAN Learning Rate {args.lr_ad}\n'\
    f'| Metric \ Domain | Source | Target |\n'\
    f'| :----: | :----: | :----: |\n'\
    f'| **Accuracy** | {Get_Best_Name_Value(accs_source)} | {Get_Best_Name_Value(accs_target)} |\n'\
    f'| **Precision** | {Get_Best_Name_Value(precs_source)} | {Get_Best_Name_Value(precs_target)} |\n'\
    f'| **Recall** | {Get_Best_Name_Value(recalls_source)} | {Get_Best_Name_Value(recalls_target)} |\n'\
    f'| **F1-Score** | {Get_Best_Name_Value(f1s_source)} | {Get_Best_Name_Value(f1s_target)} |\n\n'\

    HistoryInfo = f'==Current Best Metircs==\n'\
    f'| Metric \ Domain | Source | Target |\n'\
    f'| :----: | :----: | :----: |\n'\
    f'| **Accuracy** | ({Best_Metrics["acc"][source][0]:.2%}, epoch {Best_Metrics["acc"][source][1]}) | ({Best_Metrics["acc"][target][0]:.2%}, epoch {Best_Metrics["acc"][target][1]}) |\n'\
    f'| **Precision** | ({Best_Metrics["prec"][source][0]:.2%}, epoch {Best_Metrics["prec"][source][1]}) | ({Best_Metrics["prec"][target][0]:.2%}, epoch {Best_Metrics["prec"][target][1]}) |\n'\
    f'| **Recall** | ({Best_Metrics["recall"][source][0]:.2%}, epoch {Best_Metrics["recall"][source][1]}) | ({Best_Metrics["recall"][target][0]:.2%}, epoch {Best_Metrics["recall"][target][1]}) |\n'\
    f'| **F1-Score** | ({Best_Metrics["f1"][source][0]:.2%}, epoch {Best_Metrics["f1"][source][1]}) | ({Best_Metrics["f1"][target][0]:.2%}, epoch {Best_Metrics["f1"][target][1]}) |\n\n'\
    
    return EpochInfo, HistoryInfo

def Visualize_Transfer_OnlyinTraining_Result(epoch, writer, dan_accs, dan_loss, acc_dic, pro_dic, hin_loss):
    #@ C-DAN (Only in the Trainning State)
    writer.add_scalars('CDAN/Acc', {'global':dan_accs[0], 'left_eye':dan_accs[1], 'right_eye':dan_accs[2], 'nose':dan_accs[3], 'left_mouse':dan_accs[4], 'right_mouse':dan_accs[5], 'global_local':dan_accs[6]}, epoch)
    writer.add_scalars('CDAN/Loss', {'global':dan_loss[0].avg, 'left_eye':dan_loss[1].avg, 'right_eye':dan_loss[2].avg, 'nose':dan_loss[3].avg, 'left_mouse':dan_loss[4].avg, 'right_mouse':dan_loss[5].avg, 'global_local':dan_loss[6].avg}, epoch)
    #@ Hyperparameter - Entropy Threshold for pseudo labels generation
    writer.add_scalars('EnthropyThreshold/Accuracys', acc_dic, epoch)
    writer.add_scalars('EnthropyThreshold/Probility', pro_dic, epoch)
    #@ Hinge Loss
    writer.add_scalar('HingeLoss/train_target', hin_loss.avg, epoch)

def Visualize_Transfer_Common_Result(epoch, writer, accs, precs, recalls, f1s, ce_loss,\
     six_confidences, six_probilities, state='Train', domain='Source'):

    assert state in ['Train', 'Test']
    assert domain in ['Source', 'Target']

    key =  state + '_' + domain

    #@ Accuracy
    writer.add_scalars('Accuracy/' + key, {'global': accs[0], 'left_eye':accs[1], 'right_eye':accs[2], 'nose':accs[3], 'left_mouse':accs[4], 'right_mouse':accs[5], 'global_local':accs[6]}, epoch)
    #@ Precision
    writer.add_scalars('Precision/' + key, {'global': precs[0], 'left_eye':precs[1], 'right_eye':precs[2], 'nose':precs[3], 'left_mouse':precs[4], 'right_mouse':precs[5], 'global_local':precs[6]}, epoch)
    #@ Recall
    writer.add_scalars('Recall/' + key, {'global': recalls[0], 'left_eye':recalls[1], 'right_eye':recalls[2], 'nose':recalls[3], 'left_mouse':recalls[4], 'right_mouse':recalls[5], 'global_local':recalls[6]}, epoch)
    #@ F1 Score
    writer.add_scalars('F1/' + key, {'global': f1s[0], 'left_eye':f1s[1], 'right_eye':f1s[2], 'nose':f1s[3], 'left_mouse':f1s[4], 'right_mouse':f1s[5], 'global_local':f1s[6]}, epoch)
    #@ Loss - CrossEntropy
    writer.add_scalars('CrossEntropy/' + key, {'global': ce_loss[0].avg, 'left_eye':ce_loss[1].avg, 'right_eye':ce_loss[2].avg, 'nose':ce_loss[3].avg, 'left_mouse':ce_loss[4].avg, 'right_mouse':ce_loss[5].avg, 'global_local':ce_loss[6].avg}, epoch)
    #@ Situation (The number of classifiers with the same prediction label corresponds to different situations)
    writer.add_scalars('Situation/Probility/' + key, {'Situation_0': six_probilities[0].avg, 'Situation_1':six_probilities[1].avg, 'Situation_2':six_probilities[2].avg, 'Situation_3':six_probilities[3].avg, 'Situation_4':six_probilities[4].avg, 'Situation_5':six_probilities[5].avg}, epoch)
    writer.add_scalars('Situation/Confidence/' + key, {'Situation_0': six_confidences[0].avg, 'Situation_1':six_confidences[1].avg, 'Situation_2':six_confidences[2].avg, 'Situation_3':six_confidences[3].avg, 'Situation_4':six_confidences[4].avg, 'Situation_5':six_confidences[5].avg}, epoch)

def Visualize_Pseudo_Labels_Generator(args, epoch, writer, confidence, proportion, category_confidence, category_proportion):
    #@ The proportion and confidence of the number of target domains with pseudo labels used for training
    writer.add_scalars('Pseudo_Labels_Generator/Labeled_Prop_Confi', {'confidence': confidence, 'probility': proportion}, epoch)

    #@ Plot the proportion and confidence of pseudo labels on each category in a bar chart
    fig = Draw_PL_Distribute_Bar(category_confidence, category_proportion, args.targetDataset)
    writer.add_figure('Pseudo_Labels_Generator/Category_Confidence_Proportion', fig, global_step=epoch, close=True)

def Visualize_Class_Similarity_Heatmap(matrix, ticks=['Surprised', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral'], title='similarity [0~1]'):

    def heatmap(data, row_labels, col_labels, ax=None,
                cbar_kw=None, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (M, N).
        row_labels
            A list or array of length M with the labels for the rows.
        col_labels
            A list or array of length N with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if ax is None:
            ax = plt.gca()

        if cbar_kw is None:
            cbar_kw = {}

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                rotation_mode="anchor")

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                        textcolors=("black", "white"),
                        threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    fig, ax = plt.subplots()
                        
    im, cbar = heatmap(matrix, ticks, ticks, ax=ax, cmap="Blues", cbarlabel=title)
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    return fig

def Visualize_Input_Image(args, writer, state, domain, data_w_trans, data_s_trans=None, data_raw_path=None):
    assert state in ['train', 'test']
    assert domain in ['target', 'source']
    if domain == 'source':
        domain = 'Source-' + args.sourceDataset
    else:
        domain = 'Target-' + args.targetDataset
    #@ Visualize the input imgs
    for i in range(data_w_trans.size()[0]):
        if data_raw_path != None:
            trans = transforms.ToTensor()
            raw_img = trans(cv2.imread(data_raw_path[i]))
            writer.add_image('Input/' + state + '/' + domain + '-Raw', raw_img, i)
        if data_s_trans != None:
            writer.add_image('Input/' + state + '/' + domain + '-StrongTrans', data_s_trans[i].detach().cpu(), i)
        writer.add_image('Input/' + state + '/' + domain + '-WeakTrans', data_w_trans[i].detach().cpu(), i)

def Visualize_Collection_Pseudo_Labels_Category(pseudo_counters, mode='proportion'):
    assert mode in ['proportion', 'quantity']
    props = []
    for classifier_id in range(len(pseudo_counters)):
        counter = pseudo_counters[classifier_id]
        if mode == 'proportion':
            total = sum(counter.values()) - counter[-1] if sum(counter.values()) - counter[-1] > 0 else 1
            prop = [counter[c] / total for c in range(7)]
        elif mode == 'quantity':
            prop = [counter[c] for c in range(7)]
        props.append(prop)


    data = [['Surprised'], ['Fear'], ['Disgust'], ['Happy'], ['Sad'], ['Angry'], ['Neutral']]
    for class_id in range(7):
        for classifier_id in range(len(pseudo_counters)):
            data[class_id].append(props[classifier_id][class_id])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    df=pd.DataFrame(data,columns=["Expression", "Global","L-Eye","R-Eye","Nose", "L-Mouse", "R-Mouse", "G-Local"])
    ax = df.plot(x="Expression", y=["Global","L-Eye","R-Eye","Nose", "L-Mouse", "R-Mouse", "G-Local"], kind="bar", figsize=(9,8), ax=ax, title='collection of category pl', grid=True, rot=360)
    
    return fig

def Visualize_Collection_Pseudo_Labels_Category_ClassWise(classwises):
    props = []
    for classifier_id in range(len(classwises)):
        classwise = classwises[classifier_id].cpu()
        accs = [c.data.item() for c in classwise]
        props.append(accs)


    data = [['Surprised'], ['Fear'], ['Disgust'], ['Happy'], ['Sad'], ['Angry'], ['Neutral']]
    for class_id in range(7):
        for classifier_id in range(len(classwises)):
            data[class_id].append(props[classifier_id][class_id])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    df=pd.DataFrame(data,columns=["Expression", "Global","L-Eye","R-Eye","Nose", "L-Mouse", "R-Mouse", "G-Local"])
    ax = df.plot(x="Expression", y=["Global","L-Eye","R-Eye","Nose", "L-Mouse", "R-Mouse", "G-Local"], kind="bar", figsize=(9,8), ax=ax, title='category thres of each classifiers', grid=True, rot=360)
    
    return fig

def Visualize_Collection_Pseudo_Labels_General(confidences, proportions):


    data = [['Global'], ['L-Eye'], ['R-Eye'], ['Nose'], ['L-Mouse'], ['R-Mouse'], ['G-Local']]
    for classifier_id in range(len(confidences)):
            data[classifier_id].append(confidences[classifier_id].avg)
            data[classifier_id].append(proportions[classifier_id].avg)
        
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    df=pd.DataFrame(data, columns=["Classifier", "Confidence", "Proportion"])
    ax = df.plot(x="Classifier", y=["Confidence", "Proportion"], kind="bar", figsize=(9,8), ax=ax, title='pl general', grid=True, rot=360)
    
    return fig

def Visualize_Collection_Difference_PL_between_Classifiers(writer, selected_labels):
    #@ Count the samples with pseudo labels collected by each classifier for each category
    selected_labels = [s.cpu().numpy() for s in selected_labels]
    img_indexs_list = [] # [{0: [1, 1287, 22...], 1: [1213, 32, 520...] ..} ..]
    for classifier_id in range(len(selected_labels)):
        s = selected_labels[classifier_id]
        img_indexs_list.append({})
        for class_id in range(7):
            img_indexs = np.where(s == class_id)[0]
            img_indexs_list[classifier_id][class_id] = img_indexs

    #@ Difference matrix is constructed for each class
    diff_metrixs_list = []
    for class_id in range(7):
        diff_metrixs_list.append([])
        m = diff_metrixs_list[class_id]
        for classifier_i in range(len(selected_labels)):
            m.append([])
            for classifier_j in range(len(selected_labels)):
                if classifier_i == classifier_j:
                    m[classifier_i].append(0)
                else:
                    cnt = 0
                    for index in img_indexs_list[classifier_j][class_id]:
                        if index not in img_indexs_list[classifier_i][class_id]:
                            cnt += 1
                    m[classifier_i].append(cnt)
                
    #@ Visualize them!
    title_list = ['Surprised', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
    for class_id in range(7):
        fig = Visualize_Class_Similarity_Heatmap(np.array(diff_metrixs_list[class_id]), ticks=\
            ['Global', 'L-Eye', 'R-Eye', 'Nose', 'L-Mouse', 'R-Mouse', 'G-Local'], title=title_list[class_id])
        writer.add_figure('Category_Pseudo_Labels/DiffMatrix-' + title_list[class_id], fig, close=True)


def Calculate_Visualize_Class_Similarity(args, model, train_source_dataloader, \
           test_source_dataloader, train_target_dataloader, test_target_dataloader, writer):

    train_source_sim = Cal_Class_Similarity_in_Domain(args, model, train_source_dataloader, args.sourceDataset)
    test_source_sim = Cal_Class_Similarity_in_Domain(args, model, test_source_dataloader, args.sourceDataset)
    train_target_sim = Cal_Class_Similarity_in_Domain(args, model, train_target_dataloader, args.targetDataset)
    test_target_sim = Cal_Class_Similarity_in_Domain(args, model, test_target_dataloader, args.targetDataset)

    train_source_fig = Visualize_Class_Similarity_Heatmap(train_source_sim)
    test_source_fig = Visualize_Class_Similarity_Heatmap(test_source_sim)
    train_target_fig = Visualize_Class_Similarity_Heatmap(train_target_sim)
    test_target_fig = Visualize_Class_Similarity_Heatmap(test_target_sim)

    writer.add_figure('Class_Similarity/' + args.sourceDataset + '_Train', train_source_fig, close=True)
    writer.add_figure('Class_Similarity/' + args.sourceDataset + '_Test', test_source_fig, close=True)
    writer.add_figure('Class_Similarity/' + args.targetDataset + '_Train', train_target_fig, close=True)
    writer.add_figure('Class_Similarity/' + args.targetDataset + '_Test', test_target_fig, close=True)

def Initialize_Mean(args, model, useClassify=True):
    
    model.eval()
    
    source_data_loader = BulidDataloader(args, flag1='train', flag2='source')
    target_data_loader = BulidDataloader(args, flag1='train', flag2='target')
    
    # Source Mean
    mean = None

    for step, (input, landmark, label) in enumerate(source_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Source')

        if step == 0:
            mean = torch.mean(feature,0)
        else:
            mean = step/(step+1) * torch.mean(feature,0) + 1/(step+1) * mean

    if isinstance(model, nn.DataParallel):
        model.module.SourceMean.init(mean)
    else:
        model.SourceMean.init(mean)

    # Target Mean
    mean = None

    for step, (input, landmark, label) in enumerate(target_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Target')

        if step==0:
            mean = torch.mean(feature,0)
        else:
            mean = step/(step+1) * torch.mean(feature,0) + 1/(step+1) * mean

    if isinstance(model, nn.DataParallel):
        model.module.TargetMean.init(mean)
    else:
        model.TargetMean.init(mean)

def Initialize_Mean_Cov(args, model, useClassify=True):

    model.eval()

    source_data_loader = BulidDataloader(args, flag1='train', flag2='source')
    target_data_loader = BulidDataloader(args, flag1='train', flag2='target')

    # Source Mean and Cov
    mean, cov = None, None

    for step, (input, landmark, label) in enumerate(source_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Source')

        if step == 0:
            mean = torch.mean(feature, 0)
            cov = torch.mm((feature-mean).transpose(0, 1), feature-mean) / (feature.size(0)-1) # cov是协方差
        else:
            mean = step/(step+1) * torch.mean(feature, 0) + 1/(step+1) * mean
            cov = step/(step+1) * torch.mm((feature-mean).transpose(0, 1), feature-mean) / (feature.size(0)-1) + 1/(step+1) * cov

    if isinstance(model, nn.DataParallel):
        model.module.SourceMean.init(mean, cov)
    else:
        model.SourceMean.init(mean, cov)

    # Target Mean and Cov
    mean, cov = None, None

    for step, (input, landmark, label) in enumerate(target_data_loader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, pred, loc_pred = model(input, landmark, useClassify, 'Target')

        if step == 0:
            mean = torch.mean(feature, 0)
            cov = torch.mm((feature-mean).transpose(0, 1), feature-mean) / (feature.size(0)-1)
        else:
            mean = step/(step+1) * torch.mean(feature, 0) + 1/(step+1) * mean # 这个是怎么算出来的？
            cov = step/(step+1) * torch.mm((feature-mean).transpose(0, 1), feature-mean) / (feature.size(0)-1) + 1/(step+1) * cov

    if isinstance(model, nn.DataParallel):
        model.module.TargetMean.init(mean, cov)
    else:
        model.TargetMean.init(mean, cov)

def Initialize_Mean_Cluster(args, model, useClassify=True,source_data_loader=None,target_data_loader=None):

    model.eval() # 这会让model.training = False
    
    # Source Cluster of Mean
    Feature = []
    EndTime = time.time()
    # source_data_loader = BulidDataloader(args, flag1='train', flag2='source')

    source_data_bar = tqdm(source_data_loader)
    for step, (_, input, landmark, label) in enumerate(source_data_bar):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            #! m21-11-12，注释
            # feature, output, loc_output = model(input, landmark)
            #@ m21-11-11, 新的model调用
            features, preds = model(input, landmark)
        source_data_bar.desc = 'initialize the source domain'

        Feature.append(features[6].cpu().data.numpy())
    Feature = np.vstack(Feature)    # shape is (12256, 384) while source dataset is RAF-DB

    # Using K-Means
    kmeans = KMeans(n_clusters=args.class_num, init='k-means++', algorithm='full')
    kmeans.fit(Feature)
    centers = torch.Tensor(kmeans.cluster_centers_).to('cuda' if torch.cuda.is_available else 'cpu') # shape is (7, 384)，通过feature获取七个类别的分布点

    if isinstance(model, nn.DataParallel):
        model.module.SourceMean.init(centers)
    else:
        model.SourceMean.init(centers) # 通过这里初始化了SourceMean里边的self.running_mean

    print('[Source Domain] Cost time : %fs' % (time.time()-EndTime))

    # Target Cluster of Mean
    Feature = []
    EndTime = time.time()
    # target_data_loader = BulidDataloader(args, flag1='train', flag2='target')

    target_data_bar = tqdm(target_data_loader)
    for step, (_, input, landmark, label) in enumerate(target_data_bar):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            #! m21-11-12，注释
            # feature, output, loc_output = model(input, landmark)
            #@ m21-11-11, 新的model调用
            features, preds = model(input, landmark)
        target_data_bar.desc = 'initialize the target domain'

        Feature.append(features[6].cpu().data.numpy())
    Feature = np.vstack(Feature)

    # Using K-Means
    kmeans = KMeans(n_clusters=args.class_num, init='k-means++', algorithm='full')
    kmeans.fit(Feature)
    centers = torch.Tensor(kmeans.cluster_centers_).to('cuda' if torch.cuda.is_available else 'cpu')

    if isinstance(model, nn.DataParallel):
        model.module.TargetMean.init(centers)
    else:
        model.TargetMean.init(centers)

    print('[Target Domain] Cost time : %fs' % (time.time()-EndTime))

def Visualization(figName, model, dataloader=None, useClassify=True, domain='Source'):
    '''Feature Visualization in Source/Target Domain.'''
    
    assert useClassify in [True, False], 'useClassify should be bool.'
    assert domain in ['Source', 'Target'], 'domain should be source or target.'

    dataloader = tqdm(dataloader)
    model.eval() # 先设置不保存梯度

    Feature, Label = [], []

    # Get Cluster
    for i in range(7):
        if domain == 'Source':
            Feature.append(model.SourceMean.running_mean[i].cpu().data.numpy()) # 把七个聚类点先存进Features列表中
        elif domain == 'Target':
            Feature.append(model.TargetMean.running_mean[i].cpu().data.numpy())
    Label.append(np.array([7 for i in range(7)])) # 先预设7个聚类点，这7个点对应Feature里边的前7个点，作为聚类点

    # Get Feature and Label
    for step, (input, landmark, label) in enumerate(dataloader):
        '''
        参数解释
        input: 裁剪后的输入的图片，shape应该是128*128*3
        landmark: 五个关键点的坐标
        label: 该图片所属的表情标签
        '''
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, output, loc_output = model(input, landmark, useClassify, domain)
            '''
            feature: batch * 384
            output : batch * 7 全局的预测值
            loc_output: batch * 7 局部的预测值
            '''
        Feature.append(feature.cpu().data.numpy())
        Label.append(label.cpu().data.numpy())


    Feature = np.vstack(Feature) # (7+N, 384) where N is the number of datasets
    '''
    在执行这句代码之前，Feature的前7项每一项的shape都是1*384，这七项存的是预设的七个聚类点
    然后在用dataloader弹出的图片的循环之后Feature增添的每一项的shape都是batch*384，
    然后因为没有对vstack的axis参数赋值，所以默认是第0维，因此在执行这句代码之后，会将Feature
    列表中的每个元素的第0维进行叠加，最后就形成了一个(32771,384)的列表，其中每一行就代表这张图片的features特征
    '''
    Label = np.concatenate(Label) # (N+7, 1)

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3)
    embedding = tsne.fit_transform(Feature) # 对Features进行压缩，压缩后的shape会变成（32771，2）

    # Draw Visualization of Feature
    colors = {0:'red', 1:'blue', 2:'olive',  3:'green',  4:'orange',  5:'purple',  6:'darkslategray', 7:'black'}
    labels = {0:'Surprised', 1:'Fear', 2:'Disgust',  3:'Happy',  4:'Sad',  5:'Angry',  6:'Neutral', 7:'Cluster'}
    # labels = {0:'惊讶', 1:'恐惧', 2:'厌恶', 3:'开心', 4:'悲伤', 5:'愤怒', 6:'平静', 7:'聚类中心'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min) # 求均值，data_norm的shape是（32771，2）

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(7):
        data_x, data_y = data_norm[Label==i][:,0], data_norm[Label==i][:,1] # 找出所有label=i的点
        scatter = plt.scatter(data_x, data_y, c='m', edgecolors=colors[i], s=5, label=labels[i], marker='^', alpha=0.6) # 画图
    scatter = plt.scatter(data_norm[Label==7][:,0], data_norm[Label==7][:,1], c=colors[7], s=20, label=labels[7], marker='^', alpha=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    
    plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i]) ) for i in range(8)],
               loc='upper left',
            #    prop = {'size':8},
               bbox_to_anchor=(1.05,0.85),
               borderaxespad=0)
    plt.savefig(fname='{}'.format(figName), format="jpg", bbox_inches = 'tight')
    # plt.show()
def VisualizationForTwoDomain(figName, model, source_dataloader, target_dataloader, useClassify=True, showClusterCenter=True):
    '''Feature Visualization in Source and Target Domain.'''
    
    model.eval()

    Feature_Source, Label_Source, Feature_Target, Label_Target = [], [], [], []

    # Get Feature and Label in Source Domain
    if showClusterCenter:
        for i in range(7):
            Feature_Source.append(model.SourceMean.running_mean[i].cpu().data.numpy())
        Label_Source.append(np.array([7 for i in range(7)]))   

    for step, (input, landmark, label) in enumerate(source_dataloader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, output, loc_output = model(input, landmark, useClassify, domain='Source')

        Feature_Source.append(feature.cpu().data.numpy())
        Label_Source.append(label.cpu().data.numpy())

    Feature_Source = np.vstack(Feature_Source)
    Label_Source = np.concatenate(Label_Source)

    # Get Feature and Label in Target Domain
    if showClusterCenter:
        for i in range(7):
            Feature_Target.append(model.TargetMean.running_mean[i].cpu().data.numpy())
        Label_Target.append(np.array([7 for i in range(7)]))

    for step, (input, landmark, label) in enumerate(target_dataloader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, output, loc_output = model(input, landmark, useClassify, domain='Target')

        Feature_Target.append(feature.cpu().data.numpy())
        Label_Target.append(label.cpu().data.numpy())

    Feature_Target = np.vstack(Feature_Target)
    Label_Target = np.concatenate(Label_Target)

    # Sampling from Source Domain
    Feature_Temple, Label_Temple = [], []
    for i in range(8):
        num_source = np.sum(Label_Source == i)
        num_target = np.sum(Label_Target == i)

        num = num_source if num_source <= num_target else num_target 

        Feature_Temple.append(Feature_Source[Label_Source == i][:num])
        Label_Temple.append(Label_Source[Label_Source == i][:num])
 
    Feature_Source = np.vstack(Feature_Temple)
    Label_Source = np.concatenate(Label_Temple)

    Label_Target += 8 # 加8是为了后面直接通过Label == i 来获取target中的标签，因为Source和Target已经合并在一起了，通过这个方法就很巧

    Feature = np.vstack((Feature_Source, Feature_Target))
    Label = np.concatenate((Label_Source, Label_Target)) # 做成了一个元祖

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3)
    embedding = tsne.fit_transform(Feature)

    # Draw Visualization of Feature
    colors = {0:'firebrick', 1:'aquamarine', 2:'goldenrod',  3:'cadetblue',  4:'saddlebrown',  5:'yellowgreen',  6:'navy'}
    labels = {0:'Surprised', 1:'Fear', 2:'Disgust',  3:'Happy',  4:'Sad',  5:'Angry',  6:'Neutral'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(7):

        data_source_x, data_source_y = data_norm[Label == i][:, 0], data_norm[Label == i][:, 1]
        source_scatter = plt.scatter(data_source_x, data_source_y, color="none", edgecolor=colors[i], s=20, label=labels[i], marker="o", alpha=0.4, linewidth=0.5)
        
        data_target_x, data_target_y = data_norm[Label == (i+8)][:, 0], data_norm[Label == (i+8)][:, 1]
        target_scatter = plt.scatter(data_target_x, data_target_y, color=colors[i], edgecolor="none", s=30, label=labels[i], marker="x", alpha=0.6, linewidth=0.2)

        if i == 0:
            source_legend = source_scatter
            target_legend = target_scatter

    if showClusterCenter:
        source_cluster = plt.scatter(data_norm[Label == 7][:, 0], data_norm[Label == 7][:, 1], c='black', s=20, label='Source Cluster Center', marker='^', alpha=1)
        target_cluster = plt.scatter(data_norm[Label == 15][:, 0], data_norm[Label == 15][:, 1], c='black', s=20, label='Target Cluster Center', marker='s', alpha=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    '''
    l1 = plt.legend(handles=[mpatches.Patch(color=colors[i], 
                    label="{:s}".format(labels[i]) ) for i in range(7)], 
                    loc='upper left', 
                    prop = {'size':8})
                    #bbox_to_anchor=(1.05,0.85), 
                    #borderaxespad=0)
    
    if showClusterCenter:
        plt.legend([source_legend, target_legend, source_cluster, target_cluster],
                   ['Source Domain', 'Target Domain', 'Source Cluster Center', 'Target Cluster Center'],
                   loc='lower left',
                   prop = {'size':7})
    else:
        plt.legend([source_legend, target_legend], ['Source Domain', 'Target Domain'], loc='lower left', prop = {'size':7})
    plt.gca().add_artist(l1)
    '''
    plt.savefig(fname='{}.jpg'.format(figName), format="jpg", bbox_inches='tight')

def makeFolder(args):
    os.makedirs(args.OutputPath)
    os.makedirs(args.OutputPath + '/result_pics/train/source')
    os.makedirs(args.OutputPath + '/result_pics/train/target')
    os.makedirs(args.OutputPath + '/result_pics/test/source')
    os.makedirs(args.OutputPath + '/result_pics/test/target')
    os.makedirs(args.OutputPath + '/result_pics/train_tow_domain')
    os.makedirs(args.OutputPath + '/result_pics/test_tow_domain')
    os.mkdir(args.OutputPath + '/code')
    prefixPath = args.OutputPath + '/code'
    if not os.path.exists(prefixPath):
        os.mkdir(prefixPath)

    rootdir = './'
    list = os.listdir(rootdir)
    for line in list:
        filepath = os.path.join(rootdir, line)
        if os.path.isfile(filepath):
            try:
                shutil.copyfile(filepath, os.path.join(prefixPath, filepath.split('/')[-1]))
            except:
                continue
    
    # for fileName in ['AdversarialNetwork.py', 'Dataset.py', 'demo.py', 'getPreTrainedModel_ResNet.py', 'GraphConvolutionNetwork.py', 'Loss.py', 'Model.py',
    #     'ResNet.py', 'TrainOnSourceDomain.py', 'TrainOnSourceDomain.sh', 'TransferToTargetDomain.py', 'TransferToTargetDomain.sh', 'Utils.py', 'MobileNet.py', 'VGG.py']:
    #     try:
    #         shutil.copyfile(fileName, os.path.join(prefixPath, fileName.split('/')[-1]))
    #     except:
    #         continue

def Output_Configuration_transfer(args):
    # Experiment Information
    print('Log Name: %s' % args.Log_Name)
    print('Output Path: %s' % args.OutputPath)
    print('Backbone: %s' % args.Backbone)
    print('Resume Model: %s' % args.Resume_Model)
    print('CUDA_VISIBLE_DEVICES: %s' % args.GPU_ID)

    print('================================================')

    print('Use {} * {} Image'.format(args.faceScale, args.faceScale))
    print('SourceDataset: %s' % args.sourceDataset)
    print('TargetDataset: %s' % args.targetDataset)
    print('Train Batch Size: %d' % args.train_batch_size)
    print('Test Batch Size: %d' % args.test_batch_size)

    print('================================================')

    if args.showFeature:
        print('Show Visualiza Result of Feature.')

    if args.isTest:
        print('Test Model.')
    else:
        print('Learning Rate: %f' % args.lr)
        print('Momentum: %f' % args.momentum)
        print('Train Epoch: %d' % args.epochs)
        print('Weight Decay: %f' % args.weight_decay)

        if args.useAFN:
            print('Use AFN Loss: %s' % args.methodOfAFN)
            if args.methodOfAFN == 'HAFN':
                print('Radius of HAFN Loss: %f' % args.radius)
            else:
                print('Delta Radius of SAFN Loss: %f' % args.deltaRadius)
            print('Weight L2 nrom of AFN Loss: %f' % args.weight_L2norm)

        if args.useDAN:
            print('Use DAN Loss: %s' % args.methodOfDAN)
            print('Learning Rate(Adversarial Network): %f' % args.lr_ad)

    print('================================================')
    print('Number of classes : %d' % args.class_num)
    if not args.useLocalFeature:
        print('Only use global feature.')
    else:
        print('Use global feature and local feature.')

        if args.useIntraGCN:
            print('Use Intra GCN.')
        if args.useInterGCN:
            print('Use Inter GCN.')

        if args.useRandomMatrix and args.useAllOneMatrix:
            print('Wrong : Use RandomMatrix and AllOneMatrix both!')
            return None
        elif args.useRandomMatrix:
            print('Use Random Matrix in GCN.')
        elif args.useAllOneMatrix:
            print('Use All One Matrix in GCN.')

        if args.useCov and args.useCluster:
            print('Wrong : Use Cov and Cluster both!')
            return None
        else:
            if args.useCov:
                print('Use Mean and Cov.')
            else:
                print('Use Mean.') if not args.useCluster else print('Use Mean in Cluster.')
    print('================================================')

def Output_Configuration_preTransfer(args):
    # Experiment Information
    print('Log Name: %s' % args.Log_Name)
    print('Output Path: %s' % args.OutputPath)
    print('Backbone: %s' % args.Backbone)
    print('Resume Model: %s' % args.Resume_Model)
    print('CUDA_VISIBLE_DEVICES: %s' % args.GPU_ID)

    print('================================================')

    print('Use {} * {} Image'.format(args.faceScale, args.faceScale))
    print('SourceDataset: %s' % args.sourceDataset)
    print('TargetDataset: %s' % args.targetDataset)
    print('Train Batch Size: %d' % args.train_batch_size)
    print('Test Batch Size: %d' % args.test_batch_size)

    print('================================================')

    if args.showFeature:
        print('Show Visualiza Result of Feature.')

    if args.isTest:# 只是测试一下模型的性能
        print('Test Model.')
    else: # 正常的训练，打印训练参数
        print('Train Epoch: %d' % args.epochs)
        print('Learning Rate: %f' % args.lr)
        print('Momentum: %f' % args.momentum)
        print('Weight Decay: %f' % args.weight_decay)

        if args.useAFN: # AFN的方法
            print('Use AFN Loss: %s' % args.methodOfAFN)
            if args.methodOfAFN == 'HAFN':    # hard afn
                print('Radius of HAFN Loss: %f' % args.radius)
            else:                           # soft afn
                print('Delta Radius of SAFN Loss: %f' % args.deltaRadius)
            print('Weight L2 nrom of AFN Loss: %f' % args.weight_L2norm)

    print('================================================')

    print('Number of classes : %d' % args.class_num) # 表情的类别数
    if not args.useLocalFeature:
        print('Only use global feature.') # 只使用全局特征
    else:
        print('Use global feature and local feature.')

        if args.useIntraGCN:
            print('Use Intra GCN.') # 是否使用域内GCN进行传播
        if args.useInterGCN:
            print('Use Inter GCN.') # 是否使用域间GCN进行传播

        if args.useRandomMatrix and args.useAllOneMatrix:
            print('Wrong : Use RandomMatrix and AllOneMatrix both!')
            return None
        elif args.useRandomMatrix:
            print('Use Random Matrix in GCN.')
        elif args.useAllOneMatrix:
            print('Use All One Matrix in GCN.')

        if args.useCov and args.useCluster: # 使用协方差矩阵进行初始化or采用k-means算法进行初始化
            print('Wrong : Use Cov and Cluster both!')
            return None
        else:
            if args.useCov:
                print('Use Mean and Cov.') #todo: mean是指什么？
            else:
                print('Use Mean.') if not args.useCluster else print('Use Mean in Cluster.')

    print('================================================')

def construct_args_transfer():
    parser = argparse.ArgumentParser(description='Domain adaptation for Expression Classification')

    parser.add_argument('--Log_Name', type=str, help='Log Name')
    parser.add_argument('--OutputPath', type=str, help='Output Path')
    parser.add_argument('--Backbone', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
    parser.add_argument('--Resume_Model', type=str, help='Resume_Model', default='None')
    parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

    parser.add_argument('--useDAN', type=str2bool, default=False, help='whether to use DAN Loss')
    parser.add_argument('--methodOfDAN', type=str, default='CDAN-E', choices=['CDAN', 'CDAN-E', 'DANN'])

    parser.add_argument('--useAFN', type=str2bool, default=False, help='whether to use AFN Loss')
    parser.add_argument('--methodOfAFN', type=str, default='SAFN', choices=['HAFN', 'SAFN'])
    parser.add_argument('--radius', type=float, default=25.0, help='radius of HAFN (default: 25.0)')
    parser.add_argument('--deltaRadius', type=float, default=1.0, help='radius of SAFN (default: 1.0)')
    parser.add_argument('--weight_L2norm', type=float, default=0.05, help='weight L2 norm of AFN (default: 0.05)')

    parser.add_argument('--faceScale', type=int, default=112, help='Scale of face (default: 112)')
    parser.add_argument('--sourceDataset', type=str, default='RAF', choices=['RAF', 'AFED', 'MMI','FER2013'])
    parser.add_argument('--targetDataset', type=str, default='CK+',
                        choices=['RAF', 'CK+', 'JAFFE', 'MMI', 'Oulu-CASIA', 'SFEW', 'FER2013', 'ExpW', 'AFED', 'WFED'])
    parser.add_argument('--train_batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=64, help='input batch size for testing (default: 64)')
    parser.add_argument('--datasetPath', type=str, default='./Dataset', help='the path of dataset')
    parser.add_argument('--useMultiDatasets', type=str2bool, default=False, help='whether to use MultiDataset')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_ad', type=float, default=0.01)

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='SGD weight decay (default: 0.0005)')

    parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model')
    parser.add_argument('--showFeature', type=str2bool, default=False, help='whether to show feature')

    parser.add_argument('--useIntraGCN', type=str2bool, default=False, help='whether to use Intra-GCN')
    parser.add_argument('--useInterGCN', type=str2bool, default=False, help='whether to use Inter-GCN')
    parser.add_argument('--useLocalFeature', type=str2bool, default=False, help='whether to use Local Feature')

    parser.add_argument('--useRandomMatrix', type=str2bool, default=False, help='whether to use Random Matrix')
    parser.add_argument('--useAllOneMatrix', type=str2bool, default=False, help='whether to use All One Matrix')

    parser.add_argument('--useCov', type=str2bool, default=False, help='whether to use Cov')
    parser.add_argument('--useCluster', type=str2bool, default=False, help='whether to use Cluster')

    parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num_divided', type=int, default=10, help='the number of blocks [0, loge(7)] to be divided')
    parser.add_argument('--judge_criteria', type=str, default='acc', choices=['acc', 'prec', 'recall', 'f1'], help='the criteria to judge whether to save the checkpoint')
    parser.add_argument('--f_type', type=str, default='layer_feature', choices=['layer_feature', 'feature', 'raw_img'], help='feature types used to solve category similarity')
    parser.add_argument('--randomLayer', type=str2bool, default=False, help='whether to use random')
    
    parser.add_argument('--target_loss_ratio', type=int, default=2, help='the ratio of seven classifier using on target label on the base of classifier_loss_ratio')
    parser.add_argument('--mo_m', type=float, default=0.95, help='the momentun ratio')

    args = parser.parse_args()
    return args

def construct_args_preTransfer():
    parser = argparse.ArgumentParser(description='Expression Classification Training')

    parser.add_argument('--Log_Name', type=str,default='ResNet50_CropNet_GCNwithIntraMatrixAndInterMatrix_useCluster_withoutAFN_trainOnSourceDomain_RAFtoAFED', help='Log Name')
    parser.add_argument('--OutputPath', type=str,default='.', help='Output Path')
    parser.add_argument('--Backbone', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet']) # 挑选backbone
    parser.add_argument('--Resume_Model', type=str, help='Resume_Model', default='None') # 导入pretrained模型用的
    parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')  # 选择的gpu型号

    parser.add_argument('--useAFN', type=str2bool, default=False, help='whether to use AFN Loss')
    parser.add_argument('--methodOfAFN', type=str, default='SAFN', choices=['HAFN', 'SAFN']) #  AFN --  Adaptive Feature Norm 一种特征的自适应方法
    parser.add_argument('--radius', type=float, default=40, help='radius of HAFN (default: 25.0)') # k-means计算的半径
    parser.add_argument('--deltaRadius', type=float, default=0.001, help='radius of SAFN (default: 1.0)') #! 这个跟上面那个半径有什么区别
    parser.add_argument('--weight_L2norm', type=float, default=0.05, help='weight L2 norm of AFN (default: 0.05)') # AFN是一种求损失的方法

    #! dataset
    parser.add_argument('--faceScale', type=int, default=112, help='Scale of face (default: 112)') # 人脸图片的尺寸
    parser.add_argument('--sourceDataset', type=str, default='AFED', choices=['RAF', 'AFED', 'MMI', 'FER2013']) # source dataset的名字
    parser.add_argument('--targetDataset', type=str, default='JAFFE', choices=['RAF', 'CK+', 'JAFFE', 'MMI', 'Oulu-CASIA', 'SFEW', 'FER2013', 'ExpW', 'AFED', 'WFED']) # 目标域
    parser.add_argument('--train_batch_size', type=int, default=32, help='input batch size for training (default: 64)') # 训练的batch size
    parser.add_argument('--test_batch_size', type=int, default=32, help='input batch size for testing (default: 64)')   # 测试集的batch size
    parser.add_argument('--datasetPath', type=str, default='./Dataset', help='the path of dataset')
    parser.add_argument('--useMultiDatasets', type=str2bool, default=False, help='whether to use MultiDataset')         #  是否使用多个数据集（使用多个数据集的什么意思呢）

    parser.add_argument('--lr', type=float, default=0.0001) # 学习率
    parser.add_argument('--lr_ad', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=60,help='number of epochs to train (default: 10)') # 训练代数
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.5)')       # 动量 
    parser.add_argument('--weight_decay', type=float, default=0.0001,help='SGD weight decay (default: 0.0005)') # 正则项系数

    parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model') # 是否只是进行测试，意思就是不用训练
    parser.add_argument('--showFeature', type=str2bool, default=True, help='whether to show feature') # 展示特征（咩啊，用来干嘛的，展示特征图把）

    parser.add_argument('--useIntraGCN', type=str2bool, default=True, help='whether to use Intra-GCN') #  在域内是否使用GCN传播
    parser.add_argument('--useInterGCN', type=str2bool, default=True, help='whether to use Inter-GCN') #  在域间是否使用GCN传播
    parser.add_argument('--useLocalFeature', type=str2bool, default=True, help='whether to use Local Feature') # 是否使用局部特征

    parser.add_argument('--useRandomMatrix', type=str2bool, default=False, help='whether to use Random Matrix') # 这个是用来初始化GCN构造的那个矩阵的一种方法
    parser.add_argument('--useAllOneMatrix', type=str2bool, default=False, help='whether to use All One Matrix') # 这个也是用来初始化GCN构造的那个矩阵的一种方法

    parser.add_argument('--useCov', type=str2bool, default=False, help='whether to use Cov') #  一种画图的方法
    parser.add_argument('--useCluster', type=str2bool, default=True, help='whether to use Cluster') # 后面在话TSNE画图的时候用到的

    parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)') #  最后输出的分类的类别数目 
    parser.add_argument('--judge_criteria', type=str, default='acc', choices=['acc', 'prec', 'recall', 'f1'], help='the criteria to judge whether to save the checkpoint')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')          #  随机种子

    return parser.parse_args()

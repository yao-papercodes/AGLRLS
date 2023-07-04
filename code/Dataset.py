import os
import copy
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.utils.data as data

def L_loader(path):
    return Image.open(path).convert('L')

def RGB_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(data.Dataset):
    def __init__(self, imgs, labels, bboxs, landmarks, flag, transform=None, target_transform=None, loader=RGB_loader): # flag是训练集或测试集的标签，loader是图片的加载模式
        self.imgs = imgs            # list
        self.labels = labels        # list
        self.bboxs = bboxs          # list
        self.landmarks = landmarks  # list
        self.transform = transform
        self.target_transform = target_transform # strong transform
        self.loader = loader
        self.flag = flag
        
    def __getitem__(self, index):
        img_index = index
        img, label, bbox, landmark = self.loader(self.imgs[index]), copy.deepcopy(self.labels[index]), copy.deepcopy(self.bboxs[index]), copy.deepcopy(self.landmarks[index])
        ori_img_w, ori_img_h = img.size # 获取图片的长，高
        # BoundingBox，获取对角点的坐标数值
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]

        enlarge_bbox = True # ! 咩用

        if self.flag == 'train': # 训练集的预处理方式
            random_crop = True
            random_flip = True
        elif self.flag == 'test':
            random_crop = False
            random_flip = False

        # Enlarge BoundingBox
        padding_w, padding_h = int(0.5 * max(0, int(0.20 * (right - left)))), int(0.5 * max(0, int(0.20 * (bottom - top))))
    
        if enlarge_bbox: # 扩大bbox
            left = max(left - padding_w, 0) # 减，但是不能减到小于0
            right = min(right + padding_w, ori_img_w) # 加，但是不能加到超过原本图片的宽度

            top = max(top - padding_h, 0)   # 同理
            bottom = min(bottom + padding_h, ori_img_h)

        if random_crop: # 自定义随机裁框的大小
            x_offset = random.randint(-padding_w, padding_w)
            y_offset = random.randint(-padding_h, padding_h)

            left = max(left + x_offset, 0)
            right = min(right - x_offset, ori_img_w)

            top = max(top + y_offset, 0)
            bottom = min(bottom - y_offset, ori_img_h)

        img = img.crop((left, top, right, bottom))  # 裁剪得到新的图片
        crop_img_w, crop_img_h = img.size           # 裁剪后的高宽

        landmark[:, 0] -= left
        landmark[:, 1] -= top  # 求出五个标注点对应于裁剪后的图的位置

        if random_flip and random.random() > 0.5: # 将图片随机翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            landmark[:, 0] = (right - left) - landmark[:, 0] # 翻转之后，五个标注点也要进行翻转

        # Transform Image
        trans_img = self.transform(img)
        _, trans_img_w, trans_img_h = trans_img.size()

        inputSizeOfCropNet = 28 #todo: 这个是怎么得出来的
        landmark[:, 0] = landmark[:, 0] * inputSizeOfCropNet / crop_img_w # 放大？
        landmark[:, 1] = landmark[:, 1] * inputSizeOfCropNet / crop_img_h
        landmark = landmark.astype(np.int)

        grid_len = 7 #todo: 代表什么意思
        half_grid_len = int(grid_len/2) #todo: 有什么几何意义吗？

        for index in range(landmark.shape[0]):
            if landmark[index, 0] <= (half_grid_len - 1):
                landmark[index, 0] = half_grid_len
            if landmark[index, 0] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index, 0] = inputSizeOfCropNet - half_grid_len - 1
            if landmark[index, 1] <= (half_grid_len - 1):
                landmark[index, 1] = half_grid_len
            if landmark[index, 1] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index, 1] = inputSizeOfCropNet - half_grid_len - 1
        if self.target_transform == None: # it means target data don't need to get another strong transform
            return img_index, trans_img, landmark, label # 返回的是（图片的序号，一张裁剪过后并且进行精细定位和随机翻转等预处理里后的人脸图，该图的五个关键点，该图所属的表情标签）
        else:
            strong_trans_img = self.target_transform(img)
            return img_index, trans_img,  strong_trans_img, landmark, label
            

    def __len__(self): 
        return len(self.imgs)
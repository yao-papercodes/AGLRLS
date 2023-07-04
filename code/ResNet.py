import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module

import copy
import numpy as np
from collections import namedtuple

from GraphConvolutionNetwork import GCN, GCNwithIntraAndInterMatrix
from Model import CountMeanOfFeature, CountMeanAndCovOfFeature, CountMeanOfFeatureInCluster

# Support: ['IR_18', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x

class bottleneck_IR(nn.Module): # 
    def __init__(self, in_channel, depth, stride): # 64,64,2
        super(bottleneck_IR, self).__init__()
        if in_channel == depth: # 如果in_channle==depth的话，说明是处于该残差层的第二个残差结构之后
            self.shortcut_layer = MaxPool2d(1, stride)
        else:   # 生成虚线的捷径 
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut

class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride=2):
    # 列表也可以直接拼接，
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers): # num_layers=50
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),  
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),    # 第一个残差层
            get_block(in_channel=64, depth=128, num_units=4),   # 第二个残差层 
            get_block(in_channel=128, depth=256, num_units=14), # 第三个残差层
            get_block(in_channel=256, depth=512, num_units=3)   # 第四个残差层
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class Backbone(nn.Module):
    def __init__(self, numOfLayer, useIntraGCN=True, useInterGCN=True, useRandomMatrix=False, useAllOneMatrix=False, useCov=False, useCluster=False):   
        '''
            参数解释
            num0fLayer: 网络层数（resnet 50 就是50）
            useIntraGCN：
            useInterGCN：
            useRandomMatrix：
            useAllOneMatrix：
            useCov：
            useCluster：
        '''
        
        super(Backbone, self).__init__()

        unit_module = bottleneck_IR # 定义了一个残差结构的构造函数
        
        self.input_layer = Sequential(Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                      BatchNorm2d(64), PReLU(64))

        blocks = get_blocks(numOfLayer)
        '''
        https://img-blog.csdnimg.cn/20210205082944988.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21lc3N5a2luZw==,size_16,color_FFFFFF,t_70
        blocks是一个长度为四的列表
        blocks[i]代表的是第i+1个残差层中的残差结构
        '''
        self.layer1 = Sequential(*[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in blocks[0]]) #get_block(in_channel=64, depth=64, num_units=3)])
        self.layer2 = Sequential(*[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in blocks[1]]) #get_block(in_channel=64, depth=128, num_units=4)])
        self.layer3 = Sequential(*[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in blocks[2]]) #get_block(in_channel=128, depth=256, num_units=14)])
        self.layer4 = Sequential(*[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in blocks[3]]) #get_block(in_channel=256, depth=512, num_units=3)])

        self.output_layer = Sequential(nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1, 1))) # 这个池化层会强行将输入图像的w，h更改为1，采用的是平均池化

        # 这里是用来提取局部特征的
        cropNet_modules = []
        # get_block的函数第一个先构造128->256,然后是num_units个256->256
        cropNet_blocks = [get_block(in_channel=128, depth=256, num_units=2), get_block(in_channel=256, depth=512, num_units=2)]
        for block in cropNet_blocks:
            '''
                cropNet_blocks的内容为:
                [Bottleneck(in_channel=128, depth=256, stride=2), Bottleneck(in_channel=256, depth=256, stride=1)]
                [Bottleneck(in_channel=256, depth=512, stride=2), Bottleneck(in_channel=512, depth=512, stride=1)]
            '''
            for bottleneck in block:
                cropNet_modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        cropNet_modules += [nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU()]
        self.Crop_Net = nn.ModuleList([copy.deepcopy(nn.Sequential(*cropNet_modules)) for i in range(5)]) # 这个网络包含五组同样内容的残差层，针对五组同样的区域，每个残差层里边的残差结构个数是2，2，1

        #@ m21-11-11, 修改了分类器
        # m22-06-08, 修改了分类器，改为了7个
        self.classifiers = nn.ModuleList([nn.Linear(64, 7) for i in range(2)]) # one global feature use one classifer and five landmark use another classifier
        self.classifiers += nn.ModuleList([nn.Linear(384, 7)]) # concat the five landmark and one global feature, use one classifier
        [self.classifiers[i].apply(init_weights) for i in range(len(self.classifiers))]

        # self.l2norm = Normalize(2)
        # self.fc1 = nn.Linear(64, 2048)
        # self.fc2 = nn.Linear(2048, 7)
        # self.fc1.apply(init_weights)
        # self.fc2.apply(init_weights)
        

        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        # self.SourceMean = (CountMeanAndCovOfFeature(64 + 320) if useCov else CountMeanOfFeature(64 + 320)) if not useCluster else CountMeanOfFeatureInCluster(64 + 320)
        # self.TargetMean = (CountMeanAndCovOfFeature(64 + 320) if useCov else CountMeanOfFeature(64 + 320)) if not useCluster else CountMeanOfFeatureInCluster(64 + 320)
        # self.SourceBN = BatchNorm1d(64 + 320)
        # self.TargetBN = BatchNorm1d(64 + 320)
        # self.fc_source_domain = nn.Linear(384 * 2, 384)
        # self.fc_target_domain = nn.Linear(384 * 2, 384)

    def classify(self, imgs, locations, return_layer_features=False):
        '''
            参数解释
            imgs: shape是(batchsize,3,112,112)64是batch size的数目
            locations: (batchsize, 5, 2)
        '''
        featureMap = self.input_layer(imgs)     # featureMap的shape是(batchsize,64,112,112)

        featureMap1 = self.layer1(featureMap)   # Batch * 64  * 56 * 56
        featureMap2 = self.layer2(featureMap1)  # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2)  # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3)  # Batch * 512 * 7  * 7
        global_feature = self.output_layer(featureMap4).view(featureMap.size(0), -1) # Batch * 64
        loc_feature = self.crop_featureMap(featureMap2, locations)                   # Batch * 320
        feature = torch.cat((global_feature, loc_feature), 1)                        # Batch * (64+320)，成功地将全局和局部特征融合在一起,把两个矩阵的第二维叠加

        #@ m21-11-11, 构造7份feature
        features = [feature.narrow(1, i * 64, 64) for i in range(6)]
        features += [feature] # features [global, local1, local2, local3, local4, local5, global-local]
        preds = []
        preds += [self.classifiers[0](features[0])]
        preds += [self.classifiers[1](features[i]) for i in range(1, 6)] # five landmark
        preds += [self.classifiers[2](features[6])] # global local mix feature
        
        if not return_layer_features:
            return features, preds # (batchSize, 384), (7, batchSize, 7): 七个分类器, batchSize张图片7个类别的预测得分
        else:
            return features, preds, [featureMap, featureMap1, featureMap2, featureMap3, featureMap4]

    def transfer(self, imgs, locations, domain='Target'):

        assert domain in ['Source', 'Target'], 'Parameter domain should be Source or Target.'

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * 7 * 7

        global_feature = self.output_layer(featureMap4).view(featureMap.size(0), -1)  # Batch * 64
        loc_feature = self.crop_featureMap(featureMap2, locations)                    # Batch * 320
        feature = torch.cat((global_feature, loc_feature), 1)                         # Batch * (64+320)

        if self.training:
            # Compute Feature
            SourceFeature = feature.narrow(0, 0, feature.size(0)//2)                  # Batch/2 * (64+320) #@ formula 6
            TargetFeature = feature.narrow(0, feature.size(0)//2, feature.size(0)//2) # Batch/2 * (64+320)

            SourceFeature = self.SourceMean(SourceFeature) # Batch/2 * (64+320)
            TargetFeature = self.TargetMean(TargetFeature) # Batch/2 * (64+320)

            SourceFeature = self.SourceBN(SourceFeature)   # Batch/2 * (64+320)
            TargetFeature = self.TargetBN(TargetFeature)   # Batch/2 * (64+320)

            # Compute Mean
            SourceMean = self.SourceMean.getSample(TargetFeature.detach()) # Batch/2 * (64+320)
            TargetMean = self.TargetMean.getSample(SourceFeature.detach()) # Batch/2 * (64+320) #@ formula 7

            SourceMean = self.SourceBN(SourceMean) # Batch/2 * (64+320)
            TargetMean = self.TargetBN(TargetMean) # Batch/2 * (64+320)

            # GCN
            # torch.cat((SourceFeature,TargetMean), 1) shape是（batchSize， 2*（64+320））
            # feature 是 （2*batchSize， 2*（64+320）），下面注释中的Batch = 2 * batchSize
            feature = torch.cat((torch.cat((SourceFeature,TargetMean), 1), torch.cat((SourceMean,TargetFeature), 1) ), 0) # Batch * (64+320 + 64+320)
            feature = self.GCN(feature.view(feature.size(0), 12, -1))                                                       # Batch * 12 * 64

            feature = feature.view(feature.size(0), -1)                                                                     # Batch * (64+320 + 64+320)
            feature = torch.cat((feature.narrow(0, 0, feature.size(0)//2).narrow(1, 0, 64+320), \
                                  feature.narrow(0, feature.size(0)//2, feature.size(0)//2).narrow(1, 64+320, 64+320) ), 0) # Batch * (64+320)
            
            #! m21-11-11, 注释
            # loc_feature = feature.narrow(1, 64, 320)                                                                        # Batch * 320
            # pred = self.fc(feature)             # Batch * 7
            # loc_pred = self.loc_fc(loc_feature) # Batch * 7
            return features, preds # (batchSize * 2, 384), (7, batchSize * 2, 7): 七个分类器, batchSize*2张图片7个类别的预测得分


        # Inference
        if domain=='Source':
            SourceFeature = feature                                         # Batch * (64+320)
            TargetMean = self.TargetMean.getSample(SourceFeature.detach())  # Batch * (64+320)

            SourceFeature = self.SourceBN(SourceFeature)                    # Batch * (64+320)
            TargetMean = self.TargetBN(TargetMean)                          # Batch * (64+320)

            feature = torch.cat((SourceFeature,TargetMean), 1)              # Batch * (64+320 + 64+320)
            feature = self.GCN(feature.view(feature.size(0), 12, -1))       # Batch * 12 * 64

        elif domain=='Target':
            TargetFeature = feature                                         # Batch * (64+320)
            SourceMean = self.SourceMean.getSample(TargetFeature.detach())  # Batch * (64+320)

            SourceMean = self.SourceBN(SourceMean)                          # Batch * (64+320)
            TargetFeature = self.TargetBN(TargetFeature)                    # Batch * (64+320)

            feature = torch.cat((SourceMean,TargetFeature), 1)              # Batch * (64+320 + 64+320)
            feature = self.GCN(feature.view(feature.size(0), 12, -1))       # Batch * 12 * 64

        feature = feature.view(feature.size(0), -1)      # Batch * (64+320 + 64+320)
        if domain=='Source':
            feature = feature.narrow(1, 0, 64+320)       # Batch * (64+320)
        elif domain=='Target':
            feature = feature.narrow(1, 64+320, 64+320)  # Batch * (64+320)

        #! m21-11-11, 注释
        # loc_feature = feature.narrow(1, 64, 320)         # Batch * 320
        # pred = self.fc(feature)             # Batch * 7
        # loc_pred = self.loc_fc(loc_feature) # Batch * 7
        # return feature, pred, loc_pred

        #@ m21-11-11, 构造7份feature
        features = [feature.narrow(1, i * 64, 64) for i in range(6)]
        features += [feature] # features [global, local1, local2, local3, local4, local5, global-local]
        preds = []
        preds += self.classifiers[0](features[0])
        preds += [self.classifiers[i](features[i])for i in range(1, 6)] # five landmark
        preds += [self.classifiers[6](features[6])] # global and fix feature
        return features, preds
    
    def classify_mean_cluster(self, imgs, locations, stage='first'):
        featureMap = self.input_layer(imgs)     # featureMap的shape是(batchsize,64,112,112)

        featureMap1 = self.layer1(featureMap)   # Batch * 64  * 56 * 56
        featureMap2 = self.layer2(featureMap1)  # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2)  # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3)  # Batch * 512 * 7  * 7
        global_feature = self.output_layer(featureMap4).view(featureMap.size(0), -1) # Batch * 64
        loc_feature = self.crop_featureMap(featureMap2, locations)                   # Batch * 320
        feature = torch.cat((global_feature, loc_feature), 1)                        # Batch * (64+320)，成功地将全局和局部特征融合在一起,把两个矩阵的第二维叠加

        if stage == 'first':
            SourceFeature = feature                                         # Batch * (64+320)
            # TargetMean = self.TargetMean.getSample(SourceFeature.detach())  # Batch * (64+320)
            TargetMean = self.TargetMean.getSample(SourceFeature)  # Batch * (64+320)

            SourceFeature = self.SourceBN(SourceFeature)                    # Batch * (64+320)
            TargetMean = self.TargetBN(TargetMean)                          # Batch * (64+320)

            feature = torch.cat((SourceFeature, TargetMean), 1)              # Batch * (64+320 + 64+320)
            feature = self.fc_source_domain(feature)
        
        elif stage == 'second':
            # Compute Feature
            SourceFeature = feature.narrow(0, 0, feature.size(0)//2)                  # Batch/2 * (64+320) #@ formula 6
            TargetFeature = feature.narrow(0, feature.size(0)//2, feature.size(0)//2) # Batch/2 * (64+320)

            SourceFeature = self.SourceMean(SourceFeature) # Batch/2 * (64+320)
            TargetFeature = self.TargetMean(TargetFeature) # Batch/2 * (64+320)

            SourceFeature = self.SourceBN(SourceFeature)   # Batch/2 * (64+320)
            TargetFeature = self.TargetBN(TargetFeature)   # Batch/2 * (64+320)

            # Compute Mean
            SourceMean = self.SourceMean.getSample(TargetFeature.detach()) # Batch/2 * (64+320)
            TargetMean = self.TargetMean.getSample(SourceFeature.detach()) # Batch/2 * (64+320) #@ formula 7

            SourceMean = self.SourceBN(SourceMean) # Batch/2 * (64+320)
            TargetMean = self.TargetBN(TargetMean) # Batch/2 * (64+320)

            feature = torch.cat((self.fc_source_domain(torch.cat((SourceFeature, TargetMean), 1)), self.fc_target_domain(torch.cat((SourceMean,TargetFeature), 1))), 0)


        #@ m21-11-11, 构造7份feature
        features = [feature.narrow(1, i * 64, 64) for i in range(6)]
        features += [feature] # features [global, local1, local2, local3, local4, local5, global-local]
        preds = []
        preds += [self.classifiers[0](features[0])]
        preds += [self.classifiers[i](features[i])for i in range(1, 6)] # five landmark
        preds += [self.classifiers[6](features[6])] # global local mix feature
        preds += [preds[0] + preds[6]] # global + (global, local)
        return features, preds # (batchSize, 384), (7, batchSize, 7): 七个分类器, batchSize张图片7个类别的预测得分

    def forward(self, imgs, locations, img_aug=None, return_layer_features=False, use_flexmatch=False):


        if return_layer_features:
            feat_w, logits_w, layer_feats = self.classify(imgs, locations, return_layer_features)
            return feat_w, logits_w, layer_feats
        else:
            feat_w, logits_w = self.classify(imgs, locations, return_layer_features)

        if use_flexmatch:
            feat_s, logits_s = self.classify(img_aug, locations, return_layer_features)
            return feat_w, logits_w, feat_s, logits_s
        else:
            return feat_w, logits_w

    def output_num(self):
        return 64 * 6


    def get_parameters(self):
        parameter_list = [  {"params":self.input_layer.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer1.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer2.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer3.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer4.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.output_layer.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.classifiers[0].parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.classifiers[1].parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.classifiers[2].parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.Crop_Net.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            # {"params":self.fc_source_domain.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            # {"params":self.fc_target_domain.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            # {"params":self.SourceBN.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            # {"params":self.TargetBN.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            ]
        return parameter_list

    def crop_featureMap(self, featureMap, locations):
        '''
        参数解释：
        featureMap：(batch_size,128,28,28)
        locations : (batch_size,5,2)
        看不懂这个裁剪的原理
        '''

        batch_size = featureMap.size(0)
        map_ch = featureMap.size(1) # map_channel
        map_len = featureMap.size(2)

        grid_ch = map_ch
        grid_len = 7 # 14, 6, 4
        '''
        下面这个裁剪的算法是有问题的，只是保证了裁剪后的框不会超过原图的尺寸，但是裁剪后的尺寸
        不一定是grid_len * grid_len，grid_len=7没报错的原因是因为理想画出来的框没有一个
        超过28*28的size
        '''

        feature_list = []
        for i in range(5): # 部位的种类
            grid_list = []
            for j in range(batch_size):
                # 在原图上每张图的五个部位按照理想化计算出的矩形框的点
                w_min = locations[j, i, 0] - int(grid_len/2)
                w_max = locations[j, i, 0] + int(grid_len/2)
                h_min = locations[j, i, 1] - int(grid_len/2)
                h_max = locations[j, i, 1] + int(grid_len/2)

                # 判定越界条件之后的在featureMap的图框的四个点
                map_w_min = max(0, w_min)
                map_w_max = min(map_len-1, w_max)
                map_h_min = max(0, h_min)
                map_h_max = min(map_len-1, h_max)

                grid_w_min = max(0, 0 - w_min)
                grid_w_max = grid_len + min(0, map_len-1-w_max)
                grid_h_min = max(0, 0 - h_min)
                grid_h_max = grid_len + min(0, map_len-1-h_max)

                grid = torch.zeros(grid_ch, grid_len, grid_len)
                if featureMap.is_cuda:
                    grid = grid.cuda()

                grid[:, grid_h_min:grid_h_max+1, grid_w_min:grid_w_max+1] = featureMap[j, :, map_h_min:map_h_max+1, map_w_min:map_w_max+1]
                grid_list.append(grid)

            feature = torch.stack(grid_list, dim=0) # shape is (batchsize, 128, 7, 7)
            feature_list.append(feature)


        # feature_list: 5 * [ batch_size * channel * 7 * 7 ]
        output_list = []
        for i in range(5):
            output = self.Crop_Net[i](feature_list[i])  # (64, 7, 7)
            output = self.GAP(output)   # (64, 1, 1)
            output_list.append(output)

        loc_feature = torch.stack(output_list, dim=1)  # batch_size * 5 * 64 * 1 * 1
        loc_feature = loc_feature.view(batch_size, -1) # batch_size * 320

        return loc_feature

class Backbone_onlyGlobal(nn.Module):
    def __init__(self):

        super(Backbone_onlyGlobal, self).__init__()

        unit_module = bottleneck_IR

        self.input_layer = Sequential(Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        blocks = get_blocks(numOfLayer)
        self.layer1 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[0]]) #get_block(in_channel=64, depth=64, num_units=3)])
        self.layer2 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[1]]) #get_block(in_channel=64, depth=128, num_units=4)])
        self.layer3 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[2]]) #get_block(in_channel=128, depth=256, num_units=14)])
        self.layer4 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[3]]) #get_block(in_channel=256, depth=512, num_units=3)])

        self.output_layer = Sequential(nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1,1)))

        self.fc = nn.Linear(64, 7)
        self.fc.apply(init_weights)

    def classify(self, imgs, locations):

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * 7 * 7

        feature = self.output_layer(featureMap4).view(featureMap.size(0), -1) # Batch * 64

        pred = self.fc(feature)              # Batch * 7
        loc_pred = None

        return feature, pred, loc_pred

    def transfer(self, imgs, locations, domain='Target'):

        assert domain in ['Source', 'Target'], 'Parameter domain should be Source or Target.'

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * 7 * 7

        feature = self.output_layer(featureMap4).view(featureMap.size(0), -1)  # Batch * 64

        pred = self.fc(feature)  # Batch * 7
        loc_pred = None

        return feature, pred, loc_pred

    def forward(self, imgs, locations, flag=True, domain='Target'):

        if flag:
            return self.classify(imgs, locations)

        return self.transfer(imgs, locations, domain)

    def output_num(self):
        return 64

    def get_parameters(self):
        parameter_list = [  {"params":self.input_layer.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer1.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer2.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer3.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer4.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.output_layer.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            ]
        return parameter_list

def IR(numOfLayer, useIntraGCN, useInterGCN, useRandomMatrix, useAllOneMatrix, useCov, useCluster):
    """Constructs a ir-18/ir-50 model."""

    model = Backbone(numOfLayer, useIntraGCN, useInterGCN, useRandomMatrix, useAllOneMatrix, useCov, useCluster)

    return model

def IR_onlyGlobal(numOfLayer):
    """Constructs a ir-18/ir-50 model."""

    model = Backbone_onlyGlobal(numOfLayer)

    return model

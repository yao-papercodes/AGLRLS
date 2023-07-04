Log_Name='_CropNet_trainOnSourceDomain_'
datasetPath='/home/shuxin636/YAO/CD_FER_CODE/Dataset'
# Resume_Model='../preTrainedModel/ir50_ms1m_112_CropNet_GCNwithIntraMatrixAndInterMatrix_useCluster.pkl'
Resume_Model='../preTrainedModel/MobileNetV2.pth'
OutputPath='./exp'
GPU_ID='0'
Backbone='MobileNet'
faceScale=112
sourceDataset='RAF'
targetDataset='CK+'
train_batch_size=64
test_batch_size=64
epochs=15
lr=0.0001
momentum=0.9
weight_decay=0.0005
isTest='False'
showFeature='False'
class_num=7
judge_criteria='f1'

#@ -- Unimportant Parameters -------
lr_ad=0
radius=40
useCov='False'
useCluster='False'
methodOfAFN='SAFN'
useAFN='False'
deltaRadius=0.001
weight_L2norm=0.05
useMultiDatasets='True'
useIntraGCN='False'
useInterGCN='False'
useRandomMatrix='False'
useAllOneMatrix='False'
useLocalFeature='True'
#@ ----------------------------------

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=${GPU_ID} python3 TrainOnSourceDomain.py \
    --Log_Name ${Log_Name} \
    --OutputPath ${OutputPath} \
    --Resume_Model ${Resume_Model} \
    --GPU_ID ${GPU_ID} \
    --Backbone ${Backbone} \
    --useAFN ${useAFN} \
    --methodOfAFN ${methodOfAFN} \
    --radius ${radius} \
    --deltaRadius ${deltaRadius} \
    --weight_L2norm ${weight_L2norm} \
    --faceScale ${faceScale} \
    --sourceDataset ${sourceDataset}\
    --targetDataset ${targetDataset}\
    --train_batch_size ${train_batch_size} \
    --test_batch_size ${test_batch_size} \
    --useMultiDatasets ${useMultiDatasets} \
    --epochs ${epochs}\
    --lr ${lr}\
    --lr_ad ${lr_ad}\
    --momentum ${momentum}\
    --weight_decay ${weight_decay}\
    --isTest ${isTest}\
    --showFeature ${showFeature}\
    --class_num ${class_num}\
    --useIntraGCN ${useIntraGCN}\
    --useInterGCN ${useInterGCN}\
    --useLocalFeature ${useLocalFeature}\
    --useRandomMatrix ${useRandomMatrix}\
    --useAllOneMatrix ${useAllOneMatrix}\
    --useCov ${useCov}\
    --useCluster=${useCluster}\
    --datasetPath=${datasetPath}\
    --judge_criteria=${judge_criteria}
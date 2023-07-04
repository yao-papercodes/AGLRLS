Log_Name='_CropNet_transferToTargetDomain_'
datasetPath='/home/shuxin636/YAO/CD_FER_CODE/Dataset'
# Resume_Model='../preTrainedModel/ir50_ms1m_112_CropNet_GCNwithIntraMatrixAndInterMatrix_useCluster.pkl'
Resume_Model='./exp/1670324103/ResNet50_CropNet_trainOnSourceDomain_RAFtoFER2013.pkl'


OutputPath='./exp'
GPU_ID=0
useDAN='True'
Backbone='ResNet50'
methodOfDAN='CDAN-E'
faceScale=112
sourceDataset='RAF'
targetDataset='FER2013'
train_batch_size=16
test_batch_size=64
epochs=30
lr=0.00001
lr_ad=0.001
momentum=0.9
weight_decay=0.0005
isTest='False'
mo_m=0.2
class_num=7
num_divided=10
target_loss_ratio=5
judge_criteria='acc'
f_type='layer_feature'

#@ -- Unimportant Parameters -------
radius=25
deltaRadius=1
weight_L2norm=0.05
useCov='False'
useCluster='False'
useAllOneMatrix='False'
useLocalFeature='True'
useRandomMatrix='False'
useMultiDatasets='False'
useInterGCN='False'
useIntraGCN='False'
showFeature='False'
useAFN='False'
methodOfAFN='SAFN'
#@ ----------------------------------

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=${GPU_ID} python TransferToTargetDomain.py\
    --Log_Name ${Log_Name}\
    --OutputPath ${OutputPath}\
    --Backbone ${Backbone}\
    --Resume_Model ${Resume_Model}\
    --GPU_ID ${GPU_ID}\
    --useAFN ${useAFN}\
    --methodOfAFN ${methodOfAFN}\
    --radius ${radius}\
    --deltaRadius ${deltaRadius}\
    --weight_L2norm ${weight_L2norm}\
    --useDAN ${useDAN}\
    --methodOfDAN ${methodOfDAN}\
    --faceScale ${faceScale}\
    --sourceDataset ${sourceDataset}\
    --targetDataset ${targetDataset}\
    --train_batch_size ${train_batch_size}\
    --test_batch_size ${test_batch_size}\
    --useMultiDatasets ${useMultiDatasets}\
    --epochs ${epochs}\
    --lr ${lr}\
    --lr_ad ${lr_ad}\
    --momentum ${momentum}\
    --weight_decay ${weight_decay}\
    --isTest ${isTest}\
    --showFeature ${showFeature}\
    --class_num ${class_num}\
    --num_divided ${num_divided}\
    --useIntraGCN ${useIntraGCN}\
    --useInterGCN ${useInterGCN}\
    --useLocalFeature ${useLocalFeature}\
    --useRandomMatrix ${useRandomMatrix}\
    --useAllOneMatrix ${useAllOneMatrix}\
    --useCov ${useCov}\
    --useCluster ${useCluster}\
    --target_loss_ratio ${target_loss_ratio}\
    --datasetPath=${datasetPath}\
    --mo_m=${mo_m}\
    --judge_criteria=${judge_criteria}\
    --f_type=${f_type}
    
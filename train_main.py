# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
Workdir = "D:/AlanTan/CNN"
os.chdir(Workdir)

import pandas as pd

import dataproprocess as dp
import ResNet 
import Vgg

def proprocess():
    LabelsPath = "D:/kaggle/detection/trainLabels"
    df_Labels=pd.read_csv(os.path.join(LabelsPath,'trainLabels.csv'))
    df_Labels=df_Labels.set_index(['image'])
    
    data_source = dp.DataSource("D:/kaggle/detection/train",
                                    ['0','1','2','3','4'])
    data_source.classifyData(df_Labels)
    data_source.transformData()
    data_source.splitTrainTest()   
    # data_source.augmentImages()     
    
def resnetTrain():
    ############################################################################
    # use ResNet to training data          
    ############################################################################  
            
    # Network architecture params
    net_architecture_params = {'num_classes': 2,
                               'num_filters': 128, #128
                               'num_blocks': 3,
                               'num_sub_blocks': 2,
                               'use_max_pool': True,
                               'input_shape': (256,256,3)}
    
    data_source_path = "D:/kaggle/detection/train_transform_256_2"
    
    mymodel = ResNet.ResNetModel(data_source_path,training_params,net_architecture_params)
    
    mymodel.modelDefinition()
    mymodel.modelSaveInfo('resnet_model.h5')
    mymodel.modelTrain()
    mymodel.loadSavedModel()
    evaluation = mymodel.modelEvaluate()
    evaluation_s = mymodel.modelEvaluate(saved_model=True)
    mymodel.modelReset()
    return evaluation,evaluation_s

def vggTrain():
    ###########################################################################
    # use VGG5 to training data
    ###########################################################################
    
                        
    # Network architecture params
    net_architecture_params = {'num_classes': 5,  
                               'input_shape': (512,512,3)}
    
    data_source_path = "D:/kaggle/detection/train_transform_512"
    
    mymodel_vgg = Vgg.VggModel_5(data_source_path,training_params,net_architecture_params)
    
    mymodel_vgg.modelDefinition()
    mymodel_vgg.modelSaveInfo('vgg_model_512_5class.h5')
    mymodel_vgg.modelTrain()
    mymodel_vgg.loadSavedModel()
#    df_feature_train = mymodel_vgg.featureDF(mymodel_vgg.train_path)
#    df_feature_validation = mymodel_vgg.featureDF(mymodel_vgg.validation_path)
#    df_feature_test = mymodel_vgg.featureDF(mymodel_vgg.test_path)
    evaluation_vgg = mymodel_vgg.modelEvaluate()
    evaluation_s_vgg = mymodel_vgg.modelEvaluate(saved_model=True)
    mymodel_vgg.modelReset()
    return evaluation_vgg,evaluation_s_vgg
        

if __name__ == '__main__':    
    training_params = {'batch_size': 32,
                        'epochs': 150,
                        'class_weight': 'auto',
                        'train_steps': 1000, # train_samples_size // batch_size = 1290
                        'validation_steps': 1000} # 1000
    
    a, b = vggTrain()


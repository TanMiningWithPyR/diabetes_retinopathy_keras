# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:50:47 2017

@author: admin
"""

from __future__ import print_function

import os
import multiprocessing as mul

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout
from keras.layers import MaxPooling2D, AveragePooling2D, Input, Flatten,LeakyReLU
from keras.optimizers import Adam, rmsprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import load_model
import numpy as np
import pandas as pd
import scipy

import showprocess

class ResNetModel:
    def __init__(self,data_source_path,training_params,net_architecture_params):
        # training_params and net_architecture_params should be a dictionary        
        self.train_path = os.path.join(data_source_path,'train')
#        self.train_augmentation = os.path.join(data_source_path,'augmentation')
        self.validation_path = os.path.join(data_source_path,'validation')
        self.test_path = os.path.join(data_source_path,'test')
        self.training_params = training_params
        self.net_architecture_params = net_architecture_params

    def modelDefinition(self):
        # Start model definition.
        inputs = Input(shape=self.net_architecture_params['input_shape'])
        x = Conv2D(self.net_architecture_params['num_filters'],
                   kernel_size=7, # 7
                   padding='same',
                   strides=2,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.5)(x)
        
        if self.net_architecture_params['use_max_pool']:
            x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
            self.net_architecture_params['num_blocks']=3
            
        for i in range(self.net_architecture_params['num_blocks']):
            for j in range(self.net_architecture_params['num_sub_blocks']):
                strides = 1
                is_first_layer_but_not_first_block = j == 0 and i > 0
                if is_first_layer_but_not_first_block:
                    strides =2
                y = Conv2D(self.net_architecture_params['num_filters'],
                           kernel_size=3,
                           padding='same',
                           strides=strides,
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(x)
                y = BatchNormalization()(y)
                y = LeakyReLU(0.5)(y)
                y = Conv2D(self.net_architecture_params['num_filters'],
                           kernel_size=3,
                           padding='same',
                           strides=strides, # the code of this lines was added by myself 
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(x)
                y = BatchNormalization()(y)
                if is_first_layer_but_not_first_block:
                    x = Conv2D(self.net_architecture_params['num_filters'],
                               kernel_size=1,
                               padding='same',
                               strides=2,
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(1e-4))(x)
                x = keras.layers.add([x,y])
                x = LeakyReLU(0.5)(x)
            
            self.net_architecture_params['num_filters'] = 2 * \
            self.net_architecture_params['num_filters']
        # Add classifier on top        
        x = AveragePooling2D()(x)
        y = Flatten()(x)

        outputs = Dense(self.net_architecture_params['num_classes'],
                        activation='softmax',
                        kernel_initializer='he_normal',
                        activity_regularizer=l2(1e-2))(y)
        # Instantiate and compile model   
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(1e-6),
                      metrics=['accuracy'])
        self.model.summary()

    def modelSaveInfo(self,model_name): # model_name = 'resnet_model.h5'
        save_dir = os.path.join(os.getcwd(), 'saved_models')        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.model_filepath = os.path.join(save_dir, model_name)
        
        checkpoint = ModelCheckpoint(self.model_filepath,
                                     verbose=1,
                                     save_best_only=True)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-10)
        self.callbacks = [checkpoint, lr_reducer]
        
    def modelTrain(self):        
        print("Using real-time data augmentation.")
        # This will do preprocessing and realtime data augmentation:
        train_datagen = ImageDataGenerator(
                zoom_range=0.1,
                rescale=1./255,
                rotation_range=10,
                fill_mode='constant'            
#                channel_shift_range=10
#                width_shift_range=0.1,
#                height_shift_range=0.1,
#                horizontal_flip=True,
#                vertical_flip=True
        )
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = train_datagen.flow_from_directory(
                self.train_path,
                target_size=self.net_architecture_params['input_shape'][0:2],
                batch_size=self.training_params['batch_size'])
        
        self.validation_generator = validation_datagen.flow_from_directory(
                self.validation_path,
                target_size=self.net_architecture_params['input_shape'][0:2],
                batch_size=self.training_params['batch_size'])
        # Compute quantities required for feature-wise normalization
        # if necessary, datagen.fit(x_train)
        
        # Fit the model on the batches generated by datagen.flow_directory().
        self.model.fit_generator(self.train_generator,
                            steps_per_epoch=self.training_params['train_steps'],
                            epochs=self.training_params['epochs'],
                            validation_data=self.validation_generator,
                            validation_steps=self.training_params['validation_steps'],
                            class_weight=self.training_params['class_weight'],
                            workers=8,
                            callbacks=self.callbacks)
        
    def valuePredict(self,onefilepath,model):
        oneimage = scipy.misc.imread(onefilepath)/255 # rescale=1./255
        onevalue = model.predict(np.array([oneimage]))        
        return onevalue
    
    def classPredict(self,onefilepath,saved_model=False):
        if saved_model:
            # max value index
            onelabel = self.valuePredict(onefilepath,self.saved_model).argmax()             
        else:
            onelabel = self.valuePredict(onefilepath,self.model).argmax()
        return onelabel
    
    def modelEvaluate(self,saved_model=False):
        p_index = []
        p_real_class = []
        p_predict_class = []
        
        labels = os.listdir(self.test_path)
        for i in labels:        
            oneclasspath = os.path.join(self.test_path,i)
            files = os.listdir(oneclasspath)
            for j in files:
                p_index.append(j.split(".")[0])
                p_real_class.append(i)
                onefilepath = os.path.join(oneclasspath,j)
                onelabel = self.classPredict(onefilepath,saved_model)
                p_predict_class.append(str(onelabel))
        
        data = {'p_index':p_index,
                'p_real_class':p_real_class,
                'p_predict_class':p_predict_class}
        data_pd = pd.DataFrame(data).set_index('p_index')   
        crosstable = pd.crosstab(data_pd.p_predict_class,
                                 data_pd.p_real_class,
                                 margins=True)
        return {'predict_table': data_pd,
                'crosstable': crosstable}
        
    def intermediateOutput(self,data,layer_name='flatten_1'):
        intermediate_layer_model = Model(inputs=self.saved_model.input,
                                         outputs=self.saved_model.get_layer(layer_name).output)
        intermediate_output = self.valuePredict(data,intermediate_layer_model)
        return intermediate_output[0]
    
    def featureDF(self,file_path,layer_name='flatten_1'):
        photo_index = []
        eyes_no = []
        eyes_left_right = []
        feature = []
        each_label = []
        labels = os.listdir(file_path)        
        for i in labels:
            print("Processing in Class " + i + ".")
            oneclasspath = os.path.join(file_path,i)
            files = [j for j in os.listdir(oneclasspath) if j.find("副本") < 0] 
            process_bar = showprocess.ProgressBar(len(files))
            count = 0 # 进度条计数
            for j in files:
                process_bar.update(count)
                eye_name = j.split(".")[0]
                left_right = eye_name.split("_")[1]
                eye_no = eye_name.split("_")[0]
                one_feature = self.intermediateOutput(os.path.join(oneclasspath,j))
                
                photo_index.append(eye_name)
                eyes_no.append(eye_no)
                eyes_left_right.append(left_right)
                each_label.append(i)  
                feature.append(one_feature)        
                count = count + 1   
                 
        feature_arr = np.array(feature)
        featrue_length = feature_arr.shape[1]
        column_name = ['feature' + str(i) for i in range(featrue_length)]
        feature_df = pd.DataFrame(np.array(feature),
                                  index=photo_index,
                                  columns=column_name)
        
        data = {'photo_index':photo_index,
                'eyes_no':eyes_no,
                'eyes_left_right':eyes_left_right,                
                'each_label':each_label}
        data_pd = pd.DataFrame(data).set_index('photo_index') 
        
        data_df = pd.merge(data_pd,feature_df,left_index=True,right_index=True)
        data_df = data_df.pivot('eyes_no','eyes_left_right')
        
        return data_df
        
    def loadSavedModel(self):
        self.saved_model = load_model(self.model_filepath)      
    
    def modelReset(self):
        K.clear_session()
        print("Clear the graph in backend!")
        
if __name__ == '__main__':
    Workdir = "D:/AlanTan/CNN"
    os.chdir(Workdir)   
    
    training_params = {'batch_size': 16,
                        'epochs': 2,
                        'class_weight': 'auto',
                        'train_steps': 10, # train_samples_size // batch_size
                        'validation_steps': 10}
                        
    # Network architecture params
    net_architecture_params = {'num_classes': 5,
                               'num_filters': 128,
                               'num_blocks': 3,
                               'num_sub_blocks': 2,
                               'use_max_pool': True,
                               'input_shape': (512,512,3)}
    
    data_source_path = "D:/kaggle/detection/foldertest_transform"
    
    mymodel = ResNetModel(data_source_path,training_params,net_architecture_params)
    
    mymodel.modelDefinition()
    mymodel.modelSaveInfo('resnet_model_test.h5')
#    mymodel.modelTrain()
    mymodel.loadSavedModel()
    df_feature = mymodel.featureDF('flatten_1')
    evaluation = mymodel.modelEvaluate()
    mymodel.modelReset()
    
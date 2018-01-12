# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:21:11 2017

@author: admin
"""
import os 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import rmsprop,Adam
from keras.regularizers import l2

import ResNet

class VggModel_5(ResNet.ResNetModel):
    def modelDefinition(self):
        # Defining the model
        self.model = Sequential()
        # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        self.model.add(Conv2D(32, 5, strides=2, padding='same',
                              input_shape=self.net_architecture_params['input_shape']))
        self.model.add(LeakyReLU(0.55))
        self.model.add(Conv2D(32, 3, padding='same'))
        self.model.add(LeakyReLU(0.55))
        self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2), padding='same'))      
        
        self.model.add(Conv2D(64, 5, strides=2, padding='same'))
        self.model.add(LeakyReLU(0.55))
        self.model.add(Conv2D(64, 3, padding='same'))
        self.model.add(LeakyReLU(0.55))
        self.model.add(Conv2D(64, 3, padding='same'))
        self.model.add(LeakyReLU(0.55))
        self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2), padding='same'))

        self.model.add(Conv2D(128, 3, padding='same'))
        self.model.add(LeakyReLU(0.55))
        self.model.add(Conv2D(128, 3, padding='same'))
        self.model.add(LeakyReLU(0.01))
        self.model.add(Conv2D(128, 3, padding='same'))
        self.model.add(LeakyReLU(0.55))
        self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2), padding='same'))

        self.model.add(Conv2D(256, 3, padding='same'))
        self.model.add(LeakyReLU(0.55))
        self.model.add(Conv2D(256, 3, padding='same'))
        self.model.add(LeakyReLU(0.55))
        self.model.add(Conv2D(256, 3, padding='same'))
        self.model.add(LeakyReLU(0.55))
        self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2), padding='same'))
        
        self.model.add(Conv2D(512, 3, padding='same'))
        self.model.add(LeakyReLU(0.55))
        self.model.add(Conv2D(512, 3, padding='same'))
        self.model.add(LeakyReLU(0.55))
        self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2), padding='same'))
        self.model.add(Dropout(0.25))
        
        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(LeakyReLU(0.5))
        self.model.add(Dense(1024))
        self.model.add(LeakyReLU(0.5))
        self.model.add(Dense(512))
        self.model.add(LeakyReLU(0.5))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(self.net_architecture_params['num_classes'], 
                             activation='softmax',
                             activity_regularizer=l2(5e-4)))

        self.model.compile(loss='categorical_crossentropy', 
                      optimizer=Adam(lr=1e-4), 
                      metrics=['accuracy']) 

class VggModel_best(ResNet.ResNetModel):
    def modelDefinition(self):
        # Defining the model
        self.model = Sequential()
        # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        self.model.add(Conv2D(32, 3, strides=2,
                              input_shape=self.net_architecture_params['input_shape']))
        self.model.add(LeakyReLU(0.5))
        self.model.add(Conv2D(32, 3))
        self.model.add(LeakyReLU(0.5))
        self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))      
        
        self.model.add(Conv2D(64, 3))
        self.model.add(LeakyReLU(0.5))
        self.model.add(Conv2D(64, 3))
        self.model.add(LeakyReLU(0.5))
#        self.model.add(Conv2D(64, 3))
#        self.model.add(LeakyReLU(0.01))
        self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

        self.model.add(Conv2D(128, 3))
        self.model.add(LeakyReLU(0.5))
#        self.model.add(Conv2D(128, 3))
#        self.model.add(LeakyReLU(0.01))
        self.model.add(Conv2D(128, 3))
        self.model.add(LeakyReLU(0.5))
        self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

        self.model.add(Conv2D(256, 3))
        self.model.add(LeakyReLU(0.5))
#        self.model.add(Conv2D(256, 3))
#        self.model.add(LeakyReLU(0.01))
        self.model.add(Conv2D(256, 3))
        self.model.add(LeakyReLU(0.5))
        self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        
#        self.model.add(Conv2D(512, 3))
#        self.model.add(LeakyReLU(0.01))
#        self.model.add(Conv2D(512, 3))
#        self.model.add(LeakyReLU(0.01))
        self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        self.model.add(Dropout(0.25))
        
        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(LeakyReLU(0.5))
        self.model.add(Dense(1024))
        self.model.add(LeakyReLU(0.5))
        self.model.add(Dense(512))
        self.model.add(LeakyReLU(0.5))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(self.net_architecture_params['num_classes'], 
                             activation='softmax',
                             activity_regularizer=l2(5e-4)))

        self.model.compile(loss='categorical_crossentropy', 
                      optimizer=Adam(lr=1e-4), 
                      metrics=['accuracy']) 

class VggModel(ResNet.ResNetModel):
    def modelDefinition(self):
        # Defining the model
        self.model = Sequential()
        # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        self.model.add(Conv2D(32, (7,7), strides=2, padding='same',                             
                              input_shape=self.net_architecture_params['input_shape']))
        self.model.add(LeakyReLU(0.5))
        
        self.model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2), padding='same'))
        
        self.model.add(Conv2D(32, (3,3), strides=1, padding='same'))        
        self.model.add(LeakyReLU(0.5))
        
        self.model.add(Conv2D(32, (3,3), strides=1, padding='same'))        
        self.model.add(LeakyReLU(0.5))
        
        self.model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2), padding='same'))           

        self.model.add(Conv2D(64, (3,3), strides=1, padding='same'))
        self.model.add(LeakyReLU(0.5))
        
        self.model.add(Conv2D(64, (3,3), strides=1, padding='same'))
        self.model.add(LeakyReLU(0.5))        

        self.model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2), padding='same'))

        self.model.add(Conv2D(128, (3,3), strides=1, padding='same'))
        self.model.add(LeakyReLU(0.5))
        
        self.model.add(Conv2D(128, (3,3), strides=1, padding='same'))
        self.model.add(LeakyReLU(0.5))
        
        self.model.add(Conv2D(128, (3,3), strides=1, padding='same'))
        self.model.add(LeakyReLU(0.5)) 
        
        self.model.add(Conv2D(128, (3,3), strides=1, padding='same'))
        self.model.add(LeakyReLU(0.5))
        
        self.model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2), padding='same'))
        
        self.model.add(Conv2D(256, (3,3), strides=1, padding='same'))
        self.model.add(LeakyReLU(0.5))
        
        self.model.add(Conv2D(256, (3,3), strides=1, padding='same'))
        self.model.add(LeakyReLU(0.5))
        
        self.model.add(Conv2D(256, (3,3), strides=1, padding='same'))
        self.model.add(LeakyReLU(0.5))

        self.model.add(Conv2D(256, (3,3), strides=1, padding='same'))
        self.model.add(LeakyReLU(0.5))
        
        self.model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2), padding='same'))
        
        self.model.add(Dropout(0.5))
  
#        self.model.add(Conv2D(512, (4,4), strides=1))
#        self.model.add(LeakyReLU(0.5))
#        self.model.add(Conv2D(512, (3,3), strides=1, 
#                              activation='relu'))
#        self.model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
#        self.model.add(Dropout(0.25))
        
        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(LeakyReLU(0.5))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512))
        self.model.add(LeakyReLU(0.5))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.net_architecture_params['num_classes'], 
                             activation='softmax',
                             activity_regularizer=l2(5e-4)))

        self.model.compile(loss='categorical_crossentropy', 
                      optimizer=Adam(lr=1e-4), 
                      metrics=['accuracy'])
      

if __name__ == '__main__':
    Workdir = "D:/AlanTan/CNN"
    os.chdir(Workdir)   
    
    training_params = {'batch_size': 32,
                        'epochs': 2,
                        'class_weight': 'auto',
                        'train_steps': 15, # train_samples_size // batch_size
                        'validation_steps': 15}
                        
    # Network architecture params
    net_architecture_params = {'num_classes': 5,
                               'input_shape': (512,512,3)}
    
    data_source_path = "D:/kaggle/detection/foldertest_transform"
    
    mymodel = VggModel_5(data_source_path,training_params,net_architecture_params)
    
    mymodel.modelDefinition()
    mymodel.modelSaveInfo('vgg_model_test.h5')
    mymodel.modelTrain()
    evaluation = mymodel.modelEvaluate()
    mymodel.modelReset()














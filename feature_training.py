# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:38:09 2017

@author: admin
"""

train_feature_right = train_feature_pivot.drop([('photo_index','left'),('each_label','left')],axis=1)
train_feature_left = train_feature_pivot.drop([('photo_index','right'),('each_label','right')],axis=1)
train_feature_left['is_left']=1
train_feature_right['is_left']=0
train_feature_left['each_lable'] = train_feature_left['each_label','left']
train_feature_right['each_lable'] = train_feature_right['each_label','right']

train_feature_blend = pd.concat([train_feature_left,train_feature_right])
train_feature_blend = train_feature_blend.drop([('each_label','left'),
                                                ('each_label','right'),
                                                'photo_index'],axis=1)
train_feature_blend_arr = train_feature_blend.dropna().values

validation_feature_right = b_pivot.drop(('each_label','left'),axis=1)
validation_feature_left = b_pivot.drop(('each_label','right'),axis=1)
validation_feature_left['is_left']=1
validation_feature_right['is_left']=0
validation_feature_left['each_lable'] = validation_feature_left['each_label','left']
validation_feature_right['each_lable'] = validation_feature_right['each_label','right']

validation_feature_blend = pd.concat([validation_feature_left,validation_feature_right])
validation_feature_blend = validation_feature_blend.drop([('each_label','left'),
                                                ('each_label','right'),
                                                ],axis=1)
validation_feature_blend_arr = validation_feature_blend.dropna().values

import os
import keras
import numpy as np
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout
from keras.layers import MaxPooling2D, AveragePooling2D, Input, Flatten,LeakyReLU
from keras.optimizers import Adam, rmsprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential

x_train = train_feature_blend_arr[:,1:]
y_train = train_feature_blend_arr[:,0]
x_validation = validation_feature_blend_arr[:,1:]
y_validation = validation_feature_blend_arr[:,0]

y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_validation = keras.utils.to_categorical(y_validation, num_classes=3)




model = Sequential()

model.add(Dense(units=4096,  input_dim=4097))
model.add(Activation('hard_sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(units=2048))
model.add(Activation('hard_sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(units=1024))
model.add(Activation('hard_sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(units=512))
model.add(Activation('hard_sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(units=3, 
                activation='softmax',
                activity_regularizer=l2(5e-4)))

model.compile(loss='categorical_crossentropy', 
                      optimizer=Adam(lr=1e-4), 
                      metrics=['accuracy']) 

save_dir = os.path.join(os.getcwd(), 'saved_models')        
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_filepath = os.path.join(save_dir, 'feature_nn.h5')
        
checkpoint = ModelCheckpoint(model_filepath,verbose=1,
                             save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-10)
callbacks = [checkpoint, lr_reducer]

model.fit(x_train, y_train ,validation_data=(x_validation,y_validation),
          epochs=200, batch_size=256,callbacks=callbacks)

loss_and_metrics = model.evaluate(x_validation, y_validation)
K.clear_session()

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier

clf_RF = RandomForestClassifier(n_estimators=50,class_weight={0: 100, 2: 0.01},n_jobs=-1)
clf_RF.fit(x_train, y_train)

preds = clf_RF.predict(x_validation)

pd.crosstab(y_validation, preds, rownames=['actual'], colnames=['preds'])

clf_BC = BaggingClassifier(n_estimators=10,n_jobs=-1)
clf_BC.fit(x_train, y_train)

preds = clf_BC.predict(x_validation)

pd.crosstab(y_validation, preds, rownames=['actual'], colnames=['preds'])

clf_ABC = AdaBoostClassifier()
clf_ABC.fit(x_train, y_train)

preds = clf_ABC.predict(x_validation)

pd.crosstab(y_validation, preds, rownames=['actual'], colnames=['preds'])

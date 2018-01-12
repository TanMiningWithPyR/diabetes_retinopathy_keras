# -*- coding: utf-8 -*-
# Change Workdir
import os
Workdir = "D:/AlanTan/CNN"
os.chdir(Workdir)

# import various packages
import numpy as np
import pandas as pd
import scipy
#import sklearn
import keras
from keras.models import Sequential
from skimage import transform 
from skimage import exposure
import time
import ShowProcess
import gc
import cv2

#import io
# Defining the function of image proprocess
def improprocess(filename):
    image = scipy.misc.imread(filename)
    x_m,y_m = image.shape[0]//2 + 1,image.shape[1]//2 + 1
    x_MiddleLine_RGBsum = image[x_m,:,:].sum(1)
    r_y = (x_MiddleLine_RGBsum>x_MiddleLine_RGBsum.mean()/10).sum()//2 + 1
    y_MiddleLine_RGBsum = image[:,y_m,:].sum(1)
    r_x = (y_MiddleLine_RGBsum>y_MiddleLine_RGBsum.mean()/10).sum()//2 + 1
    image_snip = image[x_m-r_x:x_m+r_x,y_m-r_y:y_m+r_y,:]
    image_resize = transform.resize(image_snip,(256,256))
    image_eq_hist = exposure.equalize_hist(image_resize)
    image_grey_map = cv2.addWeighted(image_eq_hist,4,
                                     cv2.GaussianBlur(image_eq_hist,(0,0),10),
                                     -4,128)
    return image_grey_map

# Defining the File Path

MixedPath="D:/kaggle/detection/train"
LabelsPath="D:/kaggle/detection/trainLabels"
TransformTrainPath="D:/kaggle/detection/TransformTrainCopy"
TransformTestPath="D:/kaggle/detection/TransformTestCopy"
#goodpath=""
#badpath=""

Mixed=os.listdir(MixedPath)
TransformTrain=os.listdir(TransformTrainPath)
TransformTest=os.listdir(TransformTestPath)
#good=os.listdir("/mnt/hdd/datasets/dogs_cats/train/dog")
#bad=os.listdir("/mnt/hdd/datasets/dogs_cats/train/dog")

# read Labels dataframe
df_Labels=pd.read_csv(os.path.join(LabelsPath,'trainLabels.csv'))
df_Labels=df_Labels.set_index(['image'])

# Resize the Train Images

# 定义进度条
# 1.在循环前定义类的实体， max_steps是总的步数
max_steps = len(Mixed)
process_bar = ShowProcess.ShowProcess(max_steps) 

for i in Mixed:
    try:        
        process_bar.show_process()      # 2.显示当前进度
        time.sleep(0.05)    
        # resizing all the images
        Image = improprocess(os.path.join(MixedPath,i))
        scipy.misc.imsave(os.path.join(TransformTrainPath,i),Image)  
    except MemoryError:
        gc.collect()
        
process_bar.close('done')            # 3.处理结束后显示消息   

# Resize the Test Images

# 定义进度条
# 1.在循环前定义类的实体， max_steps是总的步数
max_steps = len(Mixed)
process_bar = ShowProcess.ShowProcess(max_steps) 

for i in Mixed:
    try:        
        process_bar.show_process()      # 2.显示当前进度
        time.sleep(0.03)   
        if not (i in TransformTrain):
            Image = improprocess(os.path.join(MixedPath,i))
            scipy.misc.imsave(os.path.join(TransformTestPath,i),Image)  
    except MemoryError:
        gc.collect()
        
process_bar.close('done')            # 3.处理结束后显示消息   

# Load Training Images 
TrainImages=[]
TrainLabels =[]
TrainImagesName=[]

max_steps = len(TransformTrain)
process_bar = ShowProcess.ShowProcess(max_steps) 

for i in TransformTrain:
    try:        
        process_bar.show_process()      # 2.显示当前进度
        time.sleep(0.03)    
        TrainImage = scipy.misc.imread(os.path.join(TransformTrainPath,i))
        TrainImages.append(TrainImage)
        TrainImageName = i.split(".")[0]
        TrainImagesName.append(TrainImageName)
        try:
            TrainLabels.append(df_Labels.ix[TrainImageName].level)
        except KeyError:
            TrainLabels.append(-1)
    except MemoryError:
        gc.collect()
        
process_bar.close('done')            # 3.处理结束后显示消息   

# converting Training images to arrays
TrainImages=np.array(TrainImages)
TrainImagesName=np.array(TrainImagesName)
TrainLabels=np.array(TrainLabels)

# save training data to file
np.savez('TrainData.npz',
         TrainImages=TrainImages,
         TrainImagesName=TrainImagesName,
         TrainLabels=TrainLabels)

# Load Testing Images 
TestImages=[]
TestLabels =[]
TestImagesName=[]

max_steps = len(TransformTest)
process_bar = ShowProcess.ShowProcess(max_steps) 

for i in TransformTest:
    try:        
        process_bar.show_process()      # 2.显示当前进度
        time.sleep(0.03)    
        TestImage = scipy.misc.imread(os.path.join(TransformTestPath,i))
        TestImages.append(TestImage)
        TestImageName = i.split(".")[0]
        TestImagesName.append(TestImageName)
        try:
            TestLabels.append(df_Labels.ix[TestImageName].level)
        except KeyError:
            TestLabels.append(-1)
    except MemoryError:
        gc.collect()
        
process_bar.close('done')            # 3.处理结束后显示消息   

# converting Testing images to arrays
TestImages=np.array(TestImages)
TestImagesName=np.array(TestImagesName)
TestLabels=np.array(TestLabels)

# save Testing data to file
np.savez('TestData.npz',
         TestImages=TestImages,
         TestImagesName=TestImagesName,
         TestLabels=TestLabels)

#*********************** segment line ***********************
# Try a simplest model 
# Defining the hyperparameters

filters=10
filtersize=(5,5)

epochs =5
batchsize=128
input_shape=(300,300,3)

# Converting the target variable to the required size from keras.utils.np_utils import to_categorical
from keras.utils.np_utils import to_categorical
TrainLabels = to_categorical(TrainLabels)

# Defining the model

model = Sequential()

model.add(keras.layers.InputLayer(input_shape=input_shape)) # input_shape=(300,300,3)

model.add(keras.layers.convolutional.Conv2D(filters, 
	filtersize, 
	strides=(1, 1), 
	padding='valid', 
	data_format="channels_last", 
	activation='relu'))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())  # covert multidimension to one dimension array 
# Full connectivity
model.add(keras.layers.Dense(units=5,input_dim=50,activation='softmax'))
# configure the learning process
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Iterate on training data in batches
model.fit(TrainImages, TrainLabels, epochs=epochs, batch_size=batchsize,validation_split=0.3)

model.summary()



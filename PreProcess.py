# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:22:42 2017

@author: admin
"""
# Change Workdir
import os
Workdir = "D:/AlanTan/CNN"
os.chdir(Workdir)

import shutil
import gc
import pandas as pd
import time
from skimage import transform
import scipy
import ShowProcess

TransformTrainPathSource = "D:/kaggle/detection/TransformTrainCopy"
TransformTrainPath="D:/kaggle/detection/TransformTrain"
TransformTestPathSource="D:/kaggle/detection/TransformTestCopy"
TransformTestPath="D:/kaggle/detection/TransformTest"
LabelsPath="D:/kaggle/detection/trainLabels"

df_Labels=pd.read_csv(os.path.join(LabelsPath,'trainLabels.csv'))
df_Labels=df_Labels.set_index(['image'])

TransformTrain=os.listdir(TransformTrainPathSource)
TransformTest=os.listdir(TransformTestPathSource)

# move training data folder
max_steps = len(TransformTrain)
process_bar = ShowProcess.ShowProcess(max_steps) 

for i in TransformTrain:
    try:        
        process_bar.show_process()      # 2.显示当前进度
        time.sleep(0.03)   
        sourcefile = os.path.join(TransformTrainPathSource,i)
        TrainImageName = i.split(".")[0]
        targetfile = os.path.join(TransformTrainPath,
                                  str(df_Labels.ix[TrainImageName].level),
                                  i)
        shutil.copy(sourcefile,targetfile)   
    except MemoryError:
        gc.collect()
        
process_bar.close('done')            # 3.处理结束后显示消息  

# split training data and testing data 
# move to testing data folder
classes = os.listdir(TransformTrainPath)
for i in classes:
    try:       
        oneclasspath = os.path.join(TransformTrainPath,i)
        oneclassfiles = os.listdir(oneclasspath)
        oneclassfiles_test = oneclassfiles[0:int(0.25*len(oneclassfiles))]
        for j in oneclassfiles_test:
            sourcefile = os.path.join(TransformTrainPath,i,j)
            targetfile = os.path.join(TransformTestPath,
                                      i,j)
            shutil.move(sourcefile,targetfile)   
    except MemoryError:
        gc.collect()

# Improve 1,2,3,4 classes 
classes = ['1','2','3','4']
for i in classes:
    oneclasspath = os.path.join(TransformTrainPath,i)
    to_rotate_list = os.listdir(oneclasspath)
    for j in to_rotate_list:
        onefilename = os.path.join(oneclasspath,j)
        onefile = scipy.misc.imread(onefilename)
        rot_5 = transform.rotate(onefile,5) 
        rot_3 = transform.rotate(onefile,3)
        rot_357 = transform.rotate(onefile,357)
        rot_355 = transform.rotate(onefile,355)
        scipy.misc.imsave(os.path.join(oneclasspath, 
                                       j.split(".")[0] + "_rot_5.jpeg"),rot_5)
        scipy.misc.imsave(os.path.join(oneclasspath, 
                                       j.split(".")[0] + "_rot_3.jpeg"),rot_3)       
        scipy.misc.imsave(os.path.join(oneclasspath, 
                                       j.split(".")[0] + "_rot_357.jpeg"),rot_357)           
        scipy.misc.imsave(os.path.join(oneclasspath, 
                                       j.split(".")[0] + "_rot_355.jpeg"),rot_355)       


        

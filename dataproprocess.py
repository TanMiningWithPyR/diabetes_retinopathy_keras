# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:54:55 2017

@author: admin
"""
# import standard module
import os
import gc
import shutil
# import extand module
import scipy
import cv2
from skimage import transform 
# import user definination module
import showprocess

# Defining the function of image proprocess
def transformImage(filename):
    image = scipy.misc.imread(filename)
    x_m,y_m = image.shape[0]//2 + 1,image.shape[1]//2 + 1
    x_MiddleLine_RGBsum = image[x_m,:,:].sum(1)
    r_y = (x_MiddleLine_RGBsum>x_MiddleLine_RGBsum.mean()/10).sum()//2 + 1
    y_MiddleLine_RGBsum = image[:,y_m,:].sum(1)
    r_x = (y_MiddleLine_RGBsum>y_MiddleLine_RGBsum.mean()/10).sum()//2 + 1
    image_snip = image[x_m-r_x:x_m+r_x,y_m-r_y:y_m+r_y,:]
    image_resize = transform.resize(image_snip,(512,512),mode='reflect')
    image_grey_map = cv2.addWeighted(image_resize,4,
                                     cv2.GaussianBlur(image_resize,(0,0),10),
                                     -4,128)
    return image_grey_map

def rotateImage(folder,angles):
    to_rotate_list = os.listdir(folder)
    to_rotate_list_count = len(to_rotate_list)
    to_rotate_list_bar = showprocess.ProgressBar(to_rotate_list_count)
    for j in to_rotate_list:
        to_rotate_list_bar.show_process()
        onefilename = os.path.join(folder,j)
        onefile = scipy.misc.imread(onefilename)
        for angle in angles:
            rot_image = transform.rotate(onefile,angle)
            scipy.misc.imsave(os.path.join(folder, 
                                           j.split(".")[0] + "_" + str(angle) + ".jpeg"),
            rot_image)    

class DataSource:
    def __init__(self,source_path,class_list):
        self.source_path = source_path                
        self.class_list = class_list      
        self.source_transform_path = source_path + '_transform'
        self.train_path = os.path.join(self.source_transform_path,'train')
        self.test_path = os.path.join(self.source_transform_path,'test')
        self.train_class_path = [os.path.join(self.train_path,i) for i in self.class_list]
        self.test_class_path = [os.path.join(self.test_path,i) for i in self.class_list]      
          
    # If the data of different classes is mixed into one folder,
    # split them into the subfolder of classes
    def classifyData(self,label_df):  
        source_list = os.listdir(self.source_path)
        if source_list == self.class_list:
            print("Source data files have been classified with list of classes!")
        else:
            # make folder of classes
            for i in self.class_list:
                os.mkdir(os.path.join(self.source_path,i))
                
            for i in source_list:            
                try:  
                    sourcefile = os.path.join(self.source_path,i)
                    image_name = i.split(".")[0]
                    targetfile = os.path.join(self.source_path,
                                              str(label_df.ix[image_name].level),
                                              i)
                    shutil.move(sourcefile,targetfile)   
                except MemoryError:
                    gc.collect()
        
    def transformData(self):
        # make transform folder
        os.mkdir(self.source_transform_path)            
              
        source_list = os.listdir(self.source_path)
        source_count = len(source_list)
        print("There are " + str(source_count) + " classes to transform totally!" )     
        
        for i in source_list:   
            # make class folders in transform folder
            transform_class_path = os.path.join(self.source_transform_path,i)

            os.mkdir(transform_class_path)   

            source_class_path = os.path.join(self.source_path,i)
            source_class_files_list = os.listdir(source_class_path)
            source_class_count = len(source_class_files_list)
            source_class_process_bar = showprocess.ProgressBar(source_class_count)
            for j in source_class_files_list:
                try:        
                    source_class_process_bar.show_process()
                    # resizing all the images
                    image = transformImage(os.path.join(source_class_path,j))                    
                    scipy.misc.imsave(os.path.join(transform_class_path,j),image)    
                except MemoryError:
                    gc.collect()                    

        # the folders of class in transform folder
        self.transform_class_path_list = os.listdir(self.source_transform_path)
        
    def splitTrainTest(self):
    # split training data and testing data 
        # make train and test folder
        os.mkdir(self.train_path)
        os.mkdir(self.test_path) 
        class_count = len(self.transform_class_path_list)
        print("There are " + str(class_count) + " classes to split totally!" )
        
        for i in self.transform_class_path_list:            
            try:       
                # make class folder in test folder
                oneclasspath_in_test = os.path.join(self.test_path,i)
                os.mkdir(oneclasspath_in_test)
                
                oneclasspath = os.path.join(self.source_transform_path,i)
                oneclassfiles = os.listdir(oneclasspath)
                oneclassfiles_test = oneclassfiles[0:int(0.25*len(oneclassfiles))]
                oneclassfiles_test_count = len(oneclassfiles_test)
                oneclassfiles_test_process_bar = showprocess.ProgressBar(oneclassfiles_test_count)
                print("Move testing the " + str(i) + "th class data now!")
                for j in oneclassfiles_test:
                    oneclassfiles_test_process_bar.show_process()
                    sourcefile = os.path.join(oneclasspath,j)
                    targetfile = os.path.join(oneclasspath_in_test,j)
                    shutil.move(sourcefile,targetfile)                       
            except MemoryError:
                gc.collect()
                
            # move the rest files to train folder
            print("Move training the " + str(i) + "th class data now!")
            oneclasspath_in_train = os.path.join(self.train_path,i)
            shutil.move(oneclasspath,oneclasspath_in_train)
            
    def augmentImages(self):
    # counts of each class in train folder
    
        each_class_counts_list = [len(os.listdir(os.path.join(self.train_path,one_class))) \
                             for one_class in self.class_list]
        class_counts = dict(zip(self.class_list,each_class_counts_list))
        class_counts_max_value = max(each_class_counts_list)
        print("Start to augment with rotating image!" )
        for one_class in class_counts:
            if class_counts[one_class] == class_counts_max_value:
                print("no augmentation in " + one_class + " class, because it's the biggest!")
            else:
                times = class_counts_max_value // class_counts[one_class]                
                angles = [one_rotate + 1 for one_rotate in range(times)]
                rotateImage(os.path.join(self.train_path,one_class),angles)
                
if __name__ == '__main__':
    Workdir = "D:/AlanTan/CNN"
    os.chdir(Workdir)
    import dataproprocess as dp
    testdata_source = dp.DataSource("D:/kaggle/detection/foldertest",['0','1','2','3','4'])
    testdata_source.transformData()
    testdata_source.splitTrainTest()   
    testdata_source.augmentImages()         
                    
                    
                
    
import numpy as np
import cv2
import glob
from PIL import Image
import imageio
##import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import time
from numpy.linalg import matrix_rank
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.utils import np_utils
import sklearn.preprocessing as preprocessing
from keras.utils.np_utils import to_categorical
import h5py

mylist=[]
filename= 'TrainingMetadata.csv'
count=0
with open(filename) as f:
    for line in f:
        line==line.strip()
        current_line=line.split(",")
        if count !=0:
            current_line=map(float, current_line)
            
        mylist.append(current_line)
        count+=1

##print mylist
strings=mylist[0]
mylist=mylist[1:]
training_dump=np.array(mylist)
y_training=training_dump[0:,1]



images_list=[]
images_training=glob.glob("TrainingImages/*.png")

#sorting the images_list
im_num_list=[]
for img_names in images_training:
    im_num=img_names.strip(".png")
    im_num=im_num.strip("TrainingImages/")
    im_num_list.append(int(im_num))

im_num_list.sort()
images_training_new=[]
for im_number in im_num_list:
    images_training_new.append("TrainingImages/" + str(im_number) +".png")


image_stack=np.empty((0,96,96,3))


for img in images_training_new:


    im_array=mpimg.imread(img)
##    for index_1 in range(
##    ((0.0 <= a) & (a <= 0.125)).sum()

    im_array=im_array[:,:,:3]
    im_array=np.reshape(im_array,(1,96,96,3))
    image_stack=np.vstack((image_stack,im_array))    




images_list_test=[]
images_test=glob.glob("TestImages/*.png")

#sorting the images_list
im_num_list_test=[]
for img_names_test in images_test:
    im_num_test=img_names_test.strip(".png")
    im_num_test=im_num_test.strip("TestImages/")
    im_num_list_test.append(int(im_num_test))

im_num_list_test.sort()
images_training_new_test=[]
for im_number_test in im_num_list_test:
    images_training_new_test.append("TestImages/" + str(im_number_test) +".png")

image_stack_test=np.empty((0,96,96,3))


for img_test in images_training_new_test:


    im_array_test=mpimg.imread(img_test)
##    im_array_test=im_array_test[:,:,3]
    im_array_test=im_array_test[:,:,:3]


    im_array_test=np.reshape(im_array_test,(1,96,96,3))
    image_stack_test=np.vstack((image_stack_test,im_array_test))

model = Sequential([
        Convolution2D(48, 5, 5, input_shape=(96,96,3), dim_ordering="tf"),
        Activation("relu"),
        MaxPooling2D((2,2)),
        
        Convolution2D(96, 5, 5, dim_ordering="tf"),
        Activation("relu"),
        MaxPooling2D((2,2)),
    
        Flatten(),
        
        Dense(96),
        Activation("relu"),
        Dense(19),
        Activation("softmax")
    ])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights("./models/model_02.hdf5")

pred_test = model.predict(image_stack_test)
##print pred_test[0]*100
##print pred_test[1]*100

array_index=pred_test.argmax(axis=1)
y_int= y_training.astype(int)

indices=[]
count=0
max_prob_list=[]
for each_array in pred_test:
    max_prob_type=np.argsort(each_array)[-3:][::-1]
    tuple_pairs=[(max_prob_type[0],each_array[max_prob_type[0]]),(max_prob_type[1],each_array[max_prob_type[1]]),(max_prob_type[2],each_array[max_prob_type[2]])]
    max_prob_list.append(tuple_pairs)
    print "for pokemon number"
    print im_num_list_test[count]
    print tuple_pairs
    count+=1
    
    indices.append(max_prob_type)
    
indices=np.array(indices)
##print indices
##print max_prob_list

##print array_index
pred_train = model.predict(image_stack)
array_index_train=pred_train.argmax(axis=1)
print np.count_nonzero(array_index_train==y_int)/601.0



filename_1= 'UnlabeledTestMetadata.csv'
mylist_1=[]
count=0
with open(filename_1) as f_1:
    for line_1 in f_1:
        line_1=line_1.strip()
        current_line_1=line_1.split(",")
        if count !=0:
            current_line_1=map(float, current_line_1)
        mylist_1.append(current_line_1)
        count+=1
str_2=mylist_1[0]

mylist_1=mylist_1[1:]
x_test=np.array(mylist_1)





##for each_array in pred_test:
##    print

#Defining model Architecture
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import color


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

model.load_weights("./models/model_02curr_best.hdf5")

# we build a new model with the activations of the old model
# this model is truncated after the first layer
model_trunc = Sequential([
    Convolution2D(48, 5, 5, input_shape=(96,96,3), dim_ordering="tf", weights=model.layers[0].get_weights()),
    Activation("relu")
])

fig=plt.figure(figsize=(16,24))
gs = gridspec.GridSpec(12,8)
gs.update(wspace=0.025, hspace=0.05)


for idx, j in enumerate([0,3,6]):
    activations = model_trunc.predict(image_stack_test[j].reshape(1,96,96,3))
    for i in range(0,32):        
        layer_filter = activations.reshape([92,92,48])[:,:,i]
        ax = plt.subplot(gs[32*idx+i])
        plt.axis("off")  
        if i == 0:
            plt.imshow(color.hsv2rgb(image_stack_test[j]))
        else:
            
            img=plt.imshow(layer_filter,cmap="afmhot",vmin=0, vmax=1)
##            plt.show(img)
##            string_1='ting'+str(j)+'.jpg'
##            fig.savefig(string_1, dpi=3)
##            plt.savefig(string_1)

plt.show(fig)
            

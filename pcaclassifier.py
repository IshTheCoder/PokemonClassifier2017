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


t0 = time.time()

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
S=np.delete(training_dump, [1], axis=1)
##print S.shape

S_mat=np.matrix(S)
##print(matrix_rank(S_mat))

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


image_stack=np.empty((0,27648))


for img in images_training_new:


    im_array=mpimg.imread(img)
##    for index_1 in range(
##    ((0.0 <= a) & (a <= 0.125)).sum()

    im_array=im_array[:,:,:3]
    im_array=np.reshape(im_array,(1,27648))
    image_stack=np.vstack((image_stack,im_array))    

##print image_stack
pca = PCA(n_components=3)
image_stack_test=pca.fit(image_stack)

numer_array=np.array(S[:,0])
np.reshape(numer_array,(601,1))
##I_mat=np.matrix(all_im_I)
##image_list_ting=im_array.tolist()
##print image_stack.shape
##print matrix_rank(np.matrix(image_stack))
final_stack=np.column_stack((numer_array,image_stack))
##print final_stack.shape
##print matrix_rank(np.matrix(final_stack))




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

image_stack_test=np.empty((0,27648))


for img_test in images_training_new_test:


    im_array_test=mpimg.imread(img_test)
##    im_array_test=im_array_test[:,:,3]
    im_array_test=im_array_test[:,:,:3]
    im_array_test=np.reshape(im_array_test,(1,27648))


##    im_array_test=np.reshape(im_array_test,(1,27648))
    image_stack_test=np.vstack((image_stack_test,im_array_test))
pca = PCA(n_components=3)
image_stack_test=pca.fit(image_stack_test)

##final_stack_test=np.column_stack((x_test[:,0],image_stack_test))

neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(image_stack, y_training.ravel())
##print neigh.predict(image_stack_test)

clf = svm.SVC(kernel='rbf',gamma=50, C=1.0)
clf.fit(image_stack, y_training.ravel())
##SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
##    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
##    max_iter=-1, probability=False, random_state=None, shrinking=True,
##    tol=0.001, verbose=False)
##print clf.predict(image_stack_test)
lin_clf = svm.LinearSVC()
lin_clf.fit(image_stack, y_training.ravel())
##print lin_clf.predict(image_stack_test)

clf_neu = MLPClassifier(solver='lbfgs',alpha=1e-5)




clf_neu.fit(image_stack, y_training.ravel()) 

##print clf_neu.predict(image_stack_test)
clf_rand = RandomForestClassifier()
clf_rand.fit(image_stack, y_training.ravel())

clf_lind = LinearDiscriminantAnalysis()
clf_lind.fit(image_stack, y_training.ravel())
clf_quad = QuadraticDiscriminantAnalysis()
clf_quad.fit(image_stack, y_training.ravel())
clf_dt = DecisionTreeClassifier()
clf_adaboost=AdaBoostClassifier(n_estimators=100)
clf_nb = GaussianNB()
reg_lasso=linear_model.Lasso(alpha = 0.1)
##scaler = StandardScaler()
##scaler.fit(image_stack)
##image_stack = scaler.transform(image_stack)
##image_stack_test = scaler.transform(image_stack_test)  


print (cross_val_score(neigh, image_stack, y_training.ravel(), cv=5))

print sum(cross_val_score(neigh, image_stack, y_training.ravel(), cv=5))/5

print (cross_val_score(clf, image_stack, y_training.ravel(), cv=5))

print sum(cross_val_score(clf, image_stack, y_training.ravel(), cv=5))/5

print (cross_val_score(lin_clf, image_stack, y_training.ravel(), cv=5))


print sum(cross_val_score(lin_clf, image_stack, y_training.ravel(), cv=5))/5

print (cross_val_score(clf_neu, image_stack, y_training.ravel(), cv=5))


print sum(cross_val_score(clf_neu, image_stack, y_training.ravel(), cv=5))/5

print (cross_val_score(clf_rand, image_stack, y_training.ravel(), cv=5))


print sum(cross_val_score(clf_rand, image_stack, y_training.ravel(), cv=5))/5

print (cross_val_score(clf_lind, image_stack, y_training.ravel(), cv=5))

print sum(cross_val_score(clf_lind, image_stack, y_training.ravel(), cv=5))/5

print (cross_val_score(clf_quad, image_stack, y_training.ravel(), cv=5))

print sum(cross_val_score(clf_quad, image_stack, y_training.ravel(), cv=5))/5

print (cross_val_score(clf_dt, image_stack, y_training.ravel(), cv=5))

print sum(cross_val_score(clf_dt, image_stack, y_training.ravel(), cv=5))/5

print (cross_val_score(clf_adaboost, image_stack, y_training.ravel(), cv=5))

print sum(cross_val_score(clf_adaboost, image_stack, y_training.ravel(), cv=5))/5

print (cross_val_score(clf_nb, image_stack, y_training.ravel(), cv=5))

print sum(cross_val_score(clf_nb, image_stack, y_training.ravel(), cv=5))/5


new_array=clf_neu.predict(image_stack_test)
new_array.astype(int)
new_array=np.reshape(new_array,(201,1))
im_test_array=np.array(im_num_list_test)
im_test_array=np.reshape(im_test_array,(201,1))
im_test_array.astype(int)
new_list=['number','type']
header=np.array(new_list)

next_array=np.hstack((im_test_array,new_array))
next_array.astype(int)
##df = pd.DataFrame(next_next_array)
##df.to_csv("output_poke")
next_array.astype(int)
np.trunc(next_array)
next_array.astype(str)
np.set_printoptions(suppress=True)
next_next_array=np.vstack((header, next_array))

np.savetxt("final_withavg_PCA.csv", next_array, delimiter=",",fmt='%i',header='number,type')

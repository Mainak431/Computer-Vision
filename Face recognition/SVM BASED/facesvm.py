from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
from numpy import *
from sklearn import svm, metrics
from keras import optimizers
import cv2

#making the dataset to the able to cope with generic mahcine learning tools
img_rows, img_cols, img_channels = 90, 90, 1
# path1 input folder having all the images
path1 = 'C:\\Users\\Mainak\\Desktop\\Face Detection\\faces'
# path2 output folder to store all the grayscale images
path2 = 'C:\\Users\\Mainak\\Desktop\\Face Detection\\faces_resized'

listing = os.listdir(path1)
num_samples = size(listing)
print(num_samples)
# processing to resize and convert to grayscale
for file in listing:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((img_rows, img_cols))
    gray = img.convert('L')
    # need to do some more processing here
    gray.save(path2 + '\\' + file, "JPEG")

imlist = os.listdir(path2)
# sorting based on len was required to get the image in order else wrong result was generating
imlist.sort(key=len)
print((imlist))
im1 = array(Image.open(path2 + '\\' + imlist[0]))  # open one image to get size
m, n = im1.shape  # get the size of the images
imnbr = len(imlist)  # get the number of images
print(m, n, imnbr)
# array to store all flattened images
# faltten makes a 2D array in a 1D array
immatrix = array([array(Image.open(path2 + '\\' + im2)).flatten()
                  for im2 in imlist], 'f')
print(immatrix.shape)
label = np.ones((num_samples,), dtype=int)

# creating expected_output array
for i in range(num_samples):
    label[i] = floor(i / 10) + 1
print(label)

data, Label = immatrix, label
train_data = [data, Label]
# checking consistency of the dataset
rand = random.randint(0, num_samples)
train_data = [immatrix,label]
img=immatrix[rand].reshape(img_rows,img_cols)
imgtest = immatrix[rand]
plt.imshow(img,cmap='gray')
plt.show()
print(Label[rand])

print(train_data[0].shape)
print(train_data[1].shape)
X_data = immatrix / 255.0
Y = label
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.2,random_state=42)
param_C = 5
param_gamma = 0.05
classifier = svm.SVC(kernel='linear', C=param_C,gamma=param_gamma)
classifier.fit(X_train, y_train)
expected = y_test
# prediction
predicted = classifier.predict(X_test)
# classification report for different classes
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
# confusion matrix
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)
# accuracy calculation
print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)*100))

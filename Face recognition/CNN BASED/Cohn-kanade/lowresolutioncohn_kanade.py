from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
from sklearn.metrics import classification_report,confusion_matrix
from numpy import *
from keras import optimizers
import h5py
import cv2
import tensorflow as tf
import pickle
img_rows, img_cols, img_channels = 90, 90, 1
path2 = 'F:\\database\\cohn-kanade-images_resized'
imlist = os.listdir(path2)
print((imlist))
with open ('outfile', 'rb') as fp:
    itemlist = pickle.load(fp)
print(itemlist)
num_samples = len(itemlist)
im1 = array(Image.open(path2 + '\\' + imlist[0]))  # open one image to get size
m, n = im1.shape  # get the size of the images
imnbr = len(imlist)  # get the number of images
print(m, n, imnbr)
# array to store all flattened images
# faltten makes a 2D array in a 1D array
immatrix = array([array(Image.open(path2 + '\\' + im2)).flatten()
                  for im2 in imlist], 'f')
print(immatrix.shape)
label = np.ones((num_samples,),dtype=int)

# creating expected_output array
count = 0
for i in range(num_samples):
    if i == 0:
        label[i] = 1
        count += 1
    elif itemlist[i] == itemlist[i - 1] :
        label[i] = count
    else :
        label[i] = count + 1
        count += 1
print(label)

data, Label = immatrix, label
train_data = [data, Label]
# checking consistency of the dataset
rand = random.randint(0, num_samples)
train_data = [immatrix,label]

print(Label[rand])

print(train_data[0].shape)
print(train_data[1].shape)
# batch_size to train
batch_size = 20
# number of output classes
nb_classes = len(set(itemlist))
# number of epochs to train
nb_epoch = 20

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 2
# X input image data y expected class data
(X, y) = (train_data[0], train_data[1])

# splitting the testing data as 20% of the total data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalization of pixel values
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vector to binary class matrices
Y_train = np_utils.to_categorical(y_train - 1, nb_classes)
Y_test = np_utils.to_categorical(y_test - 1, nb_classes)
i = 0
print("label : ", Y_train[i, :])
#CNN architecture design
model = Sequential()
model.add(Conv2D(2*nb_filters, (nb_conv, nb_conv), padding='same', activation='tanh', input_shape=(img_rows, img_cols,1)))
model.add(Conv2D(2*nb_filters, (nb_conv, nb_conv), activation='tanh'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Conv2D(2*nb_filters, (nb_conv, nb_conv), padding='same', activation='tanh'))
model.add(Conv2D(2*nb_filters, (nb_conv, nb_conv), activation='tanh'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Conv2D(2*nb_filters, (nb_conv, nb_conv), padding='same', activation='tanh'))
model.add(Conv2D(2*nb_filters, (nb_conv, nb_conv), activation='tanh'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adadelta' ,metrics=['accuracy'])

#hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs= nb_epoch,validation_data = (X_test,Y_test),
             verbose=1)


#visualizing loss and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available)# use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()
#score

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])


Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

fname = "weights-Test-CNN.hdf5"
model.save_weights(fname,overwrite=True)

fname = "weights-Test-CNN.hdf5"
model.load_weights(fname)

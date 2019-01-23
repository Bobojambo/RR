# -*- coding: utf-8 -*-
"""
This file is used to generate and train the model to classify image to water or
not water.
"""

#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.layers.core import Dense, Flatten
from keras import Sequential
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn import model_selection
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

import numpy as np
import glob
import cv2 as cv


#Function to generate the model
def generate_model(shape, number_of_classes=2):
    
    model = Sequential()
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_shape=shape, classes=1000)
    
    model.add(resnet50)
    model.add(Flatten())
    model.add(Dense(number_of_classes, activation='softmax'))
    
    model.summary()

    optimizer = Adam()
    model.compile(optimizer = optimizer,
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])
    
    return model

"""
Function to train the model with desired hyperparameters
"""
def train_model(images, targets, model, lb):
    
    labels_binarized = lb.transform(targets)

    #model implementation doesnt work without "if class amount == 2"
    labels_binarized = to_categorical(labels_binarized, num_classes=2) 
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(images, labels_binarized, random_state = 42)
    
    callbacks_list = []
    batch_size = 32
    n_epochs = 10
    
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list.append(checkpoint)
    
    history = model.fit(X_train, y_train, epochs = n_epochs, batch_size = batch_size, callbacks = callbacks_list, 
        validation_data = (X_test, y_test))
    
    return history

def binarize_targets(targets):
    
    lb = LabelBinarizer()
    lb.fit(targets)
    
    return lb

"""
Data used for training the classifier
globPath = path to images of simulation water images
test_data, used to load only first 500 images to test training
"""
def load_image_paths(globPath, target, test_data = False):
    
    image_files = []
    for filename in glob.glob(globPath):
        image_files.append(filename)
    
    filepaths = []
    targets = []
    for file in image_files:
        filepaths.append(file)
        targets.append(target)
        
    #For testing 500 first loaded images  
    if test_data == True:
        filepaths = filepaths[:5000]
        targets = targets[:5000]    
      
    return filepaths, targets

def load_and_resize_images_from_paths(filepaths):
    
    images = []
    for imagepath in filepaths:
        img = cv.imread(imagepath)
        if img.shape[0] is not 224:
            img = cv.resize(img, (224,224), interpolation = cv.INTER_LINEAR)
            img = np.array(img)
        images.append(img)
        
    return images


if __name__ == "__main__":
    print("do nothing")
    
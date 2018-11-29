# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:18:43 2018

@author: TaitavaBobo§
"""

from keras.applications.resnet50 import ResNet50
from keras import Sequential
import cv2 as cv
import numpy as np
from PIL import Image
from keras.applications.imagenet_utils import decode_predictions


def generate_model(number_of_classes):
    
    model = Sequential()
    resNet50base = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    model.add(resNet50base)
    #model.add(Flatten())
    #model.add(Dense(number_of_classes, activation='softmax'))
    
    return model

def train_model(model):
    
    return model


def predict_image(model, image):
    
    image = resize_image(image)   
    predictions = model.predict(image)
    # convert the probabilities to class labels
    # We will get top 5 predictions which is the default
    max_predictions_list = decode_predictions(predictions, top=1)[0]    
    return max_predictions_list

def resize_image(image):
    
    #Get width and height from model
    width, height = 224
    image = image.resize((width, height), Image.BILINEAR)
    numpy_image = np.array(image)
    numpy_image = numpy_image[np.newaxis, ...]
    return numpy_image

def add_mask(image, color):
    
    mask = np.zeros_like(image)
    height, width, channels = image.shape
    if color == "Green":
        mask[0:0, 0:0] = [1, 0, 0]  # Red block
        mask[0:height, 0:width] = [0, 255, 0] # Green block
        mask[0:0, 0:0] = [0, 0, 1] # Blue block
    
    elif color == "Red":
        mask[0:height, 0:width] = [1, 255, 0]  # Red block
        mask[0:0, 0:0] = [0, 1, 0] # Green block
        mask[0:0, 0:0] = [0, 0, 1] # Blue block
  
    image = cv.addWeighted(image,0.7,mask,0.3,0)
    
    return image

def split_image_to_grid(image, gridsize = 2):
    
    height, width, channels = image.shape
    image_height = int(height/gridsize)
    image_width = int(width/gridsize)
    
    height_tracker = 0
    width_tracker = 0
    
    while True:
        
        cropped_image = image[height_tracker*image_height : height_tracker*image_height+image_height,
                              width_tracker*image_width : width_tracker*image_width+image_width]
        image_name = "{}{}.jpg".format(height_tracker, width_tracker)
        cv.imwrite(image_name, cropped_image)
        height_tracker += 1
        if height_tracker == gridsize:
            height_tracker = 0
            width_tracker += 1
            if width_tracker == gridsize:
                break
       
    return


def merge_image(grid_of_images):
    #merge parts of image together
    return image

if __name__ == "__main__":
    #print("do nothing")
    number_of_classes = 2
    model = generate_model(number_of_classes)
        
    image = cv.imread("1765120.jpg")
    masked_image = add_mask(image, color = "Red")
    
    split_image_to_grid(image, 3)
    
    #stacked_image = np.concatenate((masked_image, masked_image), axis=1)
    #cv.imwrite('out.png', stacked_image)
    
    
    
    #merge = np.array(image)
    #mergePIL = Image.fromarray(merge)
    #mergePIL.show()
  
    



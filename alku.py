# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:18:43 2018

@author: TaitavaBobo§
"""

from keras.applications.resnet50 import ResNet50
from keras import Sequential
import cv2 as cv
import numpy as np
from keras.applications.imagenet_utils import decode_predictions


def generate_model(number_of_classes=2):
    
    model = Sequential()
    resNet50base = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    model.add(resNet50base)
    #model.add(Flatten())
    #model.add(Dense(number_of_classes, activation='softmax'))
    
    return model

def predict_image(image):
    
    image = resize_image(image)   
    #predictions = model.predict(image)
    # convert the probabilities to class labels
    # We will get top 5 predictions which is the default
    """
    max_predictions_list = decode_predictions(predictions, top=1)[0]
    """
    prediction = True
    return prediction

def resize_image(image):
    
    resized_image = cv.resize(image, (224,224), interpolation = cv.INTER_LINEAR)
    numpy_image = np.array(resized_image)
    numpy_image = numpy_image[np.newaxis, ...]
    
    return numpy_image
  
#Add mask to an image. If prediction is True, add green, else, red
def add_mask(image, prediction):
    
    mask = np.zeros_like(image)
    height, width, channels = image.shape
    if prediction is True:
        mask[0:0, 0:0] = [1, 0, 0]  # Red block
        mask[0:height, 0:width] = [0, 255, 0] # Green block
        mask[0:0, 0:0] = [0, 0, 1] # Blue block
    
    else:
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
    
    cropped_images = []
    
    while True:
        
        cropped_image = image[height_tracker*image_height : height_tracker*image_height+image_height,
                              width_tracker*image_width : width_tracker*image_width+image_width]
        cropped_images.append(cropped_image)
                
        #Classify image for water or not
        #Add mask to the part of the image
        prediction = predict_image(cropped_image)
        cropped_image = add_mask(cropped_image, prediction)
        
        if prediction is True:
            mask = np.zeros_like(image)
            height, width, channels = image.shape
            mask[height_tracker*image_height : height_tracker*image_height+image_height,
                 width_tracker*image_width : width_tracker*image_width+image_width] = [0,255,0]
            masked_full_image = cv.addWeighted(image,0.7,mask,0.3,0)
        
        
        """
        #Save crops to folder        
        image_name = "{}{}.jpg".format(height_tracker, width_tracker)
        cv.imwrite(image_name, masked_full_image)
        """

        
        #Next part of the image        
        height_tracker += 1
        if height_tracker == gridsize:
            height_tracker = 0
            width_tracker += 1
            if width_tracker == gridsize:
                break
        
        #For testing
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
    #masked_image = add_mask(image, color = "Red")
    
    split_image_to_grid(image, 3)
    
    #stacked_image = np.concatenate((masked_image, masked_image), axis=1)
    #cv.imwrite('out.png', stacked_image)
    
    
    
    #merge = np.array(image)
    #mergePIL = Image.fromarray(merge)
    #mergePIL.show()
  
    



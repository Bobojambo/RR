# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:18:43 2018

@author: TaitavaBobo§
"""

import cv2 as cv
import numpy as np
import classification_model


"""
Function for image resizing. Function needs to be modified in order to use
different resize methods
"""
def resize_image(image, resize_resolution = (224,224)):
    
    resized_image = cv.resize(image, resize_resolution, interpolation = cv.INTER_LINEAR)
    numpy_image = np.array(resized_image)
    numpy_image = numpy_image[np.newaxis, ...]
    
    return numpy_image

"""
Function to split frames of a video into gridframes with desired gridsize.
E.g. gridsize = 2 generates 2x2 grid with 4 different subimages
"""
def split_and_predict_grid_images(model, image, gridsize = 4):
    
    #Get image shape
    height, width, channels = image.shape
    
    #Single grid height and width for editing
    image_height = int(height/gridsize)
    image_width = int(width/gridsize)
    
    #Trackers for keeping track of the crop
    height_tracker = 0
    width_tracker = 0
    
    mask = np.zeros_like(image)

    while True:
        
        #Crop part of the image for editing
        cropped_image = image[height_tracker*image_height : height_tracker*image_height+image_height,
                              width_tracker*image_width : width_tracker*image_width+image_width]
                
        #Boolean prediction, resize image to (224,224) to match ResNet50 input
        prediction = model.predict(resize_image(cropped_image))
        #0 = water, 1 = other
        if prediction[0][0] > 0.5:
            prediction = True
        
        #Rectangle
        if prediction is True:
            cv.rectangle(mask,(width_tracker*image_width, height_tracker*image_height),
                         (width_tracker*image_width+image_width-5, height_tracker*image_height+image_height-5),
                         (0,255,0)
                         ,3)
        else:
            cv.rectangle(mask,(width_tracker*image_width, height_tracker*image_height),
             (width_tracker*image_width+image_width-5, height_tracker*image_height+image_height-5),
             (0,0,255)
             ,3)  
                   
        #Tracker increase in order to handle full image
        height_tracker += 1
        if height_tracker == gridsize:
            height_tracker = 0
            width_tracker += 1
            if width_tracker == gridsize:
                break
              
    masked_full_image = cv.addWeighted(image,0.85,mask,0.15,0)
    
    """
    #For testing
    #Save crops to folder  
    image_name = "{}{}.jpg".format(height_tracker, width_tracker)
    cv.imwrite(image_name, masked_full_image)
    return masked_full_image
    """

if __name__ == "__main__":
    #print("do nothing")        
    image = cv.imread("1765120.jpg")
    model = classification_model.generate_model((224,224,3))
    model.load_weights("weights.best.hdf5")
    image = split_and_predict_grid_images(model, image, 4)
    predictions = model.predict(resize_image(image))

    



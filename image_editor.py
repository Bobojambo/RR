# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:18:43 2018

@author: TaitavaBobo§
"""

import cv2 as cv
import numpy as np
import random


def predict_image(image):
    
    image = resize_image(image)   
    
    roll = random.randint(1,100)
    if roll < 50:
        return True
    else:
        return False

def resize_image(image):
    
    resized_image = cv.resize(image, (224,224), interpolation = cv.INTER_LINEAR)
    numpy_image = np.array(resized_image)
    numpy_image = numpy_image[np.newaxis, ...]
    
    return numpy_image

def split_image_to_grid(image, gridsize = 2):
    
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
                
        #Classify image for water or not
        #Add mask to the part of the image
        prediction = predict_image(cropped_image)
        #cropped_image = add_mask(cropped_image, prediction)
        
        if prediction is True:
            #mask[0:0, 0:0] = [1, 0, 0]  # Red block
            mask[height_tracker*image_height : height_tracker*image_height+image_height,
                 width_tracker*image_width : width_tracker*image_width+image_width] = [0, 255, 0] # Green block
            #mask[0:0, 0:0] = [0, 0, 1] # Blue block
            
        else:
            mask[height_tracker*image_height : height_tracker*image_height+image_height,
                 width_tracker*image_width : width_tracker*image_width+image_width] = [0, 0, 255]  # Red block
            #mask[0:0, 0:0] = [0, 1, 0] # Green block
            #mask[0:0, 0:0] = [0, 0, 1] # Blue block
                
                   
        #Tracker increase in order to handle full image
        height_tracker += 1
        if height_tracker == gridsize:
            height_tracker = 0
            width_tracker += 1
            if width_tracker == gridsize:
                break
              
    masked_full_image = cv.addWeighted(image,0.7,mask,0.3,0)
    
    """
    #For testing
    #Save crops to folder        
    image_name = "{}{}.jpg".format(height_tracker, width_tracker)
    cv.imwrite(image_name, masked_full_image)
    """
    return masked_full_image


if __name__ == "__main__":
    #print("do nothing")        
    image = cv.imread("1765120.jpg")    
    image = split_image_to_grid(image, 3)

    



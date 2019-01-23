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
def split_and_predict_grid_images(model, image, gridsize = 2, label_binarizer = None, testing = False):
    
    #Get image shape
    height, width, channels = image.shape
    
    #Single grid height and width for editing
    image_height = int(height/gridsize)
    image_width = int(width/gridsize)
    
    #Trackers for keeping track of the crop
    height_tracker = 0
    width_tracker = 0
    
    while True:
        
        #Crop part of the image for editing
        cropped_image = image[height_tracker*image_height : height_tracker*image_height+image_height,
                              width_tracker*image_width : width_tracker*image_width+image_width]
                
        #Boolean prediction, resize image to (224,224) to match ResNet50 input
        if testing is True:
            prediction = True
        prediction = model.predict(resize_image(cropped_image))
        #0 = water, 1 = other
        if prediction[0][0] > 0.5:
            prediction = True
            
        #Rectangle
        if prediction is True:
            #Rectangle (img, (x,y), (x+w, y+h))
            cv.rectangle(image,(width_tracker*image_width, height_tracker*image_height),
                         (width_tracker*image_width+image_width-5, height_tracker*image_height+image_height-5),
                         (0,255,0)
                         ,1)
                    #prediction percentage
            #text = prediction[0][0]
            text = "water"
            #text = "{}{}".format("water", str(prediction([0][0])))
            cv.putText(image, text, (width_tracker*image_width+int(image_width/4), height_tracker*image_height+int(image_height/2)),
                           cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), lineType=cv.LINE_AA) 
        else:
            cv.rectangle(image,(width_tracker*image_width, height_tracker*image_height),
             (width_tracker*image_width+image_width-5, height_tracker*image_height+image_height-5),
             (0,0,255)
             ,1)  
            

                   
        #Tracker increase in order to handle full image
        height_tracker += 1
        if height_tracker == gridsize:
            height_tracker = 0
            width_tracker += 1
            if width_tracker == gridsize:
                break
                  
    #For testing
    #Save crops to folder 
    if testing is True:        
        #image_name = "{}{}.jpg".format(height_tracker, width_tracker)
        image_name = "testi.jpg".format(height_tracker, width_tracker)
        cv.imwrite(image_name, image)        
    
    return image

if __name__ == "__main__":
    #print("do nothing")        
    image = cv.imread("1765120.jpg")
    testing = True
    
    model = classification_model.generate_model((224,224,3))
    model.load_weights("weights.best.hdf5")
    
    image = split_and_predict_grid_images(model, image, 2, testing)
    #predictions = model.predict(resize_image(image))

    



# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:18:43 2018

@author: TaitavaBobo§
"""

import cv2 as cv
import numpy as np
import classification_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder


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
def split_and_predict_grid_images(model, image, gridsize = 2, first_call = False, testing = False, label_binarizer = None,):
    
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
   
        
        if height_tracker == 1 and width_tracker == 1 and first_call is True:
            split_and_predict_grid_images(model, cropped_image, 5)

        if height_tracker == 2 and width_tracker == 1 and first_call is True:
            split_and_predict_grid_images(model, cropped_image, 5)
        
        if height_tracker == 0 and width_tracker == 1 and first_call is True:
            split_and_predict_grid_images(model, cropped_image, 5)
        
        #Boolean prediction, resize image to (224,224) to match ResNet50 input
        #if testing is True:
        #    prediction = True
        prediction = model.predict(resize_image(cropped_image))
        classification_threshold = 0.8

        # Water > threshold
        if prediction[0][1] > classification_threshold:
            prediction = True
            #Rectangle (img, (x,y), (x+w, y+h))
            cv.rectangle(image,(width_tracker*image_width, height_tracker*image_height),
                         (width_tracker*image_width+image_width-1, height_tracker*image_height+image_height-1),
                         (0,255,0)
                         ,1)
                    #prediction percentage
            text = " "
            cv.putText(image, text, (width_tracker*image_width+int(image_width/4), height_tracker*image_height+int(image_height/2)),
                           cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), lineType=cv.LINE_AA) 

        # Other > threshold
        elif prediction[0][0] > classification_threshold:
            prediction = False
            cv.rectangle(image,(width_tracker*image_width, height_tracker*image_height),
             (width_tracker*image_width+image_width-1, height_tracker*image_height+image_height-1),
             (0,0,255)
             ,1)

        else:
            cv.rectangle(image,(width_tracker*image_width, height_tracker*image_height),
             (width_tracker*image_width+image_width-1, height_tracker*image_height+image_height-1),
             (255,0,0)
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
    #image = cv.imread("62.jpg")
    #image = water_image
    testing = True
    first_call = True

    model = classification_model.generate_model((224,224,3))
    model.load_weights("weights.mobilenetAdam.hdf5")

    #Water images
    target = "water"
    path = 'water_images/*.jpg'
    waterimagepaths, water_targets = classification_model.load_image_paths(path, target)
    water_images = classification_model.load_and_resize_images_from_paths(waterimagepaths)
    
    #Other images
    target = "other"
    path = 'ResizedImages224x224/*.jpg'
    other_images_paths, other_targets = classification_model.load_image_paths(path, target)
    other_images = classification_model.load_and_resize_images_from_paths(other_images_paths)
    
    #Join the images
    images = water_images + other_images
    targets = water_targets + other_targets    
    images = np.array(images)

    image = split_and_predict_grid_images(model, image, 3, first_call, testing)
    #predictions = model.predict(resize_image(image))

"""
    #binarize targets
    lb = classification_model.binarize_targets(targets)

    lb = LabelBinarizer()
    lb.classes_ = np.load('classes.npy')
    print()
 
    testi = lb.inverse_transform(model.predict(image))
    print(testi)
   """
"""
    image = resize_image(image)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    print(prediction)
    print(prediction[0][0])
    print(prediction[0][1])
"""
    



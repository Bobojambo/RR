# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:48:17 2019

@author: TaitavaBobo§
"""

import image_editor
import classification_model
import numpy as np
import cv2

def classify_video_frames(model, gridsize, output_filename, label_binarizer=None):
    
    capture = cv2.VideoCapture('Videos/patka2.mp4')
    
    # Check if camera opened successfully
    if (capture.isOpened()== False): 
        print("Error opening video stream or file")
        
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
     
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_filename,fourcc, 10.0, (frame_width,frame_height))
    
    # Read until video is completed
    while(capture.isOpened()):
      # Capture frame-by-frame
      ret, frame = capture.read()
      if ret == True:
     
        # Display the resulting frame
        cv2.imshow('Frame',frame)
        
        #Edit the frame according to
        edited_frame = image_editor.split_and_predict_grid_images(model, frame, gridsize)
        #edited_frame = frame
        # write the flipped frame
        out.write(edited_frame)        
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
     
      # Break the loop
      else: 
        break
    
    # When everything done, release the video capture and video write objects

    capture.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    
    return

def main():
    input_str = str(input('Train or Load model:'))
    print(input_str)
    
    try:
    
        if input_str == "train":      
            
            test = True
    
            #Water images
            target = "water"
            path = 'water_images/*.jpg'
            waterimagepaths, water_targets = classification_model.load_image_paths(path, target, test)
            water_images = classification_model.load_and_resize_images_from_paths(waterimagepaths)
            
            #Other images
            target = "other"
            path = 'ResizedImages224x224/*.jpg'
            other_images_paths, other_targets = classification_model.load_image_paths(path, target, test)
            other_images = classification_model.load_and_resize_images_from_paths(other_images_paths)
            
            #Join the images
            images = water_images + other_images
            targets = water_targets + other_targets    
            images = np.array(images)
        
            #Generate model
            shape = images[0].shape    
            model = classification_model.generate_model(shape)
            
            #binarize targets
            label_binarizer = classification_model.binarize_targets(targets)
            
            #Train the model
            history = classification_model.train_model(images, targets, model, label_binarizer)
            
        elif input_str == "load":
            
            model = classification_model.generate_model((224,224,3))
            try:
                model.load_weights("weights.best.hdf5")
            except:
                print("no weights found")
        
        while True:
            input_str = str(input('Classify video? (y/n):'))       
            
            if input_str is 'y':
                
                output_filename = str(input('output filename: '))
                gridsize = int(input('gridsize: '))
                if output_filename is '':
                    output_filename = 'output.avi'
                    
                classify_video_frames(model, gridsize, output_filename)
            
            else:
                break
            
    except:
        print("unexpected error")
    
    return

if __name__ == "__main__":
    main()
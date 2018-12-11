# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 13:17:19 2018

@author: TaitavaBobo§
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import image_editor

#CV2 BGR, not RGB

def main():
    read_and_write_a_video()
    
"""
SOURCE
https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
YOUTUBE-videos
https://www.youtube.com/watch?v=U6uIrq2eh_o&index=3&list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq
"""

def read_and_write_a_video():
    
    capture = cv2.VideoCapture('esimerkki.mp4')
    
    # Check if camera opened successfully
    if (capture.isOpened()== False): 
        print("Error opening video stream or file")
        
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
     
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 10, (frame_width,frame_height))    
    # Read until video is completed
    while(capture.isOpened()):
      # Capture frame-by-frame
      ret, frame = capture.read()
      if ret == True:
     
        # Display the resulting frame
        cv2.imshow('Frame',frame)
        
        #Edit the frame according to 
        frame = image_editor.split_image_to_grid(frame)
        
        # write the flipped frame
        out.write(frame)        
        
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

if __name__ == "__main__":
    main()
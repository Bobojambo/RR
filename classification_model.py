# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:55:11 2018

@author: TaitavaBobo§
"""

from keras.applications.imagenet_utils import decode_predictions



def generate_model(number_of_classes=2):
    
    model = Sequential()
    resNet50base = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    model.add(resNet50base)
    #model.add(Flatten())
    #model.add(Dense(number_of_classes, activation='softmax'))
    
    return model

def main():
    
    return

if __name__ == "__main__":
    print("Do nothing")
"""
In this file, the semantic and color simulator images are handled. The script
will go the images through one by one and checks if the semantic image contains
water (100%) and if it does the script saves the corresponding color image of 
the semantic image segment.

Gridsize can be used to specify the checked/saved images. By default gridsize = 3 (3x3)
and thus the images are checked in 9 different locations.
"""

import glob
import os
import cv2 as cv


def extract_images(semantic_image_path, color_image_path, index, gridsize):

    semantic_image = cv.imread(semantic_image_path)
    color_image = cv.imread(color_image_path)
    height, width, channels = semantic_image.shape

    # Single grid height and width for editing
    image_height = int(height/gridsize)
    image_width = int(width/gridsize)

    # Trackers for keeping track of the crop
    height_tracker = 0
    width_tracker = 0
    water_boolean = None

    while True:

        # Crop part of the image for editing
        semantic_cropped_image = semantic_image[height_tracker*image_height:height_tracker*image_height+image_height,
                                                width_tracker*image_width:width_tracker*image_width+image_width]

        water_boolean = check_if_semantic_image_is_water(semantic_cropped_image)

        # Save corresponding part of color image
        if water_boolean is True:
            color_cropped_image = color_image[height_tracker*image_height:height_tracker*image_height+image_height,
                                              width_tracker*image_width:width_tracker*image_width+image_width]       
            image_name = "water_images/{}.jpg".format(index)
            cv.imwrite(image_name, color_cropped_image)
            index += 1

        # Tracker increase in order to handle full image
        height_tracker += 1
        if height_tracker == gridsize:
            height_tracker = 0
            width_tracker += 1
            if width_tracker == gridsize:
                break

    return index


# Checks every pixel in the image if it is the color of the water
# The algorithm could be speed up with numpy arrays
def check_if_semantic_image_is_water(image):

    water_pixel_color = "132154147"
    # height and width of image
    h = image.shape[0]
    w = image.shape[1]

    prediction = True
    # Go through every pixel of the image
    for y in range(0, h):
        for x in range(0, w):

            # change the pixel value to string
            image_pixel_color = str(image[y, x][0])+str(image[y, x][1])+str(image[y, x][2])

            # Return if faulty pixel is found
            if image_pixel_color != water_pixel_color:
                print("not matching pixel")
                prediction = False
                return prediction

    return prediction


def main():

    folder = "water_images/"
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    # path for data
    GT_image_paths = glob.glob("data/simulator_images/images/*GT.png")
    HD_image_paths = glob.glob("data/simulator_images/images/*HD.png")

    # Index for image naming
    index = 0
    gridsize = 3
    for semantic_image, color_image in zip(GT_image_paths, HD_image_paths):
        index = extract_images(semantic_image, color_image, index, gridsize)

    return


if __name__ == "__main__":
    main()

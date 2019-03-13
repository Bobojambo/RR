import xml.etree.ElementTree as ET
import glob
import sys
import os
import shutil
import csv
import cv2


def create_fullImage_dict():

    # Test or full path
    images_path = 'data/real_images/images/*.jpg'
    # images_path = 'data/real_images/images_example/*.jpg'

    images_dict = {}

    for filename in glob.glob(images_path):

        first_split = filename.split('.')
        without_jpg_string = first_split[0]
        image_number_string = without_jpg_string.split('\\')[-1]
        images_dict[image_number_string] = filename

    return images_dict


def get_sub_images(images_dict, resize_argument, test_images_argument):

    images = []
    classes = []

    # Full or test path
    if test_images_argument == "Y":
        xml_path = 'data/real_images/annotations_example/*.xml'
    elif test_images_argument == "N":
        xml_path = 'data/real_images/annotations/*.xml'
    else:
        xml_path = "Empty"

    image_index = 0
    for filename in glob.glob(xml_path):

        # Select image based on xml-file name
        first_split = filename.split('.')
        without_xml_string = first_split[0]
        xml_string = without_xml_string.split('\\')[-1]
        imageString = xml_string.split('-')[0]  # Without ending -xxxxxx -part
        img_filename = images_dict[imageString]  # Search from previously created dict

        img = cv2.imread(img_filename)

        # Get objects and bounding boxes from XML-files
        tree = ET.parse(filename)

        # node is the text between <object> </object>
        for node in tree.iter('object'):

            # Set 0 for error handling
            xmin = 0
            ymin = 0
            xmax = 0
            ymax = 0

            # iterates over all elements in object-node
            for elem in node.iter():

                if not elem.tag == node.tag:
                    # Prints all elements of the xml tag
                    # print("{}: {}".format(elem.tag, elem.text))
                    if elem.tag == "name":
                        object_name = elem.text
                    if elem.tag == "xmin":
                        xmin = int(elem.text)
                    if elem.tag == "ymin":
                        ymin = int(elem.text)
                    if elem.tag == "xmax":
                        xmax = int(elem.text)
                    if elem.tag == "ymax":
                        ymax = int(elem.text)

            if (xmin + ymin + xmax + ymax) == 0:
                print("No full bounding box on object")
                sys.exit("No full bounding on object")

            # Extract subimage from img with bounding box
            else:
                # Make the bounding box area square shaped to retain the aspect ratio
                if resize_argument == "True":
                    # print("x_max, x_min, x_distance: ", xmax," ", xmin , " ", xmax-xmin)
                    # print("y_max, y_min, y_distance: ", ymax," ", ymin , " ", ymax-ymin)
                    y_gap = ymax - ymin
                    x_gap = xmax - xmin
                    if x_gap > y_gap:
                        pixels_to_add = int(round((x_gap - y_gap)/2))
                        ymin = ymin - pixels_to_add
                        ymax = ymax + pixels_to_add
                        y_gap = ymax - ymin
                        if y_gap - x_gap > 0:
                            ymax -= 1
                        if y_gap - x_gap < 0:
                            ymax += 1
                    else:
                        pixels_to_add = int(round((y_gap - x_gap)/2))
                        xmin = xmin - pixels_to_add
                        xmax = xmax + pixels_to_add
                        x_gap = xmax - xmin
                        if x_gap - y_gap > 0:
                            xmax -= 1
                        if x_gap - y_gap < 0:
                            xmax += 1

                    # Extra pixels to deny black borders with the real reflected image
                    extra = 400
                    reflect_image = cv2.copyMakeBorder(img, extra, extra, extra, extra, cv2.BORDER_REFLECT_101)
                    img_cropped = reflect_image[(ymin + extra):(ymax + extra), (xmin + extra):(xmax + extra)]

                    images.append(img_cropped)
                    classes.append(object_name)

                else:
                    # The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
                    # print("xmin: ", xmin, " ymin: ", ymin, " xmax: ", xmax, " ymax: ", ymax)
                    # print(xmin,ymax,xmax-xmin,ymax-ymin)

                    img_cropped = img[ymin:ymax, xmin:xmax]
                    images.append(img_cropped)
                    classes.append(object_name)

        image_index = resize_and_save_image_batch(images, image_index)
        images = []

    return images, classes


def resize_and_save_image_batch(images, index, width=128, height=128, path="ResizedImages/"):

    paths = ["ResizedImages32x32/", "ResizedImages64x64/", "ResizedImages96x96/", "ResizedImages128x128/", "ResizedImages224x224/"]

    if index == 0:
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

    for image in images:

        resized_image = cv2.resize(image, ((32, 32)))
        cv2.imwrite("{}image{}.jpg".format("ResizedImages32x32/", index), resized_image)

        resized_image = cv2.resize(image, ((64, 64)))
        cv2.imwrite("{}image{}.jpg".format("ResizedImages64x64/", index), resized_image)

        resized_image = cv2.resize(image, ((128, 128)))
        cv2.imwrite("{}image{}.jpg".format("ResizedImages128x128/", index), resized_image)

        resized_image = cv2.resize(image, ((224, 224)))
        cv2.imwrite("{}image{}.jpg".format("ResizedImages224x224/", index), resized_image)

        index += 1

    return index


def create_classes_csv(classes):

    path = "Classes/"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    with open("Classes/classes.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        index = 0
        for line in classes:
            writer.writerow([index, line])
            index += 1

    return


if __name__ == "__main__":

    input_var = input("Square subimage extraction (True/False): ")
    argument = str(input_var)
    input_var = str(input("Test images run (Y/N): "))
    test_images_argument = input_var

    if argument == "True" or argument == "False":
        images_dict = create_fullImage_dict()

        print("Sub images")
        images, classes = get_sub_images(images_dict, argument, test_images_argument)

        # print("Parent list and dictionaries")
        # parent_list, dictionaries_list = class_extractor.return_class_dictionaries()

        print("Classes csv")
        create_classes_csv(classes)

    else:
        print("Wrong input argument")

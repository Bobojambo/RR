import cv2 as cv
import numpy as np
import classification_model


def resize_image(image, resize_resolution=(224, 224)):

    resized_image = cv.resize(image, resize_resolution, interpolation=cv.INTER_LINEAR)
    numpy_image = np.array(resized_image)
    numpy_image = numpy_image[np.newaxis, ...]

    return numpy_image


# Function to split frames of a video into gridframes with desired gridsize.
# E.g. gridsize = 2 generates 2x2 grid with 4 different subimages
def split_and_predict_grid_images(model, image, gridsize=2, first_call=False, testing=False, label_binarizer=None):

    # Threshold used for image classification to water or other class.
    # Can be varied in order to achieve different TPR and FPR rates.
    classification_threshold = 0.8

    # Get image shape
    height, width, channels = image.shape

    # Single grid height and width for editing
    image_height = int(height/gridsize)
    image_width = int(width/gridsize)

    # Trackers for keeping track of the crop
    height_tracker = 0
    width_tracker = 0

    while True:
        # Crop part of the image for editing
        cropped_image = image[height_tracker*image_height:height_tracker*image_height+image_height,
                              width_tracker*image_width:width_tracker*image_width+image_width]

        # Smaller area testing
        if height_tracker == 1 and width_tracker == 1 and first_call is True:
            split_and_predict_grid_images(model, cropped_image, 5)
        if height_tracker == 2 and width_tracker == 1 and first_call is True:
            split_and_predict_grid_images(model, cropped_image, 5)
        if height_tracker == 0 and width_tracker == 1 and first_call is True:
            split_and_predict_grid_images(model, cropped_image, 5)

        prediction = model.predict(resize_image(cropped_image))
        # Water > threshold
        if prediction[0][1] > classification_threshold:
            prediction = True
            # Rectangle (img, (x,y), (x+w, y+h))
            cv.rectangle(image, (width_tracker*image_width, height_tracker*image_height),
                         (width_tracker*image_width+image_width-1, height_tracker*image_height+image_height-1),
                         (0, 255, 0), 1)

            text = " "
            cv.putText(image, text, (width_tracker*image_width+int(image_width/4), height_tracker*image_height+int(image_height/2)),
                       cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), lineType=cv.LINE_AA)

        # Other > threshold
        elif prediction[0][0] > classification_threshold:
            prediction = False
            cv.rectangle(image, (width_tracker*image_width, height_tracker*image_height),
                         (width_tracker*image_width+image_width-1, height_tracker*image_height+image_height-1),
                         (0, 0, 255), 1)

        # Under threshold
        else:
            cv.rectangle(image, (width_tracker*image_width, height_tracker*image_height),
                         (width_tracker*image_width+image_width-1, height_tracker*image_height+image_height-1),
                         (255, 0, 0), 1)

        # Tracker increase in order to handle full image
        height_tracker += 1
        if height_tracker == gridsize:
            height_tracker = 0
            width_tracker += 1
            if width_tracker == gridsize:
                break

    # For testing, save crops to folder
    if testing is True:
        image_name = "testi.jpg".format(height_tracker, width_tracker)
        cv.imwrite(image_name, image)

    return image


if __name__ == "__main__":

    image = cv.imread("1765120.jpg")
    testing = True
    first_call = True

    model = classification_model.generate_model((224, 224, 3))
    model.load_weights("weights.best.hdf5")

    image = split_and_predict_grid_images(model, image, 3, first_call, testing)

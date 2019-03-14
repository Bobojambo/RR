import image_editor
import classification_model
import numpy as np
import cv2
import glob


def classify_video_frames(model, gridsize=3, output_filename='output.avi', label_binarizer=None):

    # Atleast .mp4 video files work
    filename = input('Enter video file name to edit: ')
    capture = cv2.VideoCapture(filename)

    # Check if camera opened successfully
    if (capture.isOpened() is False):
        print("Error opening video stream or file")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_filename, fourcc, 10.0, (frame_width, frame_height))

    # Read until video is completed
    while(capture.isOpened()):
        # Capture frame-by-frame
        ret, frame = capture.read()
        if ret is True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Edit the frame according to
            edited_frame = image_editor.split_and_predict_grid_images(model, frame, gridsize, first_call=True)
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


def load_and_resize_images_from_paths(filepaths):

    images = []	
    for imagepath in filepaths:
        img = cv2.imread(imagepath)
        if img.shape[0] != 224:
            img = cv2.resize(img, (224,224))
            img = np.array(img)
        images.append(img)

    return images


def load_image_paths(globPath, target, test_data=False):

    image_files = []
    for filename in glob.glob(globPath):
        image_files.append(filename)

    filepaths = []
    targets = []
    for file in image_files:
        filepaths.append(file)
        targets.append(target)

    # For testing 500 first loaded images
    if test_data is True:
        filepaths = filepaths[:500]
        targets = targets[:500]

    return filepaths, targets


def main():

    input_str = str(input('Train or Load model (Train/Load): '))

    if input_str == "Train":

        input_str = str(input('Test (y/n): '))
        if input_str == "y":
            test = True
        elif input_str == "n":
            test = False

        # Water images
        target = "water"
        path = 'water_images/*.jpg'
        waterimagepaths, water_targets = load_image_paths(path, target, test)
        water_images = load_and_resize_images_from_paths(waterimagepaths)

        # Other images
        target = "other"
        path = 'ResizedImages224x224/*.jpg'
        other_images_paths, other_targets = load_image_paths(path, target, test)
        other_images = load_and_resize_images_from_paths(other_images_paths)

        # Join the images
        images = water_images + other_images
        targets = water_targets + other_targets
        images = np.array(images)

        # binarize targets
        label_binarizer = classification_model.binarize_targets(targets)

        # Generate model
        shape = images[0].shape
        model = classification_model.generate_model(shape)

        # Train the model
        weights_filepath = "weights.best.hdf5"
        history = classification_model.train_model(images, targets, model, label_binarizer, weights_filepath)

    elif input_str == "Load":

        model = classification_model.generate_model((224, 224, 3))
        try:
            weights_filepath = "weights.best.hdf5"
            model.load_weights(weights_filepath)
        except IOError:
            print("no weights file with the name '{}'".format(weights_filepath))

    else:
        print("Wrong input argument")

    if model is not None:
        while True:
            input_str = str(input('Classify video? (y/n):'))       

            if input_str == 'y':

                output_filename = str(input('output filename (.avi will be added): '))
                output_filename = output_filename + ".avi"

                if output_filename == '':
                    output_filename = 'output.avi'

                gridsize = input('input gridsize (x*x): ')
                try:
                    gridsize = int(gridsize)
                except ValueError:
                    print("Cannot cast grid size, setting default 3")
                    gridsize = 3
                    pass

                classify_video_frames(model, gridsize, output_filename)

            else:
                break

    return


if __name__ == "__main__":
    main()

from keras.layers.core import Dense, Flatten
from keras import Sequential
from keras.optimizers import Adam, SGD
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn import model_selection
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import glob


def generate_model(shape, number_of_classes=2):

    model = Sequential()

    mobilenet = MobileNet(include_top=False,  input_shape=shape, weights='imagenet', classes=1000)
    #resnet50 = ResNet50(include_top=False, weights='imagenet', input_shape=shape, classes=1000)

    model.add(mobilenet)
    model.add(Flatten())
    model.add(Dense(number_of_classes, activation='softmax'))
    model.summary()

    optimizer = Adam()
    #optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(images, targets, model, lb, filepath="weights.best.hdf5"):

    labels_binarized = lb.transform(targets)
    # model implementation doesnt work without if class amount == 2"
    labels_binarized = to_categorical(labels_binarized, num_classes=2)

    """
    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True)
    """
    X_train, X_test, y_train, y_test = model_selection.train_test_split(images, labels_binarized, random_state=42)

    callbacks_list = []
    batch_size = 64
    n_epochs = 10

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list.append(checkpoint)

    history = model.fit(X_train, y_train,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        callbacks=callbacks_list,
                        validation_data=(X_test, y_test))

    return history


def binarize_targets(targets):

    lb = LabelBinarizer()
    lb.fit(targets)
    np.save('classes.npy', lb.classes_)

    return lb


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


if __name__ == "__main__":
    print("do nothing")

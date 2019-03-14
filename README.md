# Mosaic water image classification project

In this project, the purpose was to study neural networks for water image classification. This is carried out as binary classification case. The data for training consisted of 2 classes - water and other. The water images were created using a simulation environment. The other class images consist real-life images of different objects such as vessels and structures but they were treated as only 1 class. 

# Requirements

- This project has been implemented using Anaconda Python distribution with **Python 3.5**. 
- Additional packages required for the project are included in the requirements.txt file.
- Opencv is required for the project. For installing opencv for anaconda distribution, please check https://anaconda.org/conda-forge/opencv
- For training the models, tensorflow is used. For installing tensorflow, please check https://www.tensorflow.org/install/pip

# Model Training Data

- For training a model to classify simulator images of water and real images of different objects in maritime objects are used
- Simulator images consist of HD and ground truth images in maritime environment
- Real images consist of HD images with
- Before training, place the data correspondingly 
  - 'data/simulator_images/'
  - 'data/real_images/'
- Run '__real_image_extractor.py__' and '__simulator_image_extractor.py__' to acquire the dataset used for training the model.
  - water class images will be extracted from the simulation images in 'water_images/' folder
  - other class images will be extracted in different sizes to 'ResizedImages/' folders

# Usage

- '__main.py__' is used to run the program
  - At start, the program will ask to Train/Load a neural network model. The training and the networks are specified in   '__classification_model.py__'
    - In training,  the weights of the best model will be saved as 'weights.best.hdf5'
    - In loading, 'weights.best.hdf5' will be used to load the weights of the network
  - Next, video images are classified using the generated model.
    - The program will ask for the name of a video file to classify using the model. (Atleast .mp4 files work)
    - The program will ask for output filename (Default output.avi.)
    - The program will ask for desired gridsize for mosaic. (Eg. 2 will generate a 2x2 image grid for classification)
    - The video frames and their classification is handled in '__image_editor.py__'.


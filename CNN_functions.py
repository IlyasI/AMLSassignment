import logging
import os
import shutil

import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img,
)
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

"""
split_dataset_random splits the dataset into training, validation, and test sets randomly.
The train:validation:test split I used is 60:20:20.

train length: 2739 | ratio: 0.6
validate length: 913 | ratio: 0.2
test length: 913 | ratio: 0.2

parameters: 
'df': dataframe of the dataset you want to split, I used this on the filtered dataframe obtained from ImageFiltering.py
'validation_split': the ratio that you want your validation set to be
'test_split': the ratio that you want your test set to be
'seed': seed to use for the random split, by keeping this the same each run we obtain a consistent output despite the 'randomness'

returns: 
'train, validate, test': dataframes of the corresponding splits of the data
"""


def split_dataset_random(df, validation_split=0.2, test_split=0.2, seed=123):
    # sets the seed for the randomized split:
    np.random.seed(
        seed
    )  # if the same seed is used the same random split will be produced

    # calculates the ratio for the train set:
    train_split = 1 - validation_split - test_split
    # calculates the indeces on which to split the dataframe:
    indeces_array = [int(train_split * len(df)), int((1 - test_split) * len(df))]

    """
    splits the dataset according to the indeces array,
    for example, an indeces_array of [2, 3] would, for axis=0, result in

        train = df[:2]
        validate = df[2:3]
        test = df[3:]

    """
    train, validate, test = np.split(df.sample(frac=1), indeces_array)
    logging.info(
        "train length: " + str(len(train)) + "| ratio: " + str(len(train) / len(df))
    )
    logging.info(
        "validate length: "
        + str(len(validate))
        + "| ratio: "
        + str(len(validate) / len(df))
    )
    logging.info(
        "test length: " + str(len(test)) + "| ratio: " + str(len(test) / len(df))
    )

    return train, validate, test


"""reset_symlinks is a utility function used to reset the symbolic links created by symlink_classes_images
"""


def reset_symlinks():
    dest_dir = "./images/"
    walker = os.walk(dest_dir)
    rem_dirs = walker.__next__()[1]

    for dirpath, dirnames, filenames in walker:
        print(dirpath)
        if dirpath != "./images/removed":
            shutil.rmtree(dirpath)


"""symlink_classes_images creates symbolic links for use with keras's 
flow_from_directory utility function. Symbolic links save disk space compared
with copying each image

The directory structure created looks like:
-images
----train
-------human
----------0
----------1
-------smiling
----------0
----------1
----validate
-------human
-------smiling
----test

where each class has its own subdirectory under the train, validate, test directories, 
where the image symlinks are created based on the class value of the image (e.g. 0 or 1)

parameters:
train: the train dataframe
validate: validation dataframe,
test: test dataframe,
rowName: the row name for which to create symbolic links (e.g. 'human' or 'hair_color')
imagesPath: path to the directory where the images are stored
trainPath: path to where you want the train symlinks to be stored
validatePath: path to where you want the validation symlinks to be stored
testPath: path to where you want the test symlinks to be stored
reset_symlinks: boolean, if True will reset any symlinks made on previous runs
"""


def symlink_classes_images(
    train,
    validate,
    test,
    rowName,
    imagesPath=os.path.abspath("./images/"),
    trainPath=os.path.abspath("./images/train/"),
    validatePath=os.path.abspath("./images/validate/"),
    testPath=os.path.abspath("./images/test/"),
    reset_symlinks=False,
):
    if reset_symlinks:
        reset_symlinks()
    newPaths = [trainPath, validatePath, testPath]
    dfList = [train, validate, test]

    # for loop cycles through the train, validate, and test dataframes
    for df in dfList:
        # for loop cycles through each row in the current dataframe
        for idx, row in df.iterrows():
            # the className will be the same as the row name, e.g. human or eyeglasses
            className = int(row[rowName])
            # used to form trainClassPath, validateClassPath, and testClassPath
            pathAppend = rowName + "/" + str(className) + "/"
            # gets the filename by combining the id number with '.png'
            fileName = str(int(row["file_name"])) + ".png"
            filePath = os.path.join(imagesPath, fileName)
            # trainClassPath is for example: './images/train/human/'
            trainClassPath = os.path.join(trainPath, pathAppend)
            # trainFilePath is for example: './images/train/human/0/1992.png'
            trainFilePath = os.path.join(trainClassPath, fileName)
            # validateClassPath is for example: './images/validate/eyeglasses/'
            validateClassPath = os.path.join(validatePath, pathAppend)
            # validateFilePath is for example: './images/validate/eyeglasses/1/87.png'
            validateFilePath = os.path.join(validateClassPath, fileName)
            testClassPath = os.path.join(testPath, pathAppend)
            testFilePath = os.path.join(testClassPath, fileName)
            # forms a list of all the classPaths
            classPaths = [trainClassPath, validateClassPath, testClassPath]
            # creates directories specified in the classPaths list if they dont exist
            for path in classPaths:
                if os.path.isdir(path) == False:
                    os.makedirs(path)
            if df.equals(test):
                newFilePath = testFilePath
            if df.equals(train):
                newFilePath = trainFilePath
            if df.equals(validate):
                newFilePath = validateFilePath
            # looks for the images in different paths until they are found and creates symlinks in the appropriate directories
            # if an image is not found logs an error.
            if os.path.isfile(filePath):
                # for example, symlink will be made in: ./images/train/human/1/ directory
                os.symlink(filePath, newFilePath)
            elif os.path.isfile(testFilePath):
                os.symlink(testFilePath, newFilePath)
            elif os.path.isfile(validateFilePath):
                os.symlink(validateFilePath, newFilePath)
            elif os.path.isfile(trainFilePath):
                os.symlink(trainFilePath, newFilePath)
            else:
                logging.error("File missing: " + fileName)


"""get_cnn_model returns an untrained CNN network model consisting of multiple 2d convolution layers, 2d max-pooling layers and dropout layers. 
With a fully-connected dense layer near the output
If the current classification task is binary, the final output layer will be a sigmoid function and binary_crossentropy will be used as the loss
function.
if it is multiclass, then the final output layer will be softmax and categorical_crossentropy will be used as the loss function.

The general structure of a convolutional neural network on which this network is based is 
described in more detail here: https://www.nature.com/articles/nature14539

This specific structure is based on the tutorial found here: 
https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
which in turn is a less deep form of the ConvNet-A architecture defined in this paper:
https://arxiv.org/pdf/1409.1556.pdf 
which is also given as an example here: http://cs231n.github.io/convolutional-networks/

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 32, 128, 128)      896
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 126, 126)      9248
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 63, 63)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 32, 63, 63)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 63, 63)        9248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 32, 61, 61)        9248
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 32, 30, 30)        0
_________________________________________________________________
dropout_2 (Dropout)          (None, 32, 30, 30)        0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 64, 30, 30)        18496
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 64, 28, 28)        36928
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 64, 14, 14)        0
_________________________________________________________________
dropout_3 (Dropout)          (None, 64, 14, 14)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 12544)             0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               6423040
_________________________________________________________________
dropout_4 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 513
=================================================================
Total params: 6,507,617
Trainable params: 6,507,617
Non-trainable params: 0
_________________________________________________________________

parameters:
'optimizer': 'adam' or 'rmsprop', which optimization algorithm to use to train the network
              both Adam and RMSProp have good performance.
'class_mode': 'binary' or 'multiclass', use 'binary' for binary classification problems
              and 'multiclass' for multi-class classification problems.
'multiclass_n_classes': integer which defines the number of classes if 'class_mode' is set to 'multiclass'
'input_shape': input shape of the image RGB values
'channels_first': boolean, if True will use Theano ordering of input shape, where the number of channels 
                  comes first, e.g. (3, 128, 128). If false will use Tensorflow ordering e.g. (128, 128, 3)
"""


def get_cnn_model(
    optimizer="adam",
    class_mode="binary",
    multiclass_n_classes=7,
    input_shape=(3, 128, 128),
    channels_first=True,
):
    if channels_first:
        # use this if channels are first in input shape e.g. (3,128,128)
        # theano input shape ordering
        K.set_image_dim_ordering("th")
    else:
        # use this if channels are last in input shape e.g. (128, 128, 3)
        # tensorflow input shape ordering
        K.set_image_dim_ordering("tf")

    model = Sequential()
    # add a 2d convolutional layer as the first layer to the network with ReLU activation.
    # Setting filters=32 defines the dimensionality of the output space to be 32
    # Setting kernel_size=(3,3) specifies the height and width respectively of the convolution window.
    # Setting strides=(1,1) (which is also the default value) specifies the strides of the convolution along
    # the height and width. When the stride is 1, we move the filters 1 pixel at a time.

    # we add padding="same" to pad the input so the output of this layer will have the same length as
    # the input to the layer. input_shape must be defined as this is the first layer in the model, here it is
    # the shape of the rgb values of each image provided as input, e.g. (3, 128, 128).
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            input_shape=input_shape,
        )
    )
    # stack another conv2d layer with similar parameters as above
    # padding='valid' means that there is no padding, as it is not necessary prior to pooling
    # (no input_shape this time as it is not the input layer)

    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )
    )
    # add a pooling layer in order to reduce the total number of parameters in the network
    # the most common type of pooling layer is MAX pooling with filters of size (2,2) as used here.
    # setting strides=None causes strides to default to the pool_size of (2,2).
    # This pooling configuration therefore downsamples the representation by taking the max over each 2x2 region.
    # Max pooling has been shown to work better than average pooling in practice (http://cs231n.github.io/convolutional-networks/)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
    # add a dropout layer to help prevent overfitting by setting a fraction of the weights to zero during training
    # rate=0.25 means that 0.25 of the weights will be dropped.
    model.add(Dropout(rate=0.25))

    # repeat same configuration again as above to make network deeper:
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
    model.add(Dropout(rate=0.25))

    # repeat again, but increasing the number of output filters of each convolutional layer to 64
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )
    )
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None))
    model.add(Dropout(rate=0.25))

    # add a flatten layer to flatten the output from the previous layer for input into the dense layer
    model.add(Flatten())
    # add a densely-connected layer with ReLU activation with 512 neurons.
    model.add(Dense(units=512, activation="relu"))
    # add a final dropout layer with a higher dropout rate of 0.5
    model.add(Dropout(rate=0.5))
    # If this is a binary classification problem, the final output layer will be
    # a single neuron dense layer with sigmoid activation, to map the weights to a single
    # output value, corresponding to a class value of 0 or 1.
    # Further, the loss is set to binary crotrain_classifierssentropy.
    if class_mode == "binary":
        model.add(Dense(units=1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
    # If this is a multiclass problem, instead we will use a dense layer with 7 neurons,
    # with softmax regression, which is more appropriate for multiclass problems. Each neuron
    # in this dense layer corresponds to one of the classes.
    # Further, the loss is set to categorical crossentropy.
    elif class_mode == "multiclass":
        model.add(Dense(units=multiclass_n_classes, activation="softmax"))

        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
    return model


"""get_ImageDataGenerator returns a Keras image data generator. This can be used to
generate batches of tensor image data. 
Additionally, it can be used to generate augmented images from the original images to increase the training set size.
I did not end up using augmented images for training as it turned out my model was already
very accurate, but this can be useful for more difficult problems.

All parameters are set to 0 or False in order to keep the original images without augmenting them."""


def get_ImageDataGenerator(
    shear_range=0,
    zoom_range=0,
    horizontal_flip=False,
    rotation_range=0,
    zca_whitening=False,
):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=shear_range,
        zoom_range=zoom_range,
        zca_whitening=zca_whitening,
        horizontal_flip=horizontal_flip,
        rotation_range=rotation_range,
        data_format="channels_first",
    )

    return datagen


"""get_flow_from_directory returns a flow of images from the provided image directory.
A flow consists of batches of image data.
This flow is then used as the input to train and test the Keras CNN model.

parameters:
'ImageDataGenerator': the image data generator returned by get_ImageDataGenerator
'img_dir': directory where your images are stored
'target_size': the dimensions to which all images will be rescaled, set to (128, 128) as that
               is the size expected as input to the CNN model.
'class_mode': 'binary' or 'categorical', determines the type of label arrays returned
              'binary' returned 1D binary labels, 'categorical' returns 2D one-hot encoded labels
              use binary if it is a binary classification problem and categorical if it is multiclass
'batch_size': size of the batches of data generated, 32 is the expected size for our CNN model
'shuffle': boolean, if True the generated data will be shuffled
'seed': seed to use if randomly shuffling
"""


def get_flow_from_directory(
    ImageDataGenerator,
    img_dir,
    target_size=(128, 128),
    class_mode="binary",
    batch_size=32,
    shuffle=True,
    seed=123,
):
    flow = ImageDataGenerator.flow_from_directory(
        directory=img_dir,
        target_size=target_size,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        follow_links=True,
    )
    return flow

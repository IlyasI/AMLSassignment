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


def split_dataset_random(df, validation_split=0.2, test_split=0.2, seed=123):
    np.random.seed(
        seed
    )  # if the same seed is used the same random split will be produced
    train_split = 1 - validation_split - test_split
    indeces_array = [int(train_split * len(df)), int((1 - test_split) * len(df))]
    """example, [2, 3] would, for axis=0, result in

        ary[:2]
        ary[2:3]
        ary[3:]

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


def reset_symlinks():
    dest_dir = "./images/"
    walker = os.walk(dest_dir)
    rem_dirs = walker.__next__()[1]

    for dirpath, dirnames, filenames in walker:
        print(dirpath)
        if dirpath != "./images/removed":
            shutil.rmtree(dirpath)


"""Creates symlinks for use with keras's flow_from_directory utility function"""


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

    for df in dfList:
        for idx, row in df.iterrows():
            className = int(row[rowName])
            pathAppend = rowName + "/" + str(className) + "/"

            fileName = str(int(row["file_name"])) + ".png"
            filePath = os.path.join(imagesPath, fileName)
            trainClassPath = os.path.join(trainPath, pathAppend)
            trainFilePath = os.path.join(trainClassPath, fileName)
            validateClassPath = os.path.join(validatePath, pathAppend)
            validateFilePath = os.path.join(validateClassPath, fileName)
            testClassPath = os.path.join(testPath, pathAppend)
            testFilePath = os.path.join(testClassPath, fileName)
            classPaths = [trainClassPath, validateClassPath, testClassPath]
            for path in classPaths:
                if os.path.isdir(path) == False:
                    os.makedirs(path)
            if df.equals(test):
                newFilePath = testFilePath
            if df.equals(train):
                newFilePath = trainFilePath
            if df.equals(validate):
                newFilePath = validateFilePath
            if os.path.isfile(filePath):
                os.symlink(filePath, newFilePath)
            elif os.path.isfile(testFilePath):
                os.symlink(testFilePath, newFilePath)
            elif os.path.isfile(validateFilePath):
                os.symlink(validateFilePath, newFilePath)
            elif os.path.isfile(trainFilePath):
                os.symlink(trainFilePath, newFilePath)
            else:
                logging.error("File missing: " + fileName)


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
    model.add(
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape)
    )
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    # model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    if class_mode == "binary":
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
    elif class_mode == "multiclass":
        model.add(Dense(multiclass_n_classes, activation="softmax"))

        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )
    return model


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

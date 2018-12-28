import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import metrics, svm
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import CNN_functions


def prepare_for_svm(seed=0):
    filteredDf = pd.read_csv("./generated_csv/attribute_list_filtered.csv")
    train, validate, test = CNN_functions.split_dataset_random(df=filteredDf, seed=seed)
    return train, validate, test


def create_image_dataset(dataframe, images_path="./images/"):
    # df = pd.read_csv(labels_csv_path)
    # images_glob = glob.glob1(images_path, '*.png')
    # N_images = len(images_glob)
    # image_dataset = np.empty((N_images, 1, 128, 128, 3), dtype=np.uint8)
    img_arrays = {}
    for i, image in enumerate(dataframe["file_name"]):
        # print(i)
        img_path = images_path + str(image) + ".png"
        img = Image.open(img_path).convert("L").resize((128, 128))

        img_array = np.array(img)
        img_array = (img_array - img_array.mean()) / np.sqrt(img_array.var() + 1e-5)
        # adds the file number to the img_array:
        # img_array[0] = int(os.path.splitext(image)[0])
        img_arrays[image] = img_array
        print(i)

    print(sys.getsizeof(img_arrays))
    # print(img_arrays[4927])
    return img_arrays


def get_X_Y_from_dataset(image_dataset, row_name, labels_dataframe):
    df = labels_dataframe
    pca = PCA(n_components=128)
    X = []  # images
    Y = []  # labels
    for key, value in image_dataset.items():
        value_PCA = pca.fit_transform(value)
        # print(value.shape)
        # print(value_PCA.shape)
        X.append(value_PCA)
        label = df.loc[df["file_name"] == key, row_name].item()
        if label == -1:
            label = 0
        Y.append(label)
        # print(key, label)
    # print('X_SHAPE:', X.shape)
    # print('Y_SHAPE', Y.shape)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def run_svm_classifier(row_name, model_save_path, images_path="./images/"):

    train, validate, test = prepare_for_svm(seed=0)
    # print(train)
    image_dataset_train = create_image_dataset(train, images_path=images_path)
    image_dataset_test = create_image_dataset(test, images_path=images_path)
    X_train, Y_train = get_X_Y_from_dataset(image_dataset_train, row_name, train)
    X_test, Y_test = get_X_Y_from_dataset(image_dataset_test, row_name, test)

    n_train, nx_train, ny_train = X_train.shape
    n_test, nx_test, ny_test = X_test.shape
    # X_train_PCA_flat = pca.fit_transform(X_train.reshape((N_train, -1)))
    # X_test_PCA_flat = pca.fit_transform(X_test.reshape((N_test, -1)))
    svc = svm.SVC(gamma="scale", kernel="linear", verbose=True, class_weight="balanced")
    parameters={'C':[1,10,100]}
    #parameters_random = {"C": [1, 10, 100]}
    classifier = GridSearchCV(svc, parameters)
    #classifier = RandomizedSearchCV(svc, parameters_random, cv=5, n_iter=2, verbose=2)

    print("Fitting classifier...")

    X_train_flat = X_train.reshape((n_train, nx_train * ny_train))
    X_test_flat = X_test.reshape((n_test, nx_test * ny_test))

    classifier.fit(X_train_flat, Y_train)
    pickle.dump(classifier, open(model_save_path + row_name + "_svm_trained.sav", "wb"))
    print(classifier.cv_results_)
    print("Getting predictions from classifier...")
    score = classifier.score(X_test_flat, Y_test)
    print("Test Score: ", score)


run_svm_classifier("human", "./svm_saved_models/")

# print("Classification report for classifier %s:\n%s\n"
#      % (classifier, metrics.classification_report(Y_test, y_pred)))

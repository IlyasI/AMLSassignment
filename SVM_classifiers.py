import csv
import os
import pickle
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import expon
from sklearn import metrics, svm
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV

from CNN_functions import split_dataset_random
from CNN_classifiers import get_metrics
from ImageFiltering import get_features


def prepare_for_svm(seed=0):
    filteredDf = pd.read_csv("./generated_csv/attribute_list_filtered.csv")
    train, validate, test = split_dataset_random(df=filteredDf, seed=seed)
    return train, validate, test


def create_image_dataset(dataframe, images_path="./images/"):
    img_arrays = {}
    for i, image in enumerate(dataframe["file_name"]):

        img_path = images_path + str(image) + ".png"
        img = Image.open(img_path).convert("RGB").resize((128, 128))
        img_array = np.array(img).reshape(128, -1)
        img_array = (img_array - img_array.mean()) / np.sqrt(img_array.var() + 1e-5)
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

        X.append(value_PCA)
        label = df.loc[df["file_name"] == key, row_name].item()
        if label == -1:
            label = 0
        Y.append(label)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def run_svm_classifier(
    row_name, model_save_load_path, images_path="./images/", load_model=False
):
    model_path = model_save_load_path + row_name + "_svm_trained.sav"

    train, validate, test = prepare_for_svm(seed=0)

    image_dataset_train = create_image_dataset(train, images_path=images_path)
    image_dataset_test = create_image_dataset(test, images_path=images_path)
    X_train, Y_train = get_X_Y_from_dataset(image_dataset_train, row_name, train)
    X_test, Y_test = get_X_Y_from_dataset(image_dataset_test, row_name, test)
    n_train, nx_train, ny_train = X_train.shape
    n_test, nx_test, ny_test = X_test.shape
    X_train_flat = X_train.reshape((n_train, nx_train * ny_train))
    X_test_flat = X_test.reshape((n_test, nx_test * ny_test))

    if load_model == False:

        classifier = get_svm_classifier(randomized_search=True, n_iter=5)

        print("Fitting classifier...")

        classifier.fit(X_train_flat, Y_train)

        if os.path.isdir(model_save_load_path) == False:
            os.makedirs(model_save_load_path)

        pickle.dump(classifier, open(model_path, "wb"))
    else:
        classifier = pickle.load(open(model_path, "rb"))

    y_pred = classifier.predict(X_test_flat)
    filenames = test["file_name"]

    output_df, output_df_wrong, report, accuracy, confusion = get_metrics(
        Y_test, y_pred, filenames
    )
    generate_test_predictions_csv(output_df, accuracy, row_name, path_append="svm")


def get_svm_classifier(randomized_search=True, n_iter=10):
    svc = svm.SVC(gamma="scale", kernel="linear", verbose=True, class_weight="balanced")
    if randomized_search:
        parameters = {"C": expon(scale=100)}
        classifier = RandomizedSearchCV(
            svc, parameters, n_iter=n_iter, verbose=1, n_jobs=-1
        )
    else:
        classifier = svc
    return classifier


def generate_test_predictions_csv(
    output_df, accuracy, row_name, path_append="resnet50"
):
    output_df["sort"] = output_df["filename"].astype(int)
    output_df = output_df.sort_values("sort", ascending=True).reset_index(drop=True)
    output_df = output_df.drop("sort", axis=1)

    if row_name == "smiling":
        task_num = 1
    elif row_name == "young":
        task_num = 2
    elif row_name == "eyeglasses":
        task_num = 3
    elif row_name == "human":
        task_num = 4
    elif row_name == "hair_color":
        task_num = 5
    csv_filename = "task_" + str(task_num) + "_" + path_append + ".csv"

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([accuracy])
        for i in range(len(output_df)):
            writer.writerow([output_df["filename"][i], output_df["y_pred"][i]])


"""https://arxiv.org/abs/1403.6382"""


def transfer_learning_svm(row_name, load_features=False):
    filteredDf = pd.read_csv("./generated_csv/attribute_list_filtered.csv")
    # Drop rows where hair_color = -1, these are mislabeled and impact the accuracy of the trained model
    if row_name == "hair_color":
        filteredDf = filteredDf[filteredDf.hair_color != -1]

    if load_features == False:
        feature_df = get_features(
            img_df=filteredDf,
            network="resnet50",
            pooling="max",
            images_path="./images",
            df_to_pickle=True,
            pickle_save_dir="generated_csv/transfer_learning/",
        )
    else:
        feature_df = pd.read_pickle(
            "generated_csv/transfer_learning/image_features_resnet50_max.pkl"
        )

    filtered_feature_df = pd.merge(
        filteredDf, feature_df, left_on="file_name", right_on="id"
    )
    print(filtered_feature_df.head())

    train, validate, test = split_dataset_random(
        df=filtered_feature_df, validation_split=0, test_split=0.2, seed=0
    )
    print("YTRAIN SHAPE: " + str(train["features"].shape))

    classifier = get_svm_classifier(randomized_search=True)
    train_feature_list = np.array(train["features"].tolist())
    test_feature_list = np.array(test["features"].tolist())
    train_feature_list = train_feature_list.reshape((train_feature_list.shape[0], -1))
    test_feature_list = test_feature_list.reshape((test_feature_list.shape[0], -1))
    print("Test_feature_list shape: ", str(test_feature_list.shape))
    print("train_feature_list shape: ", str(train_feature_list.shape))
    print("LABEL SHAPE: ", str(train[row_name].shape))
    classifier.fit(train_feature_list, train[row_name])

    y_pred = classifier.predict(test_feature_list)
    y_true = test[row_name].values

    print("y_true shape:", str(y_true.shape))
    filenames = test["file_name"].values

    print("y_pred shape: ", y_pred.shape)
    print("filenames shape: ", filenames.shape)

    output_df, output_df_wrong, report, accuracy, confusion = get_metrics(
        y_true, y_pred, filenames
    )
    generate_test_predictions_csv(output_df, accuracy, row_name, path_append="resnet50")
import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from sklearn import cluster, decomposition, manifold, pipeline

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


"""This function will move all files contained in sub-directories of
a provided parent directory to that parent directory. 

Use this when you want to reset any symlinks or sorting/seperating
done with the images.

parameters:
'path': should be your image parent directory
'delete_dirs': boolean to determine whether to delete the sub-directories"""


def reset_image_subdirs(path="./images", delete_dirs=True):
    walker = os.walk(path)
    # returns all sub directories of path provided
    sub_dirs = walker.__next__()[1]
    for data in walker:
        for files in data[2]:
            # tries to move files in sub-directories to specified path
            # if error occurs, e.g. permissions or file does not exist
            # error will be displayed and loop continues with next file
            try:
                shutil.move(data[0] + os.sep + files, path)
            except shutil.Error as e:
                logging.error(e)
                continue
    # if delete_dirs=True, will delete the directory tree of the sub-dir
    # use this to clean up folder
    if delete_dirs:
        for dir in sub_dirs:
            dir_path = path + os.sep + dir
            logging.info("Removing sub-dir: " + dir_path)
            shutil.rmtree(dir_path)


def get_model(network="resnet50", pooling="avg"):
    # Pooling performed to reduce variance and computation complexity and extract low level features
    # Note, max pooling extracts important features like edges, average pooling extracts features smoothly.

    # include_top set to false to exclude the final dense layers in the network
    # this allows us to get the features as a vector of floats, rather than the interpretation of the features, e.g. dog, cat
    if network == "resnet50":
        # Best performing model by far
        # ResNet50 has 25,636,712 parameters
        logging.info("ResNet50 pretrained on ImageNet selected to extract features...")
        model = applications.resnet50.ResNet50(
            weights="imagenet", include_top=False, pooling=pooling
        )
    elif network == "mobilenetv2":
        # 3,538,984 parameters
        logging.info(
            "MobileNetV2 pretrained on ImageNet selected to extract features..."
        )
        model = applications.mobilenet_v2.MobileNetV2(
            weights="imagenet", include_top=False, pooling=pooling
        )
    elif network == "nasnet":
        # 5,326,716 parameters
        logging.info(
            "NASNetMobile pretrained on ImageNet selected to extract features..."
        )
        model = applications.nasnet.NASNetMobile(
            weights="imagenet",
            include_top=False,
            pooling=pooling,
            input_shape=(128, 128, 3),
        )
    return model


def get_features(
    network="resnet50",
    pooling="avg",
    images_path="./images",
    num_images=5000,
    df_to_csv=True,
    csv_save_dir="./generated_csv/",
):
    model = get_model(network=network, pooling=pooling)
    image_features = []
    image_file_type = ".png"
    for i in range(1, num_images + 1):
        # forms the full path to image with id 'i'
        # e.g. image_path = <images_path>/3987.png
        image_path = os.path.join(images_path, str(i) + image_file_type)
        if os.path.isfile(image_path):
            logging.info("Processing image: " + image_path)
            img = image.load_img(image_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            features = model.predict(img_array)[0]
            # form numpy char array from the model's float features output
            features_array = np.char.mod("%f", features)
            image_features.append(features_array)

    features_df = pd.DataFrame(
        {"id": range(1, num_images + 1), "features": image_features}
    )

    if df_to_csv:
        csv_save_path = (
            csv_save_dir + "image_features_" + network + "_" + pooling + ".csv"
        )
        if os.path.isdir(csv_save_dir) == False:
            os.mkdir(csv_save_dir)
        logging.info("Saving extracted features to csv: " + csv_save_path)
        features_df.to_csv(csv_save_path, index=False)

    return features_df


"""apply pca then t-sne dimensionality reduction to help map
image features to x, y plane for clustering and class visualization"""


def pcaTsnePipe():
    tsne = manifold.TSNE(random_state=0, perplexity=50, early_exaggeration=6.0)
    pca = decomposition.PCA(n_components=48)

    return pipeline.Pipeline([("PCA", pca), ("T-sne", tsne)])


def map_features_to_2d_plane(features_df):
    logging.info("Mapping features to x, y plane...")
    model = pcaTsnePipe()

    feature_list = features_df["features"]
    feature_list = np.array(feature_list.tolist())
    feature_list = feature_list.reshape((feature_list.shape[0], -1))

    fit_data = model.fit_transform(feature_list)

    results = []
    for i in range(0, len(features_df)):
        results.append(
            {"id": features_df["id"][i], "x": fit_data[i][0], "y": fit_data[i][1]}
        )

    results_df = pd.DataFrame(results)
    return results_df


def plot_kmeans_elbow_method(
    results_df, max_clusters=15, plt_save_dir="./generated_plots/"
):
    logging.info("Plotting elbow method for k means clustering...")
    sum_of_squared_distances = []
    # drop 'id' column for k-means
    results_df_k = results_df.drop("id", 1)
    # defines range of clusters to try
    K = range(1, max_clusters)
    for k in K:
        k_means = cluster.KMeans(n_clusters=k)
        k_means = k_means.fit(results_df_k)
        sum_of_squared_distances.append(k_means.inertia_)
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.plot(K, sum_of_squared_distances, "bx-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Sum of squared distances (k-means inertia)")
    plt.title("Elbow Method to Find Optimal Number of Clusters for K-Means")

    plt_save_path = plt_save_dir + "elbow_method.png"
    if os.path.isdir(plt_save_dir) == False:
        os.mkdir(plt_save_dir)
    plt.savefig(plt_save_path)
    plt.show()


def cluster_images(results_df, n_clusters=5, input_n_clusters=False):
    if input_n_clusters:
        # as the elbow method is mostly visual, it can be useful to prompt the user to input the number of optimal clusters observed
        n_clusters = input("Enter value for n_clusters (from k-means elbow method): ")
    logging.info("Clustering images with " + str(n_clusters) + " clusters...")
    results_df_k = results_df.drop("id", 1)
    k_means = cluster.KMeans(n_clusters=n_clusters)
    k_means.fit_predict(results_df_k)
    clusters_means = k_means.cluster_centers_.squeeze()
    image_clusters = k_means.labels_

    print("# of Observations: ", results_df_k.shape)
    print("Clusters Means: ", clusters_means)
    print("labels: ", image_clusters)
    return image_clusters


def plot_clusters(results_df, image_clusters, plt_save_dir="./generated_plots/"):
    logging.info("Plotting clusters...")
    plt.rcParams["figure.figsize"] = [10, 10]
    fig, ax = plt.subplots()
    x = results_df["x"]
    y = results_df["y"]
    sc = plt.scatter(x, y, c=image_clusters, alpha=1)
    clset = set(zip(image_clusters, image_clusters))
    handles = [
        plt.plot([], color=sc.get_cmap()(sc.norm(c)), ls="", marker="o")[0]
        for c, l in clset
    ]
    labels = [l for c, l in clset]
    plt.legend(handles, labels)

    ax.grid(True)
    # for i, txt in enumerate(results_df['id'].values):
    #    ax.annotate(txt, (x[i], y[i]), rotation=45)
    plt_save_path = plt_save_dir + "cluster_plot.png"
    if os.path.isdir(plt_save_dir) == False:
        os.mkdir(plt_save_dir)
    plt.savefig(plt_save_path)
    plt.show()


def get_attributes_clusters_df(
    image_clusters,
    attribute_list_path="./attribute_list.csv",
    csv_save_dir="./generated_csv/",
    save_to_csv=True,
):
    logging.info("Getting attributes dataframe with clusters")
    attribute_list_df = pd.read_csv(attribute_list_path)
    attribute_list_df["cluster"] = image_clusters
    if save_to_csv:
        csv_save_path = csv_save_dir + "attribute_list_w_clusters.csv"
        if os.path.isdir(csv_save_dir) == False:
            os.mkdir(csv_save_dir)
        attribute_list_df.to_csv(csv_save_path, index=False)

    return attribute_list_df


def build_filtered_dataframe(
    attribute_list_df,
    image_num_to_filter,
    move_filtered=True,
    save_to_csv=True,
    csv_save_dir="./generated_csv/",
    images_path="./images/",
    removed_images_path="./images/removed/",
):
    logging.info("Building filtered dataframe...")
    # gets the cluster the supplied image is in, that is the cluster number
    # that will get filtered
    df = attribute_list_df
    cluster_to_filter = df.loc[
        df["file_name"] == image_num_to_filter, "cluster"
    ].values[0]
    print(cluster_to_filter)
    images_to_remove_df = df.loc[df["cluster"] == cluster_to_filter]
    filtered_df = df.drop(df[df["cluster"] == cluster_to_filter].index)
    # moves filtered images to their own directory:
    if move_filtered:
        if os.path.isdir(removed_images_path) == False:
            os.mkdir(removed_images_path)
        for idx, row in images_to_remove_df.iterrows():
            file_name = str(row["file_name"]) + ".png"
            file_path = os.path.join(images_path, file_name)
            new_file_path = os.path.join(removed_images_path, file_name)
            shutil.move(file_path, new_file_path)
            logging.info("Moved file " + file_name + " to " + new_file_path + ".")

    # cluster column no longer needed
    filtered_df = filtered_df.drop("cluster", 1)

    if save_to_csv:
        csv_save_path = csv_save_dir + "attribute_list_filtered.csv"
        if os.path.isdir(csv_save_dir) == False:
            os.mkdir(csv_save_dir)
        filtered_df.to_csv(csv_save_path, index=False)
    return filtered_df


def run_image_filtering_from_scratch():
    reset_image_subdirs()

    features_df = get_features(
        network="resnet50", pooling="max", images_path="./images", num_images=5000
    )
    print(features_df.head())
    results_df = map_features_to_2d_plane(features_df)
    plot_kmeans_elbow_method(
        results_df, max_clusters=15, plt_save_dir="./generated_plots/"
    )
    image_clusters = cluster_images(results_df, n_clusters=5)
    plot_clusters(results_df, image_clusters, plt_save_dir="./generated_plots/")
    attribute_list_df = get_attributes_clusters_df(
        image_clusters,
        attribute_list_path="./attribute_list.csv",
        csv_save_dir="./generated_csv/",
        save_to_csv=True,
    )
    # image number 4 is a scenery image, all similar images will be filtered
    filtered_df = build_filtered_dataframe(
        attribute_list_df=attribute_list_df,
        image_num_to_filter=4,
        move_filtered=True,
        save_to_csv=True,
        csv_save_dir="./generated_csv/",
        images_path="./images",
        removed_images_path="./images/removed/",
    )
    print(filtered_df.head())


def run_image_filtering_from_csv():
    attribute_list_df = pd.read_csv("./generated_csv/attribute_list_w_clusters.csv")
    # image number 4 is a scenery image, all similar images will be filtered
    filtered_df = build_filtered_dataframe(
        attribute_list_df=attribute_list_df,
        image_num_to_filter=4,
        move_filtered=True,
        save_to_csv=True,
        csv_save_dir="./generated_csv/",
        images_path="./images",
        removed_images_path="./images/removed/",
    )
    print(filtered_df.head())


run_image_filtering_from_csv()

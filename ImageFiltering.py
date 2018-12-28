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


"""The reset_image_subdirs is a utility function that 
will move all files contained in sub-directories of
a provided parent directory to that parent directory. 

Use this when you want to reset any symlinks or sorting/seperating
done with the images, such as when you want to train the network for a
different classification task.

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


"""The get_model function returns a model included in Keras for image classification that was pre-trained on ImageNet. 
There are 3 different networks that this function provides: ResNet50, a 50 layer residual network 
from 'Deep Residual Learning for Image Recognition' 
(https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
which has 25,636,712 network parameters in total. This is the final model I used as the shallower networks just did not
get enough accuracy.

The other 2 networks are much smaller, MobileNetV2 has only 3,538,984 parameters and NasNetMobile has only 5,326,716 parameters. 
Both did not get good performance, that is why the use of the much larger ResNet50 is justified.

parameters:
'network': Can be either 'resnet50', 'mobilenetv2', or 'nasnet', the default and recommended option is 'resnet50'.
'pooling': 'avg', or 'max', determines whether global max pooling or global average pooling will be used, both give comparable results with resnet50.
"""


def get_model(network="resnet50", pooling="avg"):
    # Pooling performed to reduce variance and computation complexity and extract low level features
    # Note, max pooling extracts important features like edges, average pooling extracts features smoothly.

    # include_top set to false to exclude the final dense layers in the network
    # this allows us to get the features as a vector of floats, rather than the interpretation of the features, e.g. dog, cat
    if network == "resnet50":
        # Best performing model by far
        # ResNet50 has 25,636,712 parameters
        logging.info("ResNet50 pretrained on ImageNet selected to extract features...")
        # include_top is set to false to ignore the output layer of the networks in order to
        # obtain features in vectors of floats rather than labels like 'cats' or 'dogs'.
        # this is done in order to later perform k-means clustering after mapping these vectors to
        # the x-y plane.
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


"""get_features returns a pandas dataframe which has one column for image ids and
another for the feature vector of that image. The features are extracted by using one of the
pre-trained networks provided by the get_model function above to 'predict' them.

parameters:
'network': Can be either 'resnet50', 'mobilenetv2', or 'nasnet', the default and recommended option is 'resnet50'.
'pooling': 'avg', or 'max', determines whether global max pooling or global average pooling will be used, both give comparable results with resnet50.
'images_path': should be the path to the directory where your images are stores.
'num_images': the number of images in your dataset, forms the range of id's to cycle through (e.g. 1-5000)
'df_to_csv': Boolean that decided whether to save the dataframe of image features to a csv file, specify the save directory with the 'csv_save_dir' parameter.
'csv_save_dir': the directory where you want the csv file of the image features to be saved, only used if 'df_to_csv' is set to True.
"""


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
    # iterate over all the images (in the range provided, by default it is 1-5000)
    for i in range(1, num_images + 1):
        # forms the full path to image with id 'i'
        # e.g. image_path = <images_path>/3987.png
        image_path = os.path.join(images_path, str(i) + image_file_type)
        if os.path.isfile(image_path):
            logging.info("Processing image: " + image_path)
            # load image:
            img = image.load_img(image_path, target_size=(128, 128))
            # obtain array of rgb values from image:
            img_array = image.img_to_array(img)
            # expand dimensions of the image array to get it into an appropriate shape for the 'ResNet50' network:
            img_array = np.expand_dims(img_array, axis=0)
            # this function is provided by Keras to preprocess the input into the appropriate form for the 'ResNet50' network:
            img_array = preprocess_input(img_array)
            # obtain the predicted feature vector of the image, with the features being floats:
            features = model.predict(img_array)[0]
            # form numpy char array from the model's float features output:
            features_array = np.char.mod("%f", features)
            # appends the numpy char array of features to the image_features list:
            image_features.append(features_array)

    # forms a pandas dataframe from the range of image id's (1-5000),
    # and the image_features list of numpy char arrays of each images features as formed above:
    features_df = pd.DataFrame(
        {"id": range(1, num_images + 1), "features": image_features}
    )
    # if the df_to_csv parameter is set to true, the features_df dataframe will be saved to a csv file
    # in the provided csv_save_dir, named for example: 'image_features_ResNet50_max.csv':
    if df_to_csv:
        csv_save_path = (
            csv_save_dir + "image_features_" + network + "_" + pooling + ".csv"
        )
        if os.path.isdir(csv_save_dir) == False:
            os.mkdir(csv_save_dir)
        logging.info("Saving extracted features to csv: " + csv_save_path)
        features_df.to_csv(csv_save_path, index=False)

    return features_df


"""pca_tsne_pipeline returns a pipeline of principal component analysis (PCA) and then 
t-distributed stochastic neighbor embedding (T-sne) sci-kit learn functions. 
PCA is a dimensionality-reduction technique, which is used to obtain the principal compenents
from the data. The principal compenents retain most of the variation and patterns
found in the original data, but are much less complex. PCA is used because the sklearn documentation
for T-sne highly recommends another dimensionality reduction method prior to input to T-sne.

T-sne is another dimensionality reduction technique which maps high-dimensional data to a 2-d or 3-d
plane (in this case 2-d). K-means clustering can then be used on the mapping obtained and the similarity 
of all the images can be visualized. """


def pca_tsne_pipeline():
    # sklearn TSNE function, all parameters kept at default values:
    tsne = manifold.TSNE(random_state=0, perplexity=30, early_exaggeration=12.0)
    # sklearn PCA functions, n_components sets the number of components to keep, I used the
    # recommended n_components=50 for T-sne
    pca = decomposition.PCA(n_components=50)
    # returns a pipeline model consisting first of PCA, then of T-sne:
    return pipeline.Pipeline([("PCA", pca), ("T-sne", tsne)])


"""map_features_to_2d_plane applied the pca_tsne_pipeline function above on
features_df, the dataframe of all the image features.

parameters:
'features_df': the dataframe of features returned by the get_features function
returns:
'results_df': a dataframe of the x-y mappings of each of the image's features.
"""


def map_features_to_2d_plane(features_df):
    logging.info("Mapping features to x, y plane...")
    model = pca_tsne_pipeline()

    # convert the features dataframe to a numpy array of features
    # then reshape to 2-d for input into fitting and transforming with
    # the pca_tsne_pipeline model.
    feature_list = features_df["features"]
    feature_list = np.array(feature_list.tolist())
    feature_list = feature_list.reshape((feature_list.shape[0], -1))
    # fit_data is the x-y mappings of each of the feature vectors
    fit_data = model.fit_transform(feature_list)

    results = []
    for i in range(0, len(features_df)):
        results.append(
            {"id": features_df["id"][i], "x": fit_data[i][0], "y": fit_data[i][1]}
        )

    results_df = pd.DataFrame(results)
    # results_df has columns 'id', 'x', and 'y'.
    return results_df


"""plot_kmeans_elbow_method plots the elbow method to find the optimal number of
clusters for k-means clustering. Displays and saves a plot of the k-means inertia 
compared with the number of clusters.

After running this on the data, it is clear from the elbow method that
the optimal number of clusters is n=5.

parameters:
'results_df': the dataframe returned by map_features_to_2d_plane
'max_clusters': the max range of clusters to test for (e.g. n=1 to n=15)
'plt_save_dir': directory where to save the generated elbow method plot
"""


def plot_kmeans_elbow_method(
    results_df, max_clusters=15, plt_save_dir="./generated_plots/"
):
    logging.info("Plotting elbow method for k means clustering...")
    sum_of_squared_distances = []
    # drop 'id' column for k-means:
    results_df_k = results_df.drop("id", 1)
    # defines range of clusters to try:
    K = range(1, max_clusters)
    # iterate over the range of clusters specified (1-15):
    for k in K:
        k_means = cluster.KMeans(n_clusters=k)
        k_means = k_means.fit(results_df_k)
        sum_of_squared_distances.append(k_means.inertia_)
    plt.rcParams["figure.figsize"] = [10, 10]
    # plot the number of clusters compared with the k_means inertia:
    plt.plot(K, sum_of_squared_distances, "bx-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Sum of squared distances (k-means inertia)")
    plt.title("Elbow Method to Find Optimal Number of Clusters for K-Means")

    # save figure to specified save directory as 'elbow_method.png':
    plt_save_path = plt_save_dir + "elbow_method.png"
    if os.path.isdir(plt_save_dir) == False:
        os.mkdir(plt_save_dir)
    plt.savefig(plt_save_path)
    plt.show()


"""cluster_images performs k-means clustering on the results_df dataframe of x-y mappings
of each images feature vectors. returns a list of the cluster labels of all the images.

parameters:
'results_df': the dataframe returned by the map_features_to_2d_plane function
'n_clusters': the number of clusters to perform k-means clustering with, n=5 was found to
              be optimal from the elbow-method.
'input_n_clusters': boolean, if set to True will prompt the user for input for the value of n_clusters,
                    to be used if you do not know the optimal value from the elbow method already.
if 'input_n_clusters' is True, then the n_clusters parameter will be ignored.
 """


def cluster_images(results_df, n_clusters=5, input_n_clusters=False):
    if input_n_clusters:
        # as the elbow method is mostly visual, it can be useful to prompt the user to input the number of optimal clusters observed
        n_clusters = input("Enter value for n_clusters (from k-means elbow method): ")
    logging.info("Clustering images with " + str(n_clusters) + " clusters...")
    results_df_k = results_df.drop("id", 1)
    # performs K-Means clustering with the sklearn KMeans function, with n_clusters set to the provided value.
    k_means = cluster.KMeans(n_clusters=n_clusters)
    k_means.fit_predict(results_df_k)
    clusters_means = k_means.cluster_centers_.squeeze()
    image_clusters = k_means.labels_

    print("# of Observations: ", results_df_k.shape)
    print("Clusters Means: ", clusters_means)
    print("labels: ", image_clusters)
    # returns the cluster labels for each image (e.g. 1 to 5)
    return image_clusters


"""plot_clusters plots the cluster labels obtained from cluster_images.
each image point is colored according to its cluster label.

parameters:
'results_df': the dataframe returned by the map_features_to_2d_plane function
'image_clusters': the list of image_clusters returned by cluster_images
'plt_save_dir': directory where to save the plot of image clusters.

"""


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


"""get_attributes_clusters_df combines the attribute_list csv with the image_clusters list

parameters: 
'image_clusters': the list of image_clusters returned by cluster_images
'attribute_list_path': the path to the attribute_list.csv file
'csv_save_dir': directory where to store the newly combined dataframe as a csv
'save_to_csv': boolean, if True will save the combined dataframe to the 'csv_save_dir' provided.
returns:
attribute_list_df: dataframe of the attribute_list.csv combined with the image_clusters list
"""


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


"""build_filtered_dataframe filters the nature images based on the clusters obtained above and
returns a dataframe consisting only of the non-filtered images (the faces).

parameters:
'attribute_list_df': the dataframe returned by get_attributes_clusters_df
'image_num_to_filter': the id of one of the nature images, will be used to find which cluster to filter
'move_filtered': boolean, if True will move all the nature images to the 'removed_images_path'
'save_to_csv': boolean, if True will save the filtered_dataframe to csv in the 'csv_save_dir'
'csv_save_dir': directory where to save the filtered_dataframe
'images_path': directory where all the images are stored
'removed_images_path': directory where to move all the filtered nature images

returns:
filtered_df: a dataframe consisting only of facial images, with the nature images filtered out
"""


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


"""run_image_filtering_from_scratch will run all the functions above in order.
It is a standalone function and it is all you need to run in order to get the filtered dataset
and move all the nature images to their own directory.

After I ran this I checked which images were filtered and all the nature images were correctly filtered,
with no false positives either."""


def run_image_filtering_from_scratch():
    reset_image_subdirs()
    # get dataframe of image features as char arrays:
    features_df = get_features(
        network="resnet50", pooling="max", images_path="./images", num_images=5000
    )
    print(features_df.head())
    # map char arrays of features to 2d plane with PCA and T-sne:
    results_df = map_features_to_2d_plane(features_df)
    # obtain elbow method plot to get n_clusters
    plot_kmeans_elbow_method(
        results_df, max_clusters=15, plt_save_dir="./generated_plots/"
    )
    # optimal n_clusters=5 from elbow method, obtain list of each images cluster label:
    image_clusters = cluster_images(results_df, n_clusters=5)
    # get color-coded plot of the image clusters:
    plot_clusters(results_df, image_clusters, plt_save_dir="./generated_plots/")
    # obtain dataframe of the attribute_list combined with the image_clusters
    attribute_list_df = get_attributes_clusters_df(
        image_clusters,
        attribute_list_path="./attribute_list.csv",
        csv_save_dir="./generated_csv/",
        save_to_csv=True,
    )
    # image number 4 is a scenery image, all similar images (those with the same cluster label) will be filtered:
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


"""run_image_filtering_from_csv is a utility function to skip the feature extraction and clustering process,
it skips straight to separating the nature images using the 'attribute_list_w_clusters.csv' generated after running
the run_image_filtering_from_scratch function. It is useful if the removed images were reset accidently.

Note: can only be ran after the run_image_filtering_from_scratch has been ran at least once.
"""


def run_image_filtering_from_csv():
    attribute_list_df = pd.read_csv("./generated_csv/attribute_list_w_clusters.csv")
    # image number 4 is a scenery image, all similar images (those with the same cluster label) will be filtered
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


run_image_filtering_from_scratch()

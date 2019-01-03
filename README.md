# AMLSassignment
Applied Machine Learning Systems ELEC0132 (18/19) Assignment

## Image Filtering:
---
To run the image filtering code, 
run the following (without the quotes) in the command line:
'python Run_Image_Filtering.py <load_feature_csv>'

<load_feature_csv>: Boolean (True or False), optional: defaults to True,
        whether to load extracted image features from a saved csv, 
        or extract the features with ResNet50 from scratch (may take longer)

Example uses: 
'python Run_Image_Filtering.py True' will run the image filtering from pre-extracted features loaded from a csv. 
'python Run_Image_Filtering.py False' will run the image filtering by extracting image features with ResNet50 from scratch.

---
The reset_image_subdirs is a utility function that 
will move all files contained in sub-directories of
a provided parent directory to that parent directory. 

Use this when you want to reset any symlinks or sorting/seperating
done with the images, such as when you want to train the network for a
different classification task.

Parameters:
'path': should be your image parent directory
'delete_dirs': boolean to determine whether to delete the sub-directories

Returns:
None

---
The get_model function returns a model included in Keras for image classification that was pre-trained on ImageNet. 
There are different networks that this function provides: ResNet50, a 50 layer residual network 
from 'Deep Residual Learning for Image Recognition' 
(https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
which has 25,636,712 network parameters in total. This is the final model I used as the shallower networks just did not
get enough accuracy.

MobileNetV2 has only 3,538,984 parameters and NasNetMobile has only 5,326,716 parameters. 
Both did not get good performance, that is why the use of the much larger ResNet50 is justified.

Two other large networks, Xception and NasNetLarge are provided for testing.

Parameters:
'network': Can be either 'resnet50', 'xception', 'mobilenetv2', or 'nasnetlarge' or 'nasnetmobile', the default and recommended option is 'resnet50'.
'pooling': 'avg', or 'max', determines whether global max pooling or global average pooling will be used, both give comparable results with resnet50.

Returns:
model: Keras model object of the pretrained network selected.

---
get_features returns a pandas dataframe which has one column for image ids and
another for the feature vector of that image. The features are extracted by using one of the
pre-trained networks provided by the get_model function above to 'predict' them.

Parameters:
'img_df': Dataframe of the images and their labels, such as that obtained from pd.read_csv("attribute_list.csv").
'network': Can be either 'resnet50', 'xception', 'mobilenetv2', or 'nasnetlarge' or 'nasnetmobile', the default and recommended option is 'resnet50'.
'pooling': 'avg', or 'max', determines whether global max pooling or global average pooling will be used, both give comparable results with resnet50.
'images_path': should be the path to the directory where your images are stores.
'df_to_pickle': Boolean that decided whether to save the dataframe of image features to a pickle file, specify the save directory with the 'pickle_save_dir' parameter.
'pickle_save_dir': the directory where you want the pickle file of the image features dataframe, only used if 'pickle_to_csv' is set to True.

Returns:
features_df: Dataframe of each images features extracted with the chosen network.

---
pca_tsne_pipeline returns a pipeline of principal component analysis (PCA) and then 
t-distributed stochastic neighbor embedding (T-sne) sci-kit learn functions. 
PCA is a dimensionality-reduction technique, which is used to obtain the principal compenents
from the data. The principal compenents retain most of the variation and patterns
found in the original data, but are much less complex. PCA is used because the sklearn documentation
for T-sne highly recommends another dimensionality reduction method prior to input to T-sne. 

T-sne is another dimensionality reduction technique which maps high-dimensional data to a 2-d or 3-d
plane (in this case 2-d). K-means clustering can then be used on the mapping obtained and the similarity 
of all the images can be visualized. 

Parameters:
None

Returns:
pipe: sklearn pipeline of PCA then T-sne.

---
map_features_to_2d_plane applied the pca_tsne_pipeline function above on
features_df, the dataframe of all the image features.

parameters:
'features_df': the dataframe of image features returned by the get_features function

Returns:
'results_df': a dataframe of the x-y mappings of each of the image's features.

---
plot_kmeans_elbow_method plots the elbow method to find the optimal number of
clusters for k-means clustering. Displays and saves a plot of the k-means inertia 
compared with the number of clusters.

After running this on the data, it is clear from the elbow method that
the optimal number of clusters is n=5.

Parameters:
'results_df': the dataframe returned by map_features_to_2d_plane
'max_clusters': the max range of clusters to test for (e.g. n=1 to n=15)
'plt_save_dir': directory where to save the generated elbow method plot

Returns:
None

---

cluster_images performs k-means clustering on the results_df dataframe of x-y mappings
of each images feature vectors. returns a list of the cluster labels of all the images.

Parameters:
'results_df': the dataframe returned by the map_features_to_2d_plane function
'n_clusters': the number of clusters to perform k-means clustering with, n=5 was found to
            be optimal from the elbow-method.
'input_n_clusters': boolean, if set to True will prompt the user for input for the value of n_clusters,
                    to be used if you do not know the optimal value from the elbow method already.
if 'input_n_clusters' is True, then the n_clusters parameter will be ignored.

Returns:
image_clusters: the cluster labels for each image as an array (e.g. 1 to 5)

---
plot_clusters plots the cluster labels obtained from cluster_images.
each image point is colored according to its cluster label.

Parameters:
'results_df': the dataframe returned by the map_features_to_2d_plane function
'image_clusters': the list of image_clusters returned by cluster_images
'plt_save_dir': directory where to save the plot of image clusters.

Returns:
None

---

get_attributes_clusters_df combines the attribute_list csv with the image_clusters list

Parameters: 
'image_clusters': the list of image_clusters returned by cluster_images
'attribute_list_path': the path to the attribute_list.csv file
'csv_save_dir': directory where to store the newly combined dataframe as a csv
'save_to_csv': boolean, if True will save the combined dataframe to the 'csv_save_dir' provided.

Returns:
attribute_list_df: dataframe of the attribute_list.csv combined with the image_clusters list

---
build_filtered_dataframe filters the nature images based on the clusters obtained above and
returns a dataframe consisting only of the non-filtered images (the faces).

Parameters:
'attribute_list_df': the dataframe returned by get_attributes_clusters_df
'image_num_to_filter': the id of one of the nature images, will be used to find which cluster to filter
'move_filtered': boolean, if True will move all the nature images to the 'removed_images_path'
'save_to_csv': boolean, if True will save the filtered_dataframe to csv in the 'csv_save_dir'
'csv_save_dir': directory where to save the filtered_dataframe
'images_path': directory where all the images are stored
'removed_images_path': directory where to move all the filtered nature images

Returns:
filtered_df: a dataframe consisting only of facial images, with the nature images filtered out

---
run_image_filtering_from_scratch will run all the functions above in order.
It is a standalone function and it is all you need to run in order to get the filtered dataset
and move all the nature images to their own directory.

After I ran this I checked which images were filtered and all the nature images were correctly filtered,
with no false positives either.

Parameters:
None

Returns:
None

---
run_image_filtering_from_csv is a utility function to skip the feature extraction and clustering process,
it skips straight to separating the nature images using the 'attribute_list_w_clusters.csv' generated after running
the run_image_filtering_from_scratch function. It is useful if the removed images were reset accidently.

Note: can only be ran after the run_image_filtering_from_scratch has been ran at least once.

Parameters:
None

Returns:
None

---
## Custom CNN Architecture Classification:
---
To run the custom architecture CNN classifier code, run the following (without the quotes) in the command line:
'python Run_CNN_Classifier.py <row_name> <create_symlinks> <load_trained_model>'

<row_name>: row_name with which to run the classifier (human, eyeglasses, smiling, young, hair_color)
<create_symlinks>: Boolean (True or False), whether to create symbolic links for use with Keras's flow_from_directory function
                   must be True if the symbolic links for the given row_name haven't been created yet (Otherwise an error will occur).
                   must be False if the symbolic links for the given row_name have already been created (Otherwise an error will occur).
<load_trained_model>: Boolean (True or False), optional: defaults to True,
                   whether to load an already trained model, or train the model from scratch (not recommended, may take 30 min to run depending on your hardware)

Example uses: 
'python Run_CNN_Classifier.py young True True' will run the CNN for the young classification task, create symbolic links, and load the already trained model. 
'python Run_CNN_Classifier.py hair_color False False' will run the CNN for the hair color classification task, will not create symbolic links, and train the model from scratch.

## ResNet50 Feature Extraction To Linear SVM Transfer Learning Classification:
---

To run the ResNet50 feature extraction to SVM classifier transfer learning code, 
run the following (without the quotes) in the command line:
'python Run_Transfer_Learning_Classifier.py <row_name> <load_features>'

<row_name>: row_name with which to run the classifier (human, eyeglasses, smiling, young, hair_color)
<load_features>: Boolean (True or False), optional: defaults to True,
                   whether to load extracted image features from a pickle file, 
                   or extract the features with ResNet50 from scratch (may take longer)

Example uses: 
'python Run_Transfer_Learning_Classifier.py young True' will run the SVM classifier for the young classification task by loading the pre-extracted features. 
'python Run_Transfer_Learning_Classifier.py hair_color False' will run the SVM classifier for the hair color classification task by extracting the features from scratch.

## Linear SVM Classification:

To run the linear SVM classifier (on raw RGB pixel values) code, 
run the following (without the quotes) in the command line:
'python Run_Linear_SVM_Classifier.py <row_name> <load_model>'

<row_name>: row_name with which to run the classifier (human, eyeglasses, smiling, young, hair_color)
<load_model>: Boolean (True or False), optional: defaults to True,
        whether to load an already trained model, or train the model from scratch (takes a long time to train due to random search hyperparameter tuning)

Example uses: 
'python Run_Linear_SVM_Classifier.py young True' will run the SVM classifier for the young classification task by loading the pre-trained model. 
'python Run_Linear_SVM_Classifier.py hair_color False' will run the SVM classifier for the hair color classification task by training the model from scratch.
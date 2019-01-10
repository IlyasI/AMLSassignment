# AMLSassignment
Applied Machine Learning Systems ELEC0132 (18/19) Assignment Code Base

**Before running first install the external Python 3 packages listed in requirements.txt**

**Saved models and features which were over GitHub's file size limit can be found on Google Drive: https://goo.gl/wXERtg.**
The saved models and features are necessary if you do not want to train the models from scratch.

This code was only tested on Ubuntu and might not run on Windows.

The saved trained models for the custom CNN classifier were not uploaded to GitHub due to size limitations, however they are available on Google Drive (https://goo.gl/wXERtg) under the 'trained_models' directory. Without these trained models, each network will have to be trained from scratch, taking up to an hour for some tasks.

The file containing the already extracted ResNet50 image features for the transfer learning classifier was also not uploaded due to size restrictions. It can also be found on Google Drive (https://goo.gl/wXERtg) in ./generated_csv/transfer_learning/image_features_resnet50_max.pkl. Extracting these image features from scratch may take up to 30 minutes.

Note: The programs are hardcoded with specific relative paths to directories, therefore the directory structure be kept identical (e.g. images in a './images/' directory, saved trained models in a './trained_models/' directory. Further the files should be kept with the same names (e.g. the saved ResNet50 image feature file should be kept at 'image_features_resnet50_max.pkl'. Otherwise, the paths to the files and directories should be changed throughout the code base.

---
Predictions for each task on an additional unlabelled test set are provided in csv files named task_1_CNN.csv, task_1_ResNet50.csv, etc. The csv files labelled 'CNN' contain predictions obtained with the custom CNN model trained from scratch. The files labelled 'ResNet50' contain predictions obtained with the pretrained ResNet50 to linear SVM classifier transfer learning model.

The 'csv_predictions' folder contain csv files with further predictions on a labelled test set generated from part of the original image dataset. These csv files also show the test accuracy at the top.

---
## Image Filtering:

To run the image filtering code, run the following (without the quotes) in the command line:

'python Run_Image_Filtering.py <load_feature_csv>'

<load_feature_csv>: Boolean (True or False), optional: defaults to True,
        whether to load extracted image features from a saved csv, 
        or extract the features with ResNet50 from scratch (may take longer)

Example uses: 

'python Run_Image_Filtering.py True' will run the image filtering from pre-extracted features loaded from a csv. 

'python Run_Image_Filtering.py False' will run the image filtering by extracting image features with ResNet50 from scratch.

---
## Custom CNN Architecture Classification:

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

---
## ResNet50 Feature Extraction To Linear SVM Transfer Learning Classification:

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

---
## Additional Linear SVM Classification:

Although this was not covered in the report, Linear SVM classifiers were also tested on raw RGB values, rather than image features. They obtained significantly worse results than both CNN classifiers for all tasks.

To run the linear SVM classifier (on raw RGB pixel values) code, run the following (without the quotes) in the command line:

'python Run_Linear_SVM_Classifier.py <row_name> <load_model>'

<row_name>: row_name with which to run the classifier (human, eyeglasses, smiling, young, hair_color)

<load_model>: Boolean (True or False), optional: defaults to True,
        whether to load an already trained model, or train the model from scratch (takes a long time to train due to random search hyperparameter tuning)

Example uses: 

'python Run_Linear_SVM_Classifier.py young True' will run the SVM classifier for the young classification task by loading the pre-trained model. 

'python Run_Linear_SVM_Classifier.py hair_color False' will run the SVM classifier for the hair color classification task by training the model from scratch.

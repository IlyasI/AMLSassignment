import sys

from SVM_classifiers import transfer_learning_svm

'''To run the ResNet50 feature extraction to SVM classifier transfer learning code, 
run the following (without the quotes) in the command line:
'python Run_Transfer_Learning_Classifier.py <row_name> <load_features>'

<row_name>: row_name with which to run the classifier (human, eyeglasses, smiling, young, hair_color)
<load_features>: Boolean (True or False), optional: defaults to True,
                   whether to load extracted image features from a pickle file, 
                   or extract the features with ResNet50 from scratch (may take longer)

Example uses: 
'python Run_Transfer_Learning_Classifier.py young True' will run the SVM classifier for the young classification task by loading the pre-extracted features. 
'python Run_Transfer_Learning_Classifier.py hair_color False' will run the SVM classifier for the hair color classification task by extracting the features from scratch.
'''
def main(row_name, load_features):

    transfer_learning_svm(row_name, load_features=load_features)

if __name__ == "__main__":
    #row_name on which to classify taken from first provided command line argument:
    row_name = str(sys.argv[1])
    #load_features taken from third command line argument (optional: will default to True):
    if len(sys.argv)>2:
        load_features = bool(sys.argv[2])
    else:
        load_features = True

    transfer_learning_svm(row_name, load_features=load_features)

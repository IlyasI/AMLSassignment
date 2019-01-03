import sys

from SVM_classifiers import run_svm_classifier


'''To run the linear SVM classifier (on raw RGB pixel values) code, 
run the following (without the quotes) in the command line:
'python Run_Linear_SVM_Classifier.py <row_name> <load_model>'

<row_name>: row_name with which to run the classifier (human, eyeglasses, smiling, young, hair_color)
<load_model>: Boolean (True or False), optional: defaults to True,
        whether to load an already trained model, or train the model from scratch (takes a long time to train due to random search hyperparameter tuning)

Example uses: 
'python Run_Linear_SVM_Classifier.py young True' will run the SVM classifier for the young classification task by loading the pre-trained model. 
'python Run_Linear_SVM_Classifier.py hair_color False' will run the SVM classifier for the hair color classification task by training the model from scratch.
'''

def main(row_name, load_model):

    run_svm_classifier(row_name, "./svm_saved_models/", load_model=load_model)

if __name__ == "__main__":
    #row_name on which to classify taken from first provided command line argument:
    row_name = str(sys.argv[1])
    #load_model taken from second command line argument (optional: will default to True):
    if len(sys.argv)>2:
        load_model = bool(sys.argv[2])
    else:
        load_model = True

    main(row_name, load_model=load_model)
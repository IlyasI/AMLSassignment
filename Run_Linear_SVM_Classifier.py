import sys

from SVM_classifiers import run_svm_classifier

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
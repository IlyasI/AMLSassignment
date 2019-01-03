import sys

from SVM_classifiers import transfer_learning_svm

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

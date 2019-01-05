import sys

from CNN_classifiers import cnn_classifier
from ImageFiltering import run_image_filtering_from_scratch

'''To run the custom architecture CNN classifier code, run the following (without the quotes) in the command line:
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
'''
def main(row_name, create_symlinks, load_trained_model, trained_model_path):

    cnn_classifier(
        row_name=row_name,
        load_trained_model=load_trained_model,
        trained_model_path=trained_model_path,
        create_symlinks=create_symlinks,
        target_size=(128, 128),
    )

if __name__ == "__main__":
    # row_name on which to classify taken from first provided command line argument:
    row_name = str(sys.argv[1])
    # create_symlinks taken from second command line argument provided:
    create_symlinks = eval(sys.argv[2])
    assert isinstance(create_symlinks, bool), TypeError('create_symlinks should be a bool')
    # load_trained_model taken from third command line argument (optional: will default to True):
    if len(sys.argv) > 3:
        load_trained_model = eval(sys.argv[3])
        assert isinstance(load_trained_model, bool), TypeError('load_trained_model should be a bool')
    else:
        load_trained_model = True
    # run cnn_classifier based on row_name provided as argument.
    # if load_trained_model is True, will load the already trained model from the train_model_path defined for each task.
    if row_name == "human":
        main(
            row_name=row_name,
            create_symlinks=create_symlinks,
            load_trained_model=load_trained_model,
            trained_model_path="trained_models/human/Human_fullmodel.h5",
        )
    elif row_name == "eyeglasses":
        main(
            row_name=row_name,
            create_symlinks=create_symlinks,
            load_trained_model=load_trained_model,
            trained_model_path="trained_models/eyeglasses/Eyeglasses_fullmodel.h5",
        )
    elif row_name == "hair_color":
        main(
            row_name=row_name,
            create_symlinks=create_symlinks,
            load_trained_model=load_trained_model,
            trained_model_path="trained_models/hair_color/Hair_color_fullmodel.h5",
        )
    elif row_name == "smiling":
        main(
            row_name=row_name,
            create_symlinks=create_symlinks,
            load_trained_model=load_trained_model,
            trained_model_path="trained_models/smiling/Smiling_fullmodel.h5",
        )
    elif row_name == "young":
        main(
            row_name=row_name,
            create_symlinks=create_symlinks,
            load_trained_model=load_trained_model,
            trained_model_path="trained_models/young/Young_fullmodel.h5",
        )

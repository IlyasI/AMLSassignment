from CNN_classifiers import cnn_classifier
from ImageFiltering import run_image_filtering_from_scratch
import sys

def main(row_name, create_symlinks, load_trained_model, trained_model_path):

    cnn_classifier(
        row_name=row_name,
        load_trained_model=load_trained_model,
        trained_model_path=trained_model_path,
        create_symlinks=create_symlinks,
        target_size=(128, 128),
    )
if __name__ == "__main__":
    #row_name on which to classify taken from first provided command line argument:
    row_name = str(sys.argv[1])
    #create_symlinks taken from second command line argument provided:
    create_symlinks = bool(sys.argv[2])
    #load_trained_model taken from third command line argument (optional: will default to True):
    if len(sys.argv)>3:
        load_trained_model = bool(sys.argv[3])
    else:
        load_trained_model = True

    if row_name == 'human':
        main(row_name, create_symlinks, load_trained_model=load_trained_model, trained_model_path='trained_models/human/Human_fullmodel.h5')
    elif row_name == 'eyeglasses':
        main(row_name, create_symlinks, load_trained_model=load_trained_model, trained_model_path='trained_models/eyeglasses/Eyeglasses_fullmodel.h5')
    elif row_name == 'hair_color':
        main(row_name, create_symlinks, load_trained_model=load_trained_model, trained_model_path='trained_models/hair_color/Hair_color_fullmodel.h5')
    elif row_name == 'smiling':
        main(row_name, create_symlinks, load_trained_model=load_trained_model, trained_model_path='trained_models/smiling/Smiling_fullmodel.h5')
    elif row_name == 'young':
        main(row_name, create_symlinks, load_trained_model=load_trained_model, trained_model_path='trained_models/young/Young_fullmodel.h5')
          
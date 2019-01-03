from CNN_classifiers import cnn_classifier
from ImageFiltering import run_image_filtering_from_scratch
import sys

def main():
    #row_name on which to classify taken from first provided command line argument:
    row_name = str(sys.argv)[1]

    cnn_classifier(
        row_name='hair_color',
        load_trained_model=True,
        trained_model_path="trained_models/hair_color/Hair_color_fullmodel.h5",
        create_symlinks=True,
        target_size=(128, 128),
    )
if __name__ == "__main__":
    main()
import sys

from ImageFiltering import run_image_filtering_from_csv, run_image_filtering_from_scratch


'''To run the image filtering code, 
run the following (without the quotes) in the command line:
'python Run_Image_Filtering.py <load_feature_csv>'

<load_feature_csv>: Boolean (True or False), optional: defaults to True,
        whether to load extracted image features from a saved csv, 
        or extract the features with ResNet50 from scratch (may take longer)

Example uses: 
'python Run_Image_Filtering.py True' will run the image filtering from pre-extracted features loaded from a csv. 
'python Run_Image_Filtering.py False' will run the image filtering by extracting image features with ResNet50 from scratch.
'''

def main(load_feature_csv):
    if load_feature_csv:
        run_image_filtering_from_csv()
    else:
        run_image_filtering_from_scratch()

if __name__ == "__main__":
    #load_features taken from first command line argument (optional: will default to True):
    if len(sys.argv)>1:
        load_feature_csv = bool(sys.argv[1])
    else:
        load_feature_csv = True

    main(load_feature_csv)

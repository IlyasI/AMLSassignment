import sys

from ImageFiltering import run_image_filtering_from_csv, run_image_filtering_from_scratch

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

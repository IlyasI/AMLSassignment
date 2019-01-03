from classifiers import cnn_classifier


def main():
    cnn_classifier(
        row_name='hair_color',
        load_trained_model=True,
        trained_model_path="trained_models/hair_color/Hair_color_fullmodel.h5",
        create_symlinks=True,
        target_size=(128, 128),
    )
if __name__ == "__main__":
    main()
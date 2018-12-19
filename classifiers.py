import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.models import load_model
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

import CNN_functions

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def plot_training_history(history, classifier_name, plt_save_dir):
    if os.path.isdir(plt_save_dir) == False:
        os.makedirs(plt_save_dir)

    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title(classifier_name + " Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt_save_path = plt_save_dir + classifier_name + "_model_accuracy.png"
    plt.savefig(plt_save_path)
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(classifier_name + " Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt_save_path = plt_save_dir + classifier_name + "_model_loss.png"
    plt.savefig(plt_save_path)
    plt.show()


def train_classifier(
    classifier_name,
    train_path,
    validate_path,
    test_path,
    class_mode="binary",
    optimizer="adam",
    model_save_path="./trained_models/",
    load_trained_model=False,
    trained_model_path=None,
    target_size=(128, 128),
):
    batch_size = 32
    shuffle = True
    seed = 123

    # this is the augmentation configuration we will use for training
    train_datagen = CNN_functions.get_ImageDataGenerator(
        shear_range=0, zoom_range=0, horizontal_flip=False
    )

    # Takes the dataframe and the path to a directory and
    # generates batches of augmented/normalized data

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = CNN_functions.get_ImageDataGenerator()

    if load_trained_model:
        model = load_model(trained_model_path)
    else:
        model = CNN_functions.get_cnn_model(
            optimizer=optimizer,
            class_mode=class_mode,
            multiclass_n_classes=7,
            input_shape=(3,) + target_size,
            channels_first=True,
        )

        if os.path.isdir(model_save_path) == False:
            os.makedirs(model_save_path)
        plot_model(model, to_file=model_save_path + "model.png")

        train_gen = CNN_functions.get_flow_from_directory(
            train_datagen,
            train_path,
            target_size,
            class_mode,
            batch_size,
            shuffle,
            seed,
        )

        validation_gen = CNN_functions.get_flow_from_directory(
            test_datagen,
            validate_path,
            target_size,
            class_mode,
            batch_size,
            shuffle,
            seed,
        )

        earlyStopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=0,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        )

        LR = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=0,
            verbose=1,
            mode="auto",
            min_delta=0.01,
            cooldown=0,
            min_lr=0.001,
        )

        TB = TensorBoard(
            log_dir=model_save_path,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=True,
            write_images=True,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None,
            embeddings_data=None,
            update_freq="batch",
        )
        train_gen.reset()
        validation_gen.reset()

        train_steps = train_gen.n / train_gen.batch_size
        valid_steps = validation_gen.n / validation_gen.batch_size

        logging.info(
            "Train steps: "
            + str(train_steps)
            + "|"
            + "Validation steps: "
            + str(valid_steps)
        )

        logging.info(model.summary())

        history = model.fit_generator(
            generator=train_gen,
            steps_per_epoch=train_steps,
            validation_data=validation_gen,
            validation_steps=valid_steps,
            epochs=50,
            callbacks=[earlyStopping, LR, TB],
            class_weight="auto",
        )

        plot_training_history(history, classifier_name, model_save_path)

        model.save(model_save_path + classifier_name + "_fullmodel.h5")
        model.save_weights(model_save_path + classifier_name + "_weights.h5")

    validation_gen = CNN_functions.get_flow_from_directory(
        test_datagen, validate_path, target_size, class_mode, 1, False
    )

    validation_gen.reset()

    validation_loss = model.evaluate_generator(
        generator=validation_gen, steps=913, verbose=1
    )
    print("Validation Loss: ", validation_loss[0])
    print("Validation Accuracy: ", validation_loss[1])

    test_gen = CNN_functions.get_flow_from_directory(
        test_datagen, test_path, target_size, class_mode, 1, False
    )

    test_gen.reset()

    test_loss = model.evaluate_generator(generator=test_gen, steps=913, verbose=1)

    print("Test Loss: ", test_loss[0])
    print("Test Accuracy: ", test_loss[1])

    test_gen.reset()
    probabilities = model.predict_generator(test_gen, verbose=1, steps=len(test_gen))

    output_df = pd.DataFrame(
        {
            "filename": test_gen.filenames,
            "y_true": test_gen.classes,
            "y_pred": list(map(int, np.rint(probabilities).flatten())),
        }
    )

    output_df_wrong = output_df.query("y_true != y_pred")

    y_true = test_gen.classes
    y_pred = np.rint(probabilities)
    print(confusion_matrix(y_true=y_true, y_pred=y_pred))

    return output_df_wrong


def prepare_for_cnn(row_name):
    filteredDf = pd.read_csv("./generated_csv/attribute_list_filtered.csv")
    train, validate, test = CNN_functions.split_dataset_random(df=filteredDf, seed=0)
    CNN_functions.symlink_classes_images(
        train, validate, test, row_name, reset_symlinks=False
    )


def cnn_classifier(
    row_name,
    load_trained_model=True,
    trained_model_path="",
    create_symlinks=False,
    target_size=(128, 128),
):
    valid_row_names = ["human", "young", "smiling", "eyeglasses", "hair_color"]
    if row_name not in valid_row_names:
        logging.error(
            "row_name supplied to cnn_classifier function is not a valid row name."
        )
        sys.exit(0)
    if row_name == "hair_color":
        class_mode = "multiclass"
    else:
        class_mode = "binary"

    train_path = "./images/train/" + row_name
    validate_path = "./images/validate/" + row_name
    test_path = "./images/test/" + row_name
    classifier_name = row_name.capitalize()

    if create_symlinks:
        prepare_for_cnn(row_name=row_name)

    output_df_wrong = train_classifier(
        classifier_name,
        train_path,
        validate_path,
        test_path,
        class_mode=class_mode,
        optimizer="adam",
        model_save_path="./trained_models/"+row_name+"/",
        load_trained_model=load_trained_model,
        trained_model_path=trained_model_path,
        target_size=target_size,
    )
    print(output_df_wrong)


cnn_classifier(
    row_name="eyeglasses",
    load_trained_model=False,
    trained_model_path="",
    create_symlinks=False,
    target_size=(256, 256),
)

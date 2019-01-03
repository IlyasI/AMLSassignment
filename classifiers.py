import logging
import os
import sys
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.models import load_model
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import CNN_functions

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

"""plot_training_history takes the history object generated during training of the model
and plots both the model accuracy and model loss during each epoch during training.
This function is closely based off of the example provided at the bottom of this documentation page:
https://keras.io/visualization/"""


def plot_training_history(history, classifier_name, plt_save_dir):
    if os.path.isdir(plt_save_dir) == False:
        os.makedirs(plt_save_dir)

    plt.figure(figsize=(1,1))
    plt.rcParams.update({'font.size': 22})
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title(classifier_name + " Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt_save_path = plt_save_dir + classifier_name + "_model_accuracy.png"
    plt.savefig(plt_save_path)
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(1,1))
    plt.rcParams.update({'font.size': 22})
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(classifier_name + " Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt_save_path = plt_save_dir + classifier_name + "_model_loss.png"
    plt.savefig(plt_save_path)
    plt.show()


"""train_classifier trains and tests the model provided using flows from directories.

parameters:
'classifier_name': string, the name you want to give to the classifier, e.g. 'Human'
'train_path': path where the training symlinks are stored
'validate_path': path where the validation symlinks are stored
'test_path': path where the test symlinks are stored
'class_mode': 'binary' or 'categorical'
'optimizer': 'adam' or 'rmsprop' both work well, although any optimizer included in Keras can be used
'model_save_path': where to save the model and its weights after training
'load_trained_model': boolean, if True will load an already trained model from the trained_model_path
                      useful to skip training again and simply obtain predictions on the test set
'trained_model_path': path to the pre-trained model to use if 'load_trained_model' is set to True
'target_size': the dimensions to which all images will be rescaled, set to (128, 128) as that
               is the size expected as input to the CNN model.
"""


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

    # Takes the dataframe and the path to a directory and
    # generates batches of augmented/normalized data

    # this is the data generator configuration we will use for training
    # no augmentation will be done, only rescaling:

    train_datagen = CNN_functions.get_ImageDataGenerator(
        shear_range=0, zoom_range=0, horizontal_flip=False
    )

    # this is the augmentation configuration we will use for testing
    # again, only rescaling:
    test_datagen = CNN_functions.get_ImageDataGenerator()

    if load_trained_model:
        model = load_model(trained_model_path)
    else:
        # instantiates the CNN model with the specified optimizer and class_mode
        # the input shape is (3, 128, 128)
        model = CNN_functions.get_cnn_model(
            optimizer=optimizer,
            class_mode=class_mode,
            categorical_n_classes=6,
            input_shape=(3,) + target_size,
            channels_first=True,
        )
        # creates the model_save_path if it does not exist
        if os.path.isdir(model_save_path) == False:
            os.makedirs(model_save_path)
        # saves a plot of the model architecture
        plot_model(model, to_file=model_save_path + "model.png", show_shapes=True, show_layer_names=False)
        # get flow used for training
        train_gen = CNN_functions.get_flow_from_directory(
            train_datagen,
            train_path,
            target_size,
            class_mode,
            batch_size,
            shuffle,
            seed,
        )
        # get flow used for validation
        validation_gen = CNN_functions.get_flow_from_directory(
            test_datagen,
            validate_path,
            target_size,
            class_mode,
            batch_size,
            shuffle,
            seed,
        )
        # early stopping callback, used to stop training once the validation loss stops
        # improving past a certain difference during an epoch (min_delta=0.001)
        # After early stopping, the model weights will be reset to the training epoch with the lowest validation loss
        earlyStopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=0,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        )
        # reduce learning rate on plateau callback, reduces the learning rate once the
        # validation_loss stops improving past a certain difference during an epoch (min_delta=0.01)
        # This is used because models often benefit from reducing the
        # learning rate once learning stagnates.
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
        # TensorBoard callback, generates a variety of graphs about the model and training:
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
        # resets the training and validation generators prior to training
        # this makes sure the outputs of each will be in the correct order
        train_gen.reset()
        validation_gen.reset()

        # calculates the number of steps needed to iterate over all the images in the dataset:
        train_steps = train_gen.n / train_gen.batch_size
        valid_steps = validation_gen.n / validation_gen.batch_size

        logging.info(
            "Train steps: "
            + str(train_steps)
            + "|"
            + "Validation steps: "
            + str(valid_steps)
        )
        # logs a summary of the model architecture:
        logging.info(model.summary())

        # trains and validates the model using all the callbacks defined above.
        # epochs is set to 50 but in practice does not reach above 10 due to early stopping.
        # class_weight is set to auto to automatically weigh each class based on the number of
        # images contained in each class.
        # This is used because often the classes are imbalanced, e.g. 200 images of cartoons and 600 images of humans
        history = model.fit_generator(
            generator=train_gen,
            steps_per_epoch=train_steps,
            validation_data=validation_gen,
            validation_steps=valid_steps,
            epochs=50,
            callbacks=[earlyStopping, LR, TB],
            class_weight="auto",
        )
        # plots the accuracy and loss during each epoch during training:
        plot_training_history(history, classifier_name, model_save_path)
        # save the fully trained model:
        model.save(model_save_path + classifier_name + "_fullmodel.h5")
        # as a backup, save only the weights of the fully trained model as well:
        model.save_weights(model_save_path + classifier_name + "_weights.h5")

    # get a flow from the validation set with a batch_size of 1 to validate the trained model:
    validation_gen = CNN_functions.get_flow_from_directory(
        test_datagen, validate_path, target_size, class_mode, 1, False
    )

    validation_gen.reset()
    # validate the trained model with all 913 validation images:
    validation_loss = model.evaluate_generator(
        generator=validation_gen, steps=913, verbose=1
    )
    print("Validation Loss: ", validation_loss[0])
    print("Validation Accuracy: ", validation_loss[1])

    # get a flow from the test set with a batch size of 1 to test the trained model:
    test_gen = CNN_functions.get_flow_from_directory(
        test_datagen, test_path, target_size, class_mode, 1, False
    )

    #test_gen.reset()
    # test the trained model with all 913 test images:
    #test_loss = model.evaluate_generator(generator=test_gen, steps=913, verbose=1)

    #print("Test Loss: ", test_loss[0])
    #print("Test Accuracy: ", test_loss[1])

    test_gen.reset()
    # obtain predictions of the class of each of the 913 images in the test set:
    probabilities = model.predict_generator(test_gen, verbose=1, steps=len(test_gen))

    y_true = test_gen.classes
    if class_mode == 'binary':
        y_pred = np.rint(probabilities)
    elif class_mode == 'categorical':
        y_pred = probabilities.argmax(axis=-1)

    filenames = test_gen.filenames
    # output_df is a dataframe that contains each images actual and predicted class 
    output_df, output_df_wrong, report, accuracy, confusion = get_metrics(y_true, y_pred, filenames, flatten_y_pred=True)

    return output_df, output_df_wrong, confusion, report, accuracy

def get_metrics(y_true, y_pred, filenames, flatten_y_pred=False):
    # output_df is a dataframe that contains each images actual and predicted class 
    if flatten_y_pred:
        y_pred_flat = list(map(int, y_pred.flatten()))
    else:
        y_pred_flat = y_pred
    output_df = pd.DataFrame(
        {
            "filename": filenames,
            "y_true": y_true,
            "y_pred": y_pred_flat,
        }
    )
    report = classification_report(y_true, y_pred)
    print(report)
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy)

    # output_df_wrong is a dataframe of only the predictions which were incorrect:
    output_df_wrong = output_df.query("y_true != y_pred")
    # prints the predictions of the model which were wrong:
    print(output_df_wrong)

    confusion = confusion_matrix(y_true=y_true, y_pred=y_pred)
    # print confusion matrix of the true and predicted values
    print(confusion)

    return output_df, output_df_wrong, report, accuracy, confusion


"""
prepare_for_cnn loads the filtered dataframe (with nature images filtered),
splits it into train, validate and test sets and creates symlinks in directories
based on the classes of each image (of the row_name).

parameters:
'row_name': row name to use to create symlinks. e.g. 'human', 'hair_color'.
"""


def prepare_for_cnn(row_name):
    filteredDf = pd.read_csv("./generated_csv/attribute_list_filtered.csv")
    #Drop rows where hair_color = -1, these are mislabeled and impact the accuracy of the trained model
    if row_name == 'hair_color':
        filteredDf = filteredDf[filteredDf.hair_color != -1]
    train, validate, test = CNN_functions.split_dataset_random(df=filteredDf, seed=0)
    CNN_functions.symlink_classes_images(
        train, validate, test, row_name, reset_symlinks=False
    )


"""cnn_classifier trains and tests the CNN for the given row_name

parameters:
'row_name': string, a valid row name from the labels csv.
'load_trained_model': boolean, if True will use an already trained CNN specified by 'trained_model_path'
'create_symlinks': boolean, if True will create symlinks based on the row_name provided
'target_size': size to which to rescale the images. (128, 128) works well without taking too long to train.
"""


def cnn_classifier(
    row_name,
    load_trained_model=True,
    trained_model_path="",
    create_symlinks=False,
    target_size=(128, 128),
):
    # validates the provided row_name, making sure it is a valid row, otherwise logs an error and exits:
    valid_row_names = ["human", "young", "smiling", "eyeglasses", "hair_color"]
    if row_name not in valid_row_names:
        logging.error(
            "row_name supplied to cnn_classifier function is not a valid row name."
        )
        sys.exit(0)
    if row_name == "hair_color":
        class_mode = "categorical"
    else:
        class_mode = "binary"

    # train_path could be './images/train/human' for example:
    train_path = "./images/train/" + row_name
    validate_path = "./images/validate/" + row_name
    test_path = "./images/test/" + row_name

    # capitalizes the first letter in the row_name to obtain the classifier_name (used for labelling plots):
    classifier_name = row_name.capitalize()

    # creates symlinks based on row_name for use with Keras's flow_from_directory function.
    if create_symlinks:
        prepare_for_cnn(row_name=row_name)

    # runs train_classifier function, which trains and evaluates the model:
    output_df, output_df_wrong, confusion, report, accuracy = train_classifier(
        classifier_name,
        train_path,
        validate_path,
        test_path,
        class_mode=class_mode,
        optimizer="adam",
        model_save_path="./trained_models/" + row_name + "/",
        load_trained_model=load_trained_model,
        trained_model_path=trained_model_path,
        target_size=target_size,
    )
    generate_test_predictions_csv(output_df, accuracy, row_name)

def generate_test_predictions_csv(output_df, accuracy, row_name, path_append=None):
    output_df['sort'] = output_df['filename'].str.extract('\/(\d+)\.', expand=False).astype(int)
    output_df = output_df.sort_values('sort', ascending=True).reset_index(drop=True)
    output_df = output_df.drop('sort', axis=1)
    output_df.loc[output_df.y_pred==0, 'y_pred'] = -1

    if row_name == 'smiling':
        task_num = 1
    elif row_name == 'young':
        task_num = 2
    elif row_name == 'eyeglasses':
        task_num=3
    elif row_name == 'human':
        task_num=4
    elif row_name == 'hair_color':
        task_num=5
    if path_append==None:
        csv_filename = 'task_'+str(task_num)+'.csv'
    else:
        csv_filename = 'task_'+str(task_num)+'_'+path_append+'.csv'

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([accuracy])
        for i in range(len(output_df)):
            writer.writerow([output_df['filename'][i], output_df['y_pred'][i]])

'''cnn_classifier(
    row_name='young',
    load_trained_model=True,
    trained_model_path="/home/ilyas/dev/AMLSassignment/trained_models/young/Young_fullmodel.h5",
    create_symlinks=False,
    target_size=(128, 128),
)'''

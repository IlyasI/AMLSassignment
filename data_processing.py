#Import all libraries (as listed in requirements.txt):
import pandas as pd #used for data importing and manipulation
import numpy as np
from PIL import Image #good for generating new image data
import skimage
import glob
import os

image_height = 256
image_width = 256
channels = 4 #rgba

def csv_to_df(csvPath):
    df = pd.read_csv(csvPath)
    return df

def image_data_from_number(imageNumber, imagesPath):
    filename = imagesPath+str(imageNumber)+'.png'
    image = skimage.io.imread(filename)
    return image#raw image data

def create_image_dataset(csvPath, imagesPath):
    df = csv_to_df(csvPath)
    image_shape = (df.shape[0], image_height, image_width, 3)
    dataset = np.ndarray(shape=image_shape,
                         dtype=np.float32)
    for index, row in df.iterrows():
        image = image_data_from_number(row['file_name'], imagesPath)
        #if the image is rgba or grey, convert to rgb for uniform image data shape
        if(image.ndim == 3):
            image_channels = image.shape[2]
            if(image.shape[2]==4):
                image = skimage.color.rgba2rgb(image)
        else:
            image_channels = 1
            image = skimage.color.gray2rgb(image)

        dataset[index] = image

    df = pd.concat([df, pd.DataFrame.from_records(dataset)], axis=1)
    return df

def main():
    currentDirectory = os.path.dirname(__file__)
    csvPath = os.path.join(currentDirectory, 'attribute_list.csv')
    imagesPath =  os.path.join(currentDirectory, 'dataset/')
    dataset = create_image_dataset(csvPath, imagesPath)


    print(dataset)

main()

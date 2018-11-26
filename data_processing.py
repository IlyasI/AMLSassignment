#Import all libraries (as listed in requirements.txt):
import pandas as pd #used for data importing and manipulation
import numpy as np
from PIL import Image #good for generating new image data
from sklearn import cluster
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
    image = skimage.io.imread(filename, as_grey=True)
    return image#raw image data

def create_image_dataset(csvPath, imagesPath):
    df = csv_to_df(csvPath)
    image_shape = (df.shape[0], image_height, image_width, 5)
    flattenedImageShape = (df.shape[0], 65536)
    dataset = np.ndarray(shape=image_shape,
                         dtype=np.float32)
    greyDataset = np.ndarray(shape=flattenedImageShape,
                                  dtype=np.float32)
    for index, row in df.iterrows():
        image = image_data_from_number(row['file_name'], imagesPath)
        downsample = skimage.measure.block_reduce(image, (2,2), np.max)

        #if the image is rgba or grey, convert to rgb for uniform image data shape
        if(0):
            if(image.ndim == 3):
                image_channels = image.shape[2]
                if(image.shape[2]==4):
                    image = skimage.color.rgba2rgb(image)
            else:
                image_channels = 1
                image = skimage.color.gray2rgb(image)

        greyDataset[index] = image.flatten()

    #df = pd.concat([df, pd.DataFrame.from_records(dataset)], axis=1)
    return greyDataset

def cluster_images(images_rgb, path):
    #images_rgb should be array-like or sparse matrix, shape=(n_samples, n_features)
    k_means = cluster.KMeans(n_clusters=3, verbose=1, n_jobs=-1, n_init=3)
    k_means.fit_predict(images_rgb)
    clusters_means = k_means.cluster_centers_.squeeze()
    image_clusters = k_means.labels_

    print('# of Observations:', images_rgb.shape)
    print('Clusters Means:', clusters_means)
    print('labels:', image_clusters)
    np.savetxt('labelsarray.csv', image_clusters)
    return image_clusters

def main():
    currentDirectory = os.path.dirname(__file__)
    csvPath = os.path.join(currentDirectory, 'attribute_list.csv')
    imagesPath =  os.path.join(currentDirectory, 'dataset/')
    dataset = create_image_dataset(csvPath, imagesPath)

    if(os.path.isfile('GreyImages.npy')):
        print('Found and Loading Existing GreyImages.npy...')
        dataset = np.load('GreyImages.npy')
    else:
        print('GreyImages.npy not found... \nGenerating and Saving Data...')
        dataset = create_image_dataset(csvPath, imagesPath)
        np.save('GreyImages', dataset)

    if(0):
        if (os.path.isfile('RGBImagesMatrix.pickle')):
            dataset = np.load('RGBImagesMatrix.pickle')
        else:
            dataset = create_image_dataset(csvPath, imagesPath)
            np.save('RGBImagesMatrix.pickle', dataset)

    image_clusters = cluster_images(dataset, currentDirectory)
    df = csv_to_df(csvPath)
    df['cluster'] = image_clusters
    df.to_csv('attribute_list_w_clusters.csv')


    print(dataset)

main()

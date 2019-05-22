# Cargar Librerias
import os
import sys
import random
import shutil

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import defaultdict

def load_data(_url,_filename, _extract = True):
    zip_dir = tf.keras.utils.get_file(_filename, origin=_url, extract=_extract)
    return zip_dir

def read_pascalvoc(filepath):
    pass

def preparate_dataset():
    #image,label = read_pascalvoc(image)
    raise NotImplementedError

def preprocess_data():
    pass

def augment_image():
    pass

def make_dataset(sources, training=False, batch_size=1,
    num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None):
    """
    Returns an operation to iterate over the dataset specified in sources

    Args:
        sources (list): A list of (lable_id, filepath) pairs.
        training (bool): whether to apply certain processing steps
            defined only in training mode (e.g. shuffle).
        batch_size (int): number of elements the resulting tensor
            should have.
        num_epochs (int): Number of epochs to repeat the dataset.
        num_parallel_calls (int): Number of parallel calls to use in
            map operations.
        shuffle_buffer_size (int): Number of elements from this dataset
            from which the new dataset will sample.

    Returns:
        A tf.data.Dataset object. It will return a tuple images of shape
        [N, H, W, CH] and labels shape [N, 1].
    """
    def load(row):
        filepath = row['image']
        img = tf.io.read_file(filepath)
        img = tf.io.decode_jpeg(img)
        return img, row['label']

    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size*4

    images, labels = zip(*sources)
    ds = tf.data.Dataset.from_tensor_slices({
        'image': list(images), 'label': list(labels)}) 

    if training:
        ds = ds.shuffle(shuffle_buffer_size)
    
    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    #ds = ds.map(lambda x,y: (preprocess_image(x), y))
    #ds = ds.map(lambda x,y: (augmentation_image(x), y))
    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(1)

    return ds

if __name__ == "__main__":
    _URL = 'https://s3-sa-east-1.amazonaws.com/darkanita/DatasetPeople.zip'
    _Filename ='DatasetPeople.zip'
    zip_dir = load_data(_URL,_Filename)
    print(zip_dir)
    prepare_dataset('/Users/anita/.keras/datasets/DatasetPeople')
    
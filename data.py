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
import xml.etree.ElementTree as ET

import xml.etree.ElementTree as ET


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes

name, boxes = read_content("/Users/anita/.keras/datasets/Dataset/4/HV_20.xml")


def load_data(_url,_filename, _extract = True):
    zip_dir = tf.keras.utils.get_file(_filename, origin=_url, extract=_extract)
    return zip_dir
    
def find_sources(data_dir, exclude_dirs=None, file_ext='.jpg', shuffle=True):
    """Find all files with its label.

    This function assumes that data_dir is set up in the following format:
    data_dir/
        - label 1/
            - file1.ext
            - file2.ext
            - ...
        - label 2/
        - .../
    Args:
        data_dir (str): the directory with the data sources in the format
            specified above.
        exclude_dirs (set): A set or iterable with the name of some directories
            to exclude. Defaults to None.
        file_ext (str): Defaults to '.JPG'
        shuffle (bool): whether to shuffle the resulting list. Defaults to True
    
    Returns:
        A list of (lable_id, filepath) pairs.
        
    """
    if exclude_dirs is None:
        exclude_dirs = set()
    if isinstance(exclude_dirs, (list, tuple)):
        exclude_dirs = set(exclude_dirs)

    sources = [
        (os.path.join(data_dir, label_dir, name), int(label_dir),name.split('.')[0],os.path.join(data_dir, label_dir))
        for label_dir in os.listdir(data_dir) 
        for name in os.listdir(os.path.join(data_dir, label_dir)) 
        if label_dir not in exclude_dirs and name.endswith(file_ext)]

    random.shuffle(sources)

    return sources 


def snipping_images(sources,pascalvoc):
    e = ET.fromstring(response.content)
    pass

def prepare_dataset(data_dir, exclude_dirs=None):
    
    sources = find_sources(data_dir, exclude_dirs=exclude_dirs, file_ext='.jpeg')
    pascalvoc = find_sources(data_dir, exclude_dirs=exclude_dirs, file_ext='.xml')

    pd.DataFrame(sources).merge(pd.DataFrame(pascalvoc),how='left', left_on = [''])
    
    fn = lambda x: str(hash(x) % ((sys.maxsize + 1) * 2)) + '.jpg'
    names = [fn(name) for name in filepaths]
    splits = ['train' if random.random() <= 0.7 else 'valid' for _ in names]
    metadata = pd.DataFrame({'label': labels, 'image_name': names, 'split': splits})
    metadata.to_csv('metadata.csv', index=False)
  
    
    os.makedirs('image_files', exist_ok=True)
    for name, fpath in zip(names, filepaths):
        #print(fpath, name)
        try:
            shutil.copy(fpath, os.path.join('image_files', name))
        except Exception as e:
            
            raise e




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
    data_dir = zip_dir[:-10]
    print(zip_dir)
    prepare_dataset('/Users/anita/.keras/datasets/DatasetPeople')
    

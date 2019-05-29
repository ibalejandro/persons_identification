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
        (os.path.join(data_dir, label_dir, name), int(label_dir))
        for label_dir in os.listdir(data_dir) 
        for name in os.listdir(os.path.join(data_dir, label_dir)) 
        if label_dir not in exclude_dirs and name.endswith(file_ext)]

    random.shuffle(sources)

    return sources 

def read_pascalvoc(xml_file: str):

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

def crop_images(pascalvoc):
    #filepath = '/Users/anita/.keras/datasets/Dataset/4/HV_20.jpeg'
    name, boxes = read_pascalvoc(pascalvoc)
    filepath = pascalvoc[:-3]+name.split('.')[-1]
    exists = os.path.isfile(filepath)
    if exists:
        # Store configuration file values
        img = tf.io.read_file(filepath)
        img = tf.io.decode_jpeg(img)
        #pascalvoc = '/Users/anita/.keras/datasets/Dataset/4/HV_20.xml'
        name, boxes = read_pascalvoc(pascalvoc)
        for idbox in range(0,len(boxes)):
            """ offset_height: Vertical coordinate of the top-left corner of the result in the input.
                offset_width: Horizontal coordinate of the top-left corner of the result in the input.
                target_height: Height of the result.
                target_width: Width of the result.
            """
            xmin , ymin , xmax , ymax = boxes[idbox]
            offset_height = ymin
            offset_width = xmin
            target_height = ymax - ymin
            target_width = xmax - xmin
        
            cropped_image_tensor = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
            output_image = tf.image.encode_jpeg(cropped_image_tensor)
            file_name = tf.constant(pascalvoc[:-4]+str(idbox)+'.jpeg')
            print('OK : ' + filepath)
            file = tf.io.write_file(file_name, output_image)
    else:
        # Keep presets
        print('BAD : ' + filepath)

def prepare_dataset(data_dir, exclude_dirs=None):
    pascalvoc = find_sources(data_dir, exclude_dirs=exclude_dirs, file_ext='.xml')
    filepaths, labels = zip(*pascalvoc)
    
    for pascal in filepaths:
        print('Procesar : ' + pascal)
        crop_images(pascal)
        
    
    sources = find_sources(data_dir, exclude_dirs=exclude_dirs, file_ext='.jpeg')
    filepaths, labels = zip(*sources)
    
    fn = lambda x: str(hash(x) % ((sys.maxsize + 1) * 2)) + '.jpeg'
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


if __name__ == "__main__":
    _URL = 'https://s3-sa-east-1.amazonaws.com/darkanita/DatasetPeople.zip'
    _Filename ='DatasetPeople.zip'
    zip_dir = load_data(_URL,_Filename)
    data_dir = zip_dir[:-10]
    print(zip_dir)
    prepare_dataset(data_dir)


    #### Metadata 1 una imagen por clase


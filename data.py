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
import json

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
        #print(filename)
        width  = root.find('size').find('width').text
        #print(width)
        height = root.find('size').find('height').text
        #print(height)
        name = boxes.find('name').text
        #print(name)      
        ymin, xmin, ymax, xmax = None, None, None, None
        for box in boxes.findall('bndbox'):
            ymin = int(box.find('ymin').text)
            #print(ymin)
            xmin = int(box.find('xmin').text)
            #print(xmin)
            ymax = int(box.find('ymax').text)
            #print(ymax)
            xmax = int(box.find('xmax').text)
            #print(xmax)
        list_with_single_boxes = [name, xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, width, height, list_with_all_boxes

def crop_images(pascalvoc):
    filename, width, height, boxes = read_pascalvoc(pascalvoc)
    filepath = pascalvoc[:-3]+filename.split('.')[-1]
    label_ori = filename[:2]
    exists = os.path.isfile(filepath)
    if exists:
        
        img = tf.io.read_file(filepath)
        img = tf.io.decode_jpeg(img)
        image = tf.convert_to_tensor(img, name='image')
        heightTF, widthTF, _ = image.get_shape().as_list()
        
        for idbox in range(0,len(boxes)):
            
            if heightTF == int(height) and widthTF == int(width):
                label, xmin , ymin , xmax , ymax = boxes[idbox]
                offset_height = ymin
                offset_width = xmin
                target_height = ymax - ymin
                target_width = xmax - xmin
                cropped_image_tensor = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
                output_image = tf.image.encode_jpeg(cropped_image_tensor)
                file_name = tf.constant(pascalvoc[:-4].replace(label_ori,label)+'_'+str(idbox)+'.jpeg')
                print('OK : ' + filepath)
                file = tf.io.write_file(file_name, output_image)
            
            elif heightTF == int(width) and widthTF == int(height):
                label, xmin , ymin , xmax , ymax  = boxes[idbox]
                w1 = xmax - xmin
                h1 = ymax - ymin
                offset_height = (heightTF - xmin) - w1
                offset_width = ymin
                target_height = w1
                target_width = h1     
                cropped_image_tensor = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
                output_image = tf.image.encode_jpeg(cropped_image_tensor)
                file_name = tf.constant(pascalvoc[:-4].replace(label_ori,label)+'_'+str(idbox)+'.jpeg')
                print('OK : ' + filepath)
                file = tf.io.write_file(file_name, output_image)
            
            else:
                print('BAD_SHAPE : ' + filepath)
        
        os.remove(filepath)
        print('File Removed! : ' + filepath)
    else:
        print('NOT_EXISTS : ' + filepath)

def split_dataset(ds_len):
    test =  int(ds_len * 0.2)
    split = ['test'] * test
    train = int(ds_len * 0.7)
    split.extend(['train'] * train)
    val = ds_len - (test + train)
    split.extend(['val'] * val)
    random.shuffle(split)
    return split

def prepare_dataset(data_dir, exclude_dirs=None):
    pascalvoc = find_sources(data_dir, exclude_dirs=exclude_dirs, file_ext='.xml')
    filepaths, labels = zip(*pascalvoc)
    
    for pascal in filepaths:
        print('Procesar : ' + pascal)
        crop_images(pascal)
        
    """
    0 : NP
    1 : MJ
    2 : MV
    3 : HJ
    4 : HV 
    """
    sources = find_sources(data_dir, exclude_dirs=exclude_dirs, file_ext='.jpeg')
    filepaths, labels = zip(*sources)
    labels = list(labels)
    dict_labels = {'NP':0,'MJ':1,'MV':2,'HJ':3,'HV':4}
    for i in range(len(filepaths)):
        lbl = filepaths[i].split('/')[-1][:2]
        if labels[i] == dict_labels[lbl]:
            print('Label OK : ' + filepaths[i] + ' Label : ' + str(labels[i]))
        else:
            labels[i] = dict_labels[lbl]
            print('Label Changed : ' + filepaths[i] + ' Label : ' + str(labels[i]))
    labels = tuple(labels)
    fn = lambda x: str(hash(x) % ((sys.maxsize + 1) * 2)) + '.jpeg' 
    names = [fn(name) for name in filepaths]
    #splits = ['train' if random.random() <= 0.6 else 'valid' for _ in names]
    splits = split_dataset(len(names))
    metadata = pd.DataFrame({'filename':filepaths,'label': labels, 'image_name': names, 'split': splits})
    unique_labels = {0 : 'No hay persona', 1 : 'Mujer Joven', 2 : 'Mujer Adulta', 3 : 'Hombre Joven', 4 : 'Hombre Viejo'}
    with open('labels_map.json','w') as f:
        json.dump(unique_labels,f) 

    metadata.to_csv('metadata.csv', index=False)
  
    os.makedirs('image_files', exist_ok=True)
    for name, fpath in zip(names, filepaths):
        #print(fpath, name)
        try:
            shutil.copy(fpath, os.path.join('image_files', name))
        except Exception as e:
            raise e

def preprocess_image(image,img_shape=32):
    image = tf.image.resize(image, size=(img_shape, img_shape))
    image = image / 255.0
    return image
  
def make_dataset(sources, training=False, batch_size=1, num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None, img_shape=32):
    """
    Returns an operation to iterate over the dataset specified in sources

    Args:
        sources (list): A list of (filepath, label_id) pairs.
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
    ds = ds.map(lambda x,y: (preprocess_image(x,img_shape), y))
    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(1)

    return ds

def build_sources_from_metadata(metadata, data_dir, mode='train', exclude_labels=None):
    
    if exclude_labels is None:
        exclude_labels = set()
    if isinstance(exclude_labels, (list, tuple)):
        exclude_labels = set(exclude_labels)

    df = metadata.copy()
    df = df[df['split'] == mode]
    df['filepath'] = df['image_name'].apply(lambda x: os.path.join(data_dir, x))
    include_mask = df['label'].apply(lambda x: x not in exclude_labels)
    df = df[include_mask]

    sources = list(zip(df['filepath'], df['label']))
    return sources

def imshow_batch_of_three(batch):
    with open('labels_map.json') as json_file:  
        data = json.load(json_file)
    label_batch = batch[1].numpy()
    image_batch = batch[0].numpy()
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i in range(3):
        img = image_batch[i, ...]
        axarr[i].imshow(img)
        axarr[i].set(xlabel='label = {}'.format(data[str(label_batch[i])]))
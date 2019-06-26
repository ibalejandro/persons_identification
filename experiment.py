import json
import datetime
import pandas as pd
from data import *
from model import *


models = {
    "lineal" : get_lineal_model,
    "letnet" : get_letnet_model,
    "alexnet" : get_alexnet_model,
    "vgg16l" : get_vgg16_model,
    "google_net" : get_google_net_model
}

def main(config):
    with open(config) as json_file:  
        data = json.load(json_file)

    #Load metadata
    metadata = pd.read_csv(data['metadata'])
    metadata1 = pd.read_csv(data['metadata1'])

    train_sources = build_sources_from_metadata(metadata, 'image_files')
    valid_sources = build_sources_from_metadata(metadata, 'image_files', mode='val')
    test_sources = build_sources_from_metadata(metadata, 'image_files', mode='test')

    train_one_sources = build_sources_from_metadata(metadata1, 'image_files')

    #Make dataset
    train_one_dataset = make_dataset(train_one_sources, training=True,batch_size=data['batch_size'], num_epochs=data['epochs_one'],num_parallel_calls=3, img_shape=data['image_size'])
    train_dataset = make_dataset(train_sources, training=True,batch_size=data['batch_size'], num_epochs=data['epochs'],num_parallel_calls=3, img_shape=data['image_size'])
    valid_dataset = make_dataset(valid_sources, training=False,batch_size=data['batch_size'], num_epochs=data['epochs'],num_parallel_calls=3, img_shape=data['image_size'])

    #Load model
    model_ = models[data['model']](data['num_class'])
    model_.compile(loss=tf.losses.SparseCategoricalCrossentropy(),optimizer=tf.optimizers.Adam(data['learning_rate']),metrics=['accuracy'])

    #Fit model
    history_one = model_.fit(x=train_one_dataset, epochs=data['epochs_one'])
    history = model_.fit(x=train_dataset, epochs=data['epochs'],validation_data=valid_dataset, validation_steps=data['validation_steps'])
    
    #Save learning curve
    draw_result(history_one, data['epochs_one'], data['model'] + "_one_" + str(datetime.date.today()))
    draw_result(history, data['epochs'], data['model'] + "_" + str(datetime.date.today()), True)
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare dataset")

    parser.add_argument('--config', '-d',
        help="path config file"
    )
    
    args = parser.parse_args()
    main(args.config)
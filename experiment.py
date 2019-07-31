import json
import datetime
import pandas as pd
from model import *
from data import *
from tensorflow.keras.callbacks import TensorBoard

models = {
    "lineal" : get_lineal_model,
    "letnet" : get_letnet_model,
    "alexnet" : get_alexnet_model,
    "vgg16" : get_vgg16_model,
    "google_net" : get_google_net_model
}

def main(config, model_name):
    with open(config) as json_file:  
        conf = json.load(json_file)
        conf = conf[model_name]

    #Load metadata
    metadata = pd.read_csv(conf['metadata'])
    metadata1 = pd.read_csv(conf['metadata1'])

    train_sources = build_sources_from_metadata(metadata, 'image_files')
    valid_sources = build_sources_from_metadata(metadata, 'image_files', mode='val')
    test_sources = build_sources_from_metadata(metadata, 'image_files', mode='test')

    train_one_sources = build_sources_from_metadata(metadata1, 'image_files')

    #Make dataset
    train_one_dataset = make_dataset(train_one_sources, training=True,batch_size=conf['batch_size'], 
    num_epochs=conf['epochs_one'],num_parallel_calls=3, img_shape=conf['image_size'])
    train_dataset = make_dataset(train_sources, training=True,batch_size=conf['batch_size'], 
    num_epochs=conf['epochs'],num_parallel_calls=3, img_shape=conf['image_size'])
    valid_dataset = make_dataset(valid_sources, training=False,batch_size=conf['batch_size'], 
    num_epochs=conf['epochs'],num_parallel_calls=3, img_shape=conf['image_size'])


    #Load model
    model = models[model_name](conf['num_class'])
    model.compile(loss=[tf.losses.SparseCategoricalCrossentropy()]*conf['target'],
            optimizer=tf.optimizers.Adam(conf['learning_rate']),
            metrics=['accuracy'])

    #Fit model
    history_one = model.fit(x=train_one_dataset, epochs=conf['epochs_one'])
    history = model.fit(x=train_dataset, epochs=conf['epochs'],
        validation_data=valid_dataset, validation_steps=conf['validation_steps'],callbacks=[TensorBoard()])
    
    #Save learning curve
    draw_result(history_one, conf['epochs_one'], model_name + "_one_" + str(datetime.date.today()))
    draw_result(history, conf['epochs'], model_name + "_" + str(datetime.date.today()), True)
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare dataset")

    parser.add_argument('--config', '-c',
        help="path config file"
    )

    parser.add_argument('--model', '-m',
        help="name of the model"
    )
    
    args = parser.parse_args()
    main(args.config, args.model)
import errno
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from runpy import run_path
from pathlib import Path

from hyperparameter import HyperparameterDict
import models


def parse_config_for_model(config: HyperparameterDict) -> tf.keras.Model:
    optimizer = config.get('optimizer')

    # Prepare model
    if config['model_name'] == 'vae':
        latent_dim = config.get("latent_dim")
        model = models.VAE(keras.Input(shape=config['input_shape']), latent_dim)
        model.build((None, 28, 28, 1))
        model.compile(optimizer=optimizer)

    else:
        raise NotImplementedError

    # If model folder exists load weights
    if Path(config['ckpt_path']).exists():
        model.load_weights(config['ckpt_path'])

    return model


# TODO move to data file and enable loading from various sources, add other dataset
def parse_config_for_data(config: HyperparameterDict) -> tf.data.Dataset:
    def prepare_ds(dataset: tf.data.Dataset, config: HyperparameterDict) -> tf.data.Dataset:
        # Cast to float
        dataset = dataset.map(lambda x: tf.cast(x, tf.float32), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x: config['rescaling'](x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(config['resizing'], num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if config['cache_data']:
            dataset.cache()  # As the dataset fit in memory, cache before shuffling for better performance.
        dataset = dataset.shuffle(1000)  # For true randomness, set the shuffle buffer to the full dataset size.
        dataset = dataset.batch(config['batch_size'])
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    # Prepare dataset
    if config['dataset'] == 'mnist':
        ds_train, ds_validation = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, data_dir=config['data_dir'])
        ds_train = ds_train.map(lambda x: x['image'])
        ds_validation = ds_validation.map(lambda x: x['image'])
    elif config['dataset'] == 'cifar100':
        raise NotImplementedError

    elif config['dataset'] == 'celeb_a':
        ds_train, ds_validation = tfds.load('celeb_a', split=['train', 'validation'],
                                            shuffle_files=True, data_dir=config['data_dir'])
        ds_train = ds_train.map(lambda x: x['image'])
        ds_validation = ds_validation.map(lambda x: x['image'])
    else:
        raise NotImplementedError

    ds_train = prepare_ds(ds_train, config)
    ds_validation = prepare_ds(ds_validation, config)
    return ds_train, ds_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a model")
    parser.add_argument("--config_path", help='Path to configuration file', default='configs/default_config.py')
    args = parser.parse_args()
    config_path = args.config_path

    # Check that config file exit
    if not Path(config_path).is_file():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_path)

    # Run code in config file to get config object
    settings = run_path(config_path)
    config = settings['config']

    print('=========================================')
    print('Run starting with the following settings:')
    print('=========================================')
    print(config)

    model = parse_config_for_model(config)
    model.summary()

    train_ds, val_ds = parse_config_for_data(config)

    # Callbacks
    save_callback = keras.callbacks.ModelCheckpoint(config['ckpt_path'], monitor='val_loss', save_best_only=False,
                                                    save_weights_only=False, mode='min', save_freq='epoch')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config['log_dir'], write_graph=True,
                                                          histogram_freq=5, profile_batch=0, write_images=False)

    model.fit(train_ds, epochs=config.get('epochs'), validation_data=val_ds, callbacks=[save_callback,
                                                                                        tensorboard_callback])

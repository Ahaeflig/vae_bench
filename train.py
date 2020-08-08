import errno
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from runpy import run_path
from pathlib import Path

from hyperparameter import HyperparameterDict
import models


def parse_config_for_model(config: HyperparameterDict) -> tf.keras.Model:
    optimizer = config.get('optimizer')

    # Prepare model
    if config['model_name'] == 'vae':
        latent_dim = config.get("latent_dim")
        model = models.VAE(latent_dim=latent_dim)
        model.compile(optimizer=optimizer)

    else:
        raise NotImplementedError

    return model


def parse_config_for_data(config: HyperparameterDict):
    # Prepare dataset

    if config['dataset'] == 'mnist':
        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
        mnist_digits = np.concatenate([x_train, x_test], axis=0)
        mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
        return mnist_digits

    raise NotImplementedError


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

    print('Run starting with following settings:')
    print(config)

    model = parse_config_for_model(config)
    data = parse_config_for_data(config)
    epochs, batch_size = config.get('epochs'), config.get('batch_size')

    model.fit(data, epochs=30, batch_size=128)

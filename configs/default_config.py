from hyperparameter import Hyperparameter as HP, HyperparameterDict
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomTranslation, RandomFlip, RandomContrast
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop, RandomZoom, Resizing, Rescaling

# Config for MNIST + VAE

SEED = 42

data_augm_dict_base = {'factor': HP('factor', 0.15, np.linspace(0, 0.5, 100), {}, 1),
                       'height_factor': HP('height_factor', 0.1, np.linspace(0, 0.5, 100), {}, 1),
                       'width_factor': HP('width_factor', 0.1, np.linspace(0, 0.5, 100), {}, 1),
                       'seed': HP('seed', SEED, [], {}, 0),
                       'fill_mode': HP('fill_mode', 'reflect', ['constant', 'reflect', 'wrap'], {}, 1),
                       'interpolation': HP('interpolation', 'bilinear', ['nearest', 'bilinear'], {}, 1)}


# Dummies because (hypothesis) the experimental functions signature can't be inspected at runtime to check the expected arguments
def dummy_RandomRotation(factor: float, fill_mode: str, interpolation: str, seed=None, name=None):
    return RandomRotation(factor, fill_mode, interpolation, seed=seed, name=name)


def dummy_Rescaling(scale: float, offset: float, name=None):
    return Rescaling(scale, offset, name=name)


def dummy_Resizing(height: float, width: float, interpolation: str, name=None):
    return Resizing(height, width, interpolation, name=name)
# TODO add other dummies


params = [
    # Constant parameters priority = 0
    HP('model_name', 'vae', [], {}, 0),
    HP('ckpt_path', 'ckpt/vae/', [], {}, 0),
    HP('log_dir', 'logs/vae/', [], {}, 0),
    HP('dataset', 'mnist', [], {}, 0),
    HP('data_dir', 'data/', [], {}, 0),
    HP('epochs', 5, [], {}, 0),
    HP('batch_size', 64, [], {}, 0),
    HP('input_shape', [28, 28, 1], [], {}, 0),
    HP('cache_data', True, [], {}, 0),

    # Mutable parameters
    HP('latent_dim', 2, range(1, 100, 1), {}, 1),
    HP('optimizer', tf.keras.optimizers.SGD,
       [tf.keras.optimizers.Adam, tf.keras.optimizers.SGD],
       {'learning_rate': HP('learning_rate', 0.01, np.linspace(0.000005, 0.5, 500), {}, 1),
        'beta_1': HP('beta_1', 0.9, np.linspace(0.8, 0.999, 100), {}, 1)}, 1),

    # Fixed Data pre_process
    HP('rescaling', dummy_Rescaling, [],
       {'scale': HP('scale', 1.0 / 255, [], {}, 0),
        'offset': HP('offset', 0.0, [], {}, 0),
        'seed': HP('seed', SEED, [], {}, 0)},
       0),

    HP('resizing', dummy_Resizing, [],
       {'height': HP('height', 28, [], {}, 0),
        'width': HP('width', 28, [], {}, 0),
        'interpolation': HP('interpolation', 'bilinear', [], {}, 0),
        'seed': HP('seed', SEED, [], {}, 0)},
       0),

    # RGB Mutable Data Augmentations
    HP('augmentation_1', dummy_RandomRotation,
       [dummy_RandomRotation, RandomTranslation, RandomFlip, RandomContrast],
       dict(data_augm_dict_base, factor=HP('factor', 0.15, np.linspace(0, 0.5, 100), {}, 1)),
       1),

    # Model structure parameters
    # HP('base_filter', 64, range(16, 128, 1), {}, 1),
]

config = HyperparameterDict({p.name: p for p in params})

'''
HP('augmentation_1', RandomRotation,
    [RandomRotation, RandomTranslation, RandomFlip, RandomContrast],
    dict(data_augm_dict_base, factor=HP('factor', 0.15, np.linspace(0, 0.5, 100), {}, 1)),
    1),

HP('augmentation_2', RandomTranslation,
    [RandomRotation, RandomTranslation, RandomFlip, RandomContrast],
    dict(data_augm_dict_base,
        height_factor=HP('height_factor', 0.1, np.linspace(0, 0.5, 100), {}, 1),
        width_factor=HP('width_factor', 0.1, np.linspace(0, 0.5, 100), {}, 1)),
    1),

HP('augmentation_3', RandomFlip,
    [RandomRotation, RandomTranslation, RandomFlip, RandomContrast],
    dict(data_augm_dict_base),
    1),

HP('augmentation_4', RandomFlip,
    [RandomRotation, RandomTranslation, RandomFlip, RandomContrast],
    dict(data_augm_dict_base,
        factor=HP('factor', 0.1, np.linspace(0, 0.5, 100), {}, 1)),
    1),
'''

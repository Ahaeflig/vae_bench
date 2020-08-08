from hyperparameter import Hyperparameter as HP, HyperparameterDict
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomTranslation, RandomFlip, RandomContrast
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop, RandomZoom

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
# TODO add other dummies

params = [
    # Mutable parameters
    HP('latent_dim', 2, range(1, 100, 1), {}, 1),
    HP('optimizer', tf.keras.optimizers.Adam,
       [tf.keras.optimizers.Adam, tf.keras.optimizers.SGD],
       {'learning_rate': HP('learning_rate', 0.01, np.linspace(0.000005, 0.5, 500), {}, 1),
       'beta_1': HP('beta_1', 0.9, np.linspace(0.8, 0.999, 100), {}, 1)}, 1),

    # RGB Mutable Data Augmentations
    HP('augmentation_1', dummy_RandomRotation, 
       [dummy_RandomRotation, RandomTranslation, RandomFlip, RandomContrast],
       dict(data_augm_dict_base, factor=HP('factor', 0.15, np.linspace(0, 0.5, 100), {}, 1)),
       1),
   
    # Model structure parameters
    # HP('base_filter', 64, range(16, 128, 1), {}, 1),

    # Constant parameters priority = 0
    HP('model_name', 'vae', [], {}, 0),
    HP('ckpt_path', 'ckpt/my_model_weights/', [], {}, 0),
    HP('dataset', 'cifar100', [], {}, 0),
    HP('epochs', 30, [], {}, 0),
    HP('batch_size', 2, [], {}, 0)]

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

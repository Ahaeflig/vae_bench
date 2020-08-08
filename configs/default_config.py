from hyperparameter import Hyperparameter as HP, HyperparameterDict
import numpy as np
import tensorflow as tf

params = [
    # Mutable parameters
    HP('latent_dim', 2, range(1, 100, 1), 1),
    HP('optimizer', tf.keras.optimizers.Adam,
       [tf.keras.optimizers.Adam, tf.keras.optimizers.SGD],
       {'learning_rate': HP('learning_rate', 0.01, np.linspace(0.000005, 0.5, 500), {}, 1),
       'beta_1': HP('beta_1', 0.9, np.linspace(0.8, 0.999, 100), {}, 1)},
       1)

    # Constant parameters
    HP('model_name', 'vae', [], {}, 0),
    HP('ckpt_path', 'ckpt/my_model_weights/', [], {}, 0),
    HP('dataset', 'mnist', [], {}, 0),
    HP('epochs', 30, [], {}, 0),
    HP('batch_size', 2, [], {}, 0)]

config = HyperparameterDict({p.name: p for p in params})
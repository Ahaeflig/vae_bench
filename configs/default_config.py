from hyperparameter import Hyperparameter as HP, HyperparameterDict
import numpy as np
import tensorflow as tf

params = [
    # Mutable parameters
    HP('learning_rate', 0.001, np.linspace(0.000005, 0.5, 300), 1),
    HP('latent_dim', 2, range(1, 100, 1), 1),

    # Constant parameters
    HP('optimizer', tf.keras.optimizers.Adam(), [], 0),
    HP('model_name', 'vae', [], 0),
    HP('ckpt_path', 'ckpt/my_model_weights/', [], 0),
    HP('dataset', 'mnist', [], 0),
    HP('epochs', 30, [], 0),
    HP('batch_size', 2, [], 0)]

config = HyperparameterDict({p.name: p for p in params})

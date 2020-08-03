from hyperparameter import Hyperparameter as HP, HyperparameterDict
import numpy as np

params = [
    HP('learning_rate', 0.15, np.linspace(0.000005, 0.1, 300), 1),
    HP('optimizer_name', 'adam', ['adam', 'sgd'], 0),
    HP('model_path', 'model/my_model_weights/', ['model/my_model_weights/'], 0),
    HP('dataset', 'BOEUF', ['BOEUF', 'FRAISE'], 0)
    ]

config = HyperparameterDict({p.name : p for p in params})

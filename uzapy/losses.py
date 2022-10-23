import numpy as np


def cost(loss, y_true, y_pred):
  if loss not in alias:
    raise ValueError(f"loss '{loss}' doesn't exist.")

  if   loss == 'mse':
    return mse(y_true, y_pred)
  elif loss == 'mae':
    return mae(y_true, y_pred)
  elif loss in ['binary_crossentropy', 'log_loss']:
    return binary_crossentropy(y_true, y_pred)
  elif loss == 'categorical_crossentropy':
    return categorical_crossentropy(y_true, y_pred)
  elif loss == 'sparse_categorical_crossentropy':
    return sparse_categorical_crossentropy(y_true, y_pred)

def mse(y_true, y_pred):
  return 1 / y_true.shape[1] * np.sum(np.square(y_true - y_pred))

def mae(y_true, y_pred):
  return 1 / y_true.shape[1] * np.sum(np.absolute(y_true - y_pred))

def binary_crossentropy(y_true, y_pred):
  epsilon = 1e-15
  return 1 / y_true.shape[1] * np.sum(-y_true * np.log(y_pred + epsilon) - (1 - y_true) * np.log(1 - y_pred + epsilon))

def categorical_crossentropy(y_true, y_pred):
  return -np.sum(y_true * np.log(y_pred + 10 **-100)) / y_true.shape[0]

def sparse_categorical_crossentropy(y_true, y_pred):
  pass


alias = [
  'mse',
  'mae',
  'binary_crossentropy', 'log_loss',
  'categorical_crossentropy',
  'sparse_categorical_crossentropy',
]
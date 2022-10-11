import numpy as np


def cost(loss, y_true, y_predicted):
  if loss not in alias:
    raise ValueError(f"loss '{loss}' doesn't exist.")

  if   loss == 'mse':
    return mse(y_true, y_predicted)
  elif loss == 'mae':
    return mae(y_true, y_predicted)
  elif loss == 'binary_crossentropy':
    return binary_crossentropy(y_true, y_predicted)
  elif loss == 'binary_crossentropy':
    return categorical_crossentropy(y_true, y_predicted)
  elif loss == 'binary_crossentropy':
    return sparse_categorical_crossentropy(y_true, y_predicted)

def mse(y_true, y_predicted):
  return 1 / y_true.shape[1] * np.square(y_true - y_predicted)

def mae():
  pass

def binary_crossentropy(y_true, y_predicted):
  epsilon = 1e-15
  return 1 / y_true.shape[1] * np.sum(-y_true * np.log(y_predicted + epsilon) - (1 - y_true) * np.log(1 - y_predicted + epsilon))

def categorical_crossentropy(y_true, y_predicted):
  pass

def sparse_categorical_crossentropy(y_true, y_predicted):
  pass

alias = [
  'mse',
  'mae',
  'binary_crossentropy',
  'categorical_crossentropy',
  'sparse_categorical_crossentropy',
]
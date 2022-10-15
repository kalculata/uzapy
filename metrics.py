import numpy as np


def cost(loss, y_true, y_pred, t_rate=0.5):
  if loss not in alias:
    raise ValueError(f"loss '{loss}' doesn't exist.")

  if   loss == 'mse':
    return mse(y_true, y_pred)
  elif loss == 'mae':
    return mae(y_true, y_pred)
  elif loss in ['binary_crossentropy', 'log_loss']:
    return binary_crossentropy(y_true, y_pred)
  elif loss == 'binary_crossentropy':
    return categorical_crossentropy(y_true, y_pred)
  elif loss == 'binary_crossentropy':
    return sparse_categorical_crossentropy(y_true, y_pred)
  elif loss == 'binary_accuracy':
    return binary_accuracy(y_true, y_pred, t_rate)

def mse(y_true, y_pred):
  return 1 / y_true.shape[1] * np.square(y_true - y_pred)

def mae():
  pass

def binary_crossentropy(y_true, y_pred):
  epsilon = 1e-15
  return 1 / y_true.shape[1] * np.sum(-y_true * np.log(y_pred + epsilon) - (1 - y_true) * np.log(1 - y_pred + epsilon))

def categorical_crossentropy(y_true, y_pred):
  return -np.sum(y_true * np.log(y_pred + 10 **-100)) / y_true.shape[0]

def sparse_categorical_crossentropy(y_true, y_pred):
  pass

def binary_accuracy(y_true, y_pred, threshold=0.5):
  pred_map = lambda x: 0 if x <= threshold else 1

  y_pred_map = np.zeros(shape=y_true.shape)
  for i in range(y_pred.shape[1]):
    y_pred_map[0][i] = pred_map(y_pred[0][i])

  return sum(y_pred_map[0] == y_true[0]) / len(y_pred_map[0])

alias = [
  'mse',
  'mae',
  'binary_crossentropy', 'log_loss',
  'categorical_crossentropy',
  'sparse_categorical_crossentropy',
  'binary_accuracy'
]
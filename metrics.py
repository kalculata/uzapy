import numpy as np


def cost(loss, y_true, y_predicted, t_rate=0.5):
  if loss not in alias:
    raise ValueError(f"loss '{loss}' doesn't exist.")

  if   loss == 'mse':
    return mse(y_true, y_predicted)
  elif loss == 'mae':
    return mae(y_true, y_predicted)
  elif loss in ['binary_crossentropy', 'log_loss']:
    return binary_crossentropy(y_true, y_predicted)
  elif loss == 'binary_crossentropy':
    return categorical_crossentropy(y_true, y_predicted)
  elif loss == 'binary_crossentropy':
    return sparse_categorical_crossentropy(y_true, y_predicted)
  elif loss == 'binary_accuracy':
    return binary_accuracy(y_true, y_predicted, t_rate)

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

def binary_accuracy(y_true, y_predicted, t_rate=0.5):
  import tensorflow as tf
  m = tf.keras.metrics.BinaryAccuracy()
  m.update_state(y_true, y_predicted)
  
  return m.result().numpy()
  FP = 0
  FN = 0
  TP = 0
  TN = 0

  for i in range(y_true.shape[1]):
    y_pred = 0 if y_predicted[0][i] < t_rate else 1
    # False
    if y_true[0][i] == 0:
      if y_pred == 0:
        FP += 1
      else:
        FN += 1
    # True
    elif y_true[0][i] == 1:
      if y_pred == 1:
        TP += 1
      else:
        FN += 1

  return (TP + TN) / (TP + TN + FP + FN)



alias = [
  'mse',
  'mae',
  'binary_crossentropy', 'log_loss',
  'categorical_crossentropy',
  'sparse_categorical_crossentropy',
  'binary_accuracy'
]
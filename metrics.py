import numpy as np


def metrics(metric, y_true, y_pred, threshold=0.5):
  if metric not in alias:
    raise ValueError(f"metric '{metric}' doesn't exist.")

  if   metric == 'binary_accuracy':
    return binary_accuracy(y_true, y_pred, threshold)
  elif metric == 'categorical_accuracy':
    return categorical_accuracy(y_true, y_pred)

def binary_accuracy(y_true, y_pred, threshold=0.5):
  pred_map = lambda x: 0 if x <= threshold else 1

  y_pred_map = np.zeros(shape=y_true.shape)
  for i in range(y_pred.shape[1]):
    y_pred_map[0][i] = pred_map(y_pred[0][i])

  return sum(y_pred_map[0] == y_true[0]) / len(y_pred_map[0])

def categorical_accuracy(y_true, y_pred):
  maxpos = lambda x: np.argmax(x)
  y_true_max = np.array([maxpos(rec) for rec in y_true])
  y_pred_max = np.array([maxpos(rec) for rec in y_pred])

  return sum(y_true_max == y_pred_max) / len(y_true_max)

alias = [
  'binary_accuracy',
  'categorical_accuracy'
]
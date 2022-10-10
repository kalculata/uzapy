import numpy as np


def activation(function, x, alpha=None, scale=None):
  if   function == 'relu':
    return relu(x)

  elif function == 'leaky_relu':
    if alpha is None:
      alpha = 0.01
    return leaky_relu(x, alpha)

  elif function == 'elu':
    if alpha is None:
      alpha = 1.0
    return elu(x)

  elif function == 'selu':
    if alpha is None:
      alpha = 1.67326324
    if scale is None:
      scale = 1.05070098
    return selu(x, alpha, scale)

  elif function == 'sigmoid':
    return sigmoid(x)
  elif function == 'softmax':
    return softmax(x)
  elif function == 'tanh':
    return tanh(x)
  

def relu(x):
  return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
  return np.maximum(alpha * x, x)

def elu(x, alpha=1.0):
  return np.maximum(alpha * (np.exp(x) - 1), x)

def selu(x, alpha=1.67326324, scale=1.05070098):
  return scale * elu(x, alpha)

def sigmoid(x):
  return 1.0/(1+np.exp(-x))

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def tanh(x):
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


alias = [
  'relu',
  'sigmoid',
  'softmax',
  'tanh',
  'leaky_relu',
  'elu',
  'selu'
]
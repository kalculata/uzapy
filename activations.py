import numpy as np


def activate(function, x):
  alpha = None
  scale = None

  if isinstance(function, ActivationFunction):
    alpha = function.alpha
    scale = function.scale
    function = function.name


  if alpha is None:
    if   function == 'leaky_relu':
      alpha = 0.01
    elif function == 'elu':
      alpha = 1.0
    elif function == 'selu':
      alpha = 1.67326324

  if scale is None and function == 'selu':
    scale = 1.05070098
  

  if   function == 'relu':
    return relu(x)
  elif function == 'leaky_relu':
    return leaky_relu(x, alpha)
  elif function == 'elu':
    return elu(x, alpha)
  elif function == 'selu':
    return selu(x, alpha, scale)
  elif function == 'sigmoid':
    return sigmoid(x)
  elif function == 'tanh':
    return tanh(x)
  elif function == 'softmax':
    return softmax(x)
  
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
def tanh(x):
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def softmax(x):
  res = []
  x = x.T

  for row in x:
    e_x = np.exp(row - np.max(row))
    res.append(e_x / e_x.sum())

  return np.array(res).T

class ActivationFunction:
  def __init__(self, name, alpha, scale=None):
    self.name = name
    self.alpha = alpha
    self.scale = scale
  
  def __str__(self) -> str:
    return self.name


class LeakyReLU(ActivationFunction):
  def __init__(self, alpha=0.01):
    super().__init__('leaky_relu', alpha)

class ELU(ActivationFunction):
  def __init__(self, alpha=1.0):
    super().__init__('elu', alpha)


class SELU(ActivationFunction):
  def __init__(self, alpha=1.67326324, scale=1.05070098):
    super().__init__('selu', alpha, scale)

alias = [
  'relu',
  'leaky_relu',
  'selu',
  'elu',
  'sigmoid',
  'softmax',
  'tanh',
]
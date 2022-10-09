import numpy as np
from math import sqrt


alias = [
  'zeros',
  'uniform_distribution',
]

def generate(initializer, fan_in, fan_out=1, for_weights=True):
  if not for_weights:
    fan_out = fan_in
    fan_in = 1

  if initializer == 'uniform_distribution':
    return uniform_distribution(fan_in, fan_out)

  elif initializer == 'zeros':
    return zeros(fan_in, fan_out)

def zeros(fan_in, fan_out):
    return np.zeros((fan_out, fan_in))

def uniform_distribution(fan_in, fan_out):
  return np.random.uniform(-1/sqrt(fan_in), 1/sqrt(fan_in), (fan_out, fan_in))
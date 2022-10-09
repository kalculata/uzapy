import numpy as np
from math import sqrt


alias = [
  'zeros',
  'uniform_distribution',
  'xavier_normal',
  'glorat_normal',
  'xavier_uniform',
  'glorat_uniform',
]

def generate(initializer, fan_in, fan_out=1, for_weights=True):
  if not for_weights:
    fan_out = fan_in
    fan_in  = 1

  if initializer   == 'zeros':
    return zeros(fan_in, fan_out)
  elif initializer == 'uniform_distribution':
    return uniform_distribution(fan_in, fan_out)
  elif initializer == 'xavier_normal' or 'glorat_normal':
    return xavier_normal(fan_in, fan_out)
  elif initializer == 'xavier_uniform' or 'glorat_uniform':
    return xavier_normal(fan_in, fan_out)
  

def zeros(fan_in, fan_out):
    return np.zeros((fan_out, fan_in))

def uniform_distribution(fan_in, fan_out):
  low  = -1/sqrt(fan_in)
  high =  1/sqrt(fan_in)
  return np.random.uniform(low=low, high=high, size=(fan_out, fan_in))

def xavier_normal(fan_in, fan_out):
  std = sqrt(2/(fan_in + fan_out))
  return np.random.normal(loc=0, scale=std, size=(fan_out, fan_in))

def xavier_uniform(fan_in, fan_out):
  low  = -sqrt(6)/sqrt(fan_in + fan_out)
  high =  sqrt(6)/sqrt(fan_in + fan_out)
  return np.random.uniform(low=low, high=high, size=(fan_out, fan_in))

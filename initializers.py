import numpy as np
from math import sqrt


def generate(initializer, fan_in, fan_out=1, for_weights=True):
  if not for_weights:
    fan_out = fan_in
    fan_in  = 1

  if initializer   == 'zeros':
    return zeros(fan_in, fan_out)
  elif initializer == 'standard_distribution':
    return standard_distribution(fan_in, fan_out)
  elif initializer == 'uniform_distribution':
    return uniform_distribution(fan_in, fan_out)
  elif initializer == 'xavier_normal' or 'glorat_normal':
    return xavier_normal(fan_in, fan_out)
  elif initializer == 'xavier_uniform' or 'glorat_uniform':
    return xavier_uniform(fan_in, fan_out)
  elif initializer == 'he_normal':
    return he_normal(fan_in, fan_out)
  elif initializer == 'he_uniform':
    return he_uniform(fan_in, fan_out)
  elif initializer == 'lecun_normal':
    return lecun_normal(fan_in, fan_out)
  elif initializer == 'lecun_uniform':
    return lecun_uniform(fan_in, fan_out)
  

def zeros(fan_in, fan_out):
  return np.zeros((fan_out, fan_in))

def standard_distribution(fan_in, fan_out):
  return np.random.normal(loc=0, scale=1, size=(fan_out, fan_in))

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

def he_normal(fan_in, fan_out):
  std = sqrt(2/fan_in )
  return np.random.normal(loc=0, scale=std, size=(fan_out, fan_in))

def he_uniform(fan_in, fan_out):
  low  = -sqrt(6/fan_in)
  high =  sqrt(6/fan_in)
  return np.random.uniform(low=low, high=high, size=(fan_out, fan_in))

def lecun_normal(fan_in, fan_out):
  std = sqrt(1/fan_in )
  return np.random.normal(loc=0, scale=std, size=(fan_out, fan_in))

def lecun_uniform(fan_in, fan_out):
  low  = -sqrt(3/fan_in)
  high =  sqrt(3/fan_in)
  return np.random.uniform(low=low, high=high, size=(fan_out, fan_in))

alias = [
  'zeros',
  'ones',
  'standard_distribution',
  'uniform_distribution',
  'xavier_normal',
  'glorat_normal',
  'xavier_uniform',
  'glorat_uniform',
  'he_normal',
  'he_uniform',
  'lecun_normal',
  'lecun_uniform',
]
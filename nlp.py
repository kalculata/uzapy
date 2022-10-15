import numpy as np


def one_hot_encoding(classes, data):
  y = []

  for i in range(data.shape[0]):
    one_hot = np.zeros(shape=(classes, 1))
    one_hot[data[i]][0] = 1
    y.append(one_hot)
  
  return np.array(y)
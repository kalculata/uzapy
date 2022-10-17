import matplotlib.pyplot as plt
import numpy as np

from random import randrange


class ImageDataset:
  def __init__(self, X, y) -> None:
    self.X = X
    self.y = y
    
  def random_samples(self):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Random samples', fontsize=20)
    for i in range(10):
      idx = randrange(0, self.X.shape[0])
      plt.subplot(4, 5, i+1)
      plt.imshow(self.X[idx], cmap='gray')
      plt.title(self.y[idx])
      plt.tight_layout()
    plt.show()
  
  def show(self, idx):
    # fig = plt.figure(figsize=(16, 10))
    plt.imshow(self.X[idx], cmap='gray')
    plt.title(self.y[idx])
    plt.show()

def compute_iterations(n, batch_size):
  if batch_size >= n:
    raise ValueError('batch size must be less than n.')
  
  return n // batch_size if (n % batch_size == 0) else (n // batch_size) + 1

def shuffle(x_train, y_train, axis=1):
  indices = np.random.permutation(x_train.shape[1])
  return (np.take(x_train, indices, axis=axis)), (np.take(y_train, indices, axis=axis)) 


import matplotlib.pyplot as plt
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
import numpy as np

from metrics import cost


class Optimezer:
  def __init__(self):
    self.layers = []
    self.optimezer = None
    self.loss = None
    self.metrics = None
    self.lr = 0.01
    self.cost = []

  def forward(self, input):
    activations = {'A0': input}

    for c in range(1, len(self.layers)+1):
      z = self.layers[c-1].output(activations['A' + str(c-1)])
      activations['A' + str(c)] = z

    return activations

  def optimeze(self, epoch, train_data, test_data, batch_size):
    if self.optimezer not in alias:
      raise ValueError(f"optimezer '{self.optimezer}' does'nt exist")

    if self.optimezer == 'gd':
      return self.gd(train_data, test_data)

  def gd(self, train_data, test_data):
    x_train, y_train = train_data
    activations = self.forward(x_train)

    m = y_train.shape[1]
    C = len(self.layers)
    self.cost.append(cost(self.loss, y_train, activations['A' + str(C)]))

    dZ = activations['A' + str(C)] - y_train
    gradients = {}

    for c in reversed(range(1, C+1)):
      gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c-1)].T)
      gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
      
      if c > 1:
        dZ = np.dot(self.layers[c-1].weights.T, dZ) * activations['A' + str(c-1)] * (1 - activations['A' + str(c-1)])

    for c in range(C):
      self.layers[c].weights = self.layers[c].weights - self.lr * gradients['dW' + str(c+1)]
      self.layers[c].biais = self.layers[c].biais - self.lr * gradients['db' + str(c+1)]
    
  def sgd():
    pass

  def sgd_momentum():
    pass

  def rmsprop():
    pass

  def adam():
    pass

  def adadelta():
    pass

  def adagrad():
    pass

  def adamax():
    pass

  def nadam():
    pass

  def ftrl():
    pass


  

  def update_parameters(cost):

    pass

alias = [
  'gd',
  'sgd',
  'sgd_momentum',
  'rmsprop',
  'adam',
  'adadelta',
  'adagrad',
  'adamax',
  'nadam',
  'ftrl'
]
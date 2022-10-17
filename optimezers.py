import numpy as np
from tools import compute_iterations, shuffle


class Optimezer:
  def __init__(self):
    self.x_train      = None
    self.y_train      = None
    self.x_test       = None
    self.y_test       = None
    self.layers       = []
    self.metrics      = []
    self.history      = {'train_cost': [], 'test_cost': []}
    self.optimezer    = None
    self.loss_func    = None
    self.lr           = 0.01
    self.batch_size   = None

  def optimeze(self):
    if self.optimezer not in alias:
      raise ValueError(f"optimezer '{self.optimezer}' does'nt exist")
    if self.batch_size is None and self.optimezer != 'gd':
      raise ValueError('batch size is None')
    
    if self.shuffle and self.optimezer != 'gd':
      self.x_train, self.y_train = shuffle(self.x_train, self.y_train)

    if self.optimezer == 'gd':
      self.gd(self.x_train, self.y_train)
    if self.optimezer == 'sgd':
      self.sgd()

  def forward(self, input):
    activations = {'A0': input}

    for c in range(1, len(self.layers)+1):
      z = self.layers[c-1].output(activations['A' + str(c-1)])
      activations['A' + str(c)] = z

    return activations

  def backward(self, activations, y_train):
    gradients = {}
    C         = len(self.layers)
    dZ        = activations['A' + str(C)] - y_train
    m         = y_train.shape[1]

    for c in reversed(range(1, C+1)): 
      gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c-1)].T)
      gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
      if c > 1:
        dZ = np.dot(self.layers[c-1].weights.T, dZ) * activations['A' + str(c-1)] * (1 - activations['A' + str(c-1)])

    self.update(gradients)

  def update(self, gradients): 
    for c in range(len(self.layers)):
      self.layers[c].weights = self.layers[c].weights - self.lr * gradients['dW' + str(c+1)]
      self.layers[c].biais   = self.layers[c].biais   - self.lr * gradients['db' + str(c+1)]

  def gd(self, x_train, y_train):
    train_activations      = self.forward(x_train)
    self.backward(train_activations, y_train)
    
  def sgd(self):
    n             = self.x_train.shape[1]
    iterations    = compute_iterations(n, self.batch_size)
    itr_start_idx = 0
    itr_end_idx   = self.batch_size

    for itr in range(1, iterations+1):
      x_train = self.x_train[:, itr_start_idx:itr_end_idx]
      y_train = self.y_train[:, itr_start_idx:itr_end_idx]

      self.gd(x_train, y_train)

      if itr != iterations:
        itr_start_idx = itr_end_idx
        itr_end_idx   = itr_end_idx + self.batch_size if (itr_end_idx + self.batch_size < n) else n
    
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
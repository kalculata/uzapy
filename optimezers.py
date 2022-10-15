import numpy as np

from metrics import cost


class Optimezer:
  def __init__(self):
    self.train_data   = None
    self.test_data    = None
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

    if self.optimezer == 'gd':
      return self.gd()
    if self.optimezer == 'sgd':
      return self.sgd()

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

  def gd(self):
    x_train, y_train = self.train_data
    x_test, y_test   = self.test_data

    train_activations      = self.forward(x_train)
    test_activations       = self.forward(x_test)

    train_y_pred           = train_activations['A' + str(len(self.layers))]
    test_y_pred            = test_activations[ 'A' + str(len(self.layers))]

    train_y_pred = train_y_pred.reshape(1, 10, -1)
    test_y_pred  = test_y_pred.reshape(1, 10, -1)

    self.history['train_cost'].append(cost(self.loss_func, y_train, train_y_pred))
    self.history[ 'test_cost'].append(cost(self.loss_func, y_test, test_y_pred))

    for metric in self.metrics:
      if 'train_' + metric not in self.history.keys():
        self.history['train_' + metric] = []
        self.history['test_'  + metric] = []
      self.history['train_' + metric].append(cost(metric, y_train, train_y_pred))
      self.history['test_'  + metric].append(cost(metric, y_train,  test_y_pred))

    self.backward(train_activations, y_train)
    
  def sgd(self):
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
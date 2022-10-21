import numpy as np
from tools import compute_iterations, shuffle


class Model:
  def __init__(self):
    self.layers       = []
    self.metrics      = []
    self.gradients    = {}
    self.lr           = 0.01
    self.epislon      = 1e-7
    self.beta         = 0.9
    self.beta2        = 0.999
    self.x_train      = None
    self.y_train      = None
    self.x_test       = None
    self.y_test       = None
    self.history      = None
    self.optimezer    = None
    self.loss_func    = None
    self.batch_size   = None

  def optimeze(self):
    if isinstance(self.optimezer, OptimezerClass):
      self.epislon   = self.optimezer.epislon
      self.beta      = self.optimezer.beta
      self.beta2     = self.optimezer.beta2
      self.optimezer = self.optimezer.name
    elif self.optimezer not in alias:
      raise ValueError(f"optimezer '{self.optimezer}' does'nt exist")

    if self.batch_size is None and self.optimezer != 'gd':
      raise ValueError('batch size is None')

    if self.shuffle and self.optimezer != 'gd':
      self.x_train, self.y_train = shuffle(self.x_train, self.y_train)

    if self.optimezer == 'gd':
      self.gd(self.x_train, self.y_train)
    elif self.optimezer in alias:
      self.sgd()
    
  def forward(self, input):
    activations = {'A0': input}

    for c in range(1, len(self.layers)+1):
      z = self.layers[c-1].output(activations['A' + str(c-1)])
      activations['A' + str(c)] = z

    return activations

  def backward(self, activations, y_train):
    C  = len(self.layers)
    dZ = activations['A' + str(C)] - y_train
    m  = y_train.shape[1]

    for c in reversed(range(1, C+1)): 
      dW = 1/m * np.dot(dZ, activations['A' + str(c-1)].T)
      db = 1/m * np.sum(dZ, axis=1, keepdims=True)

      if self.optimezer == 'gd' or self.optimezer == 'sgd' or self.optimezer == 'rmsprop':
        self.gradients['dW' + str(c)] = dW
        self.gradients['db' + str(c)] = db

      if self.optimezer == 'sgd_momentum' or self.optimezer == 'adam':
        if 'vW_' + str(c) not in self.gradients.keys() or 'vb_' + str(c) not in self.gradients.keys():
          self.gradients['vW_' + str(c)] = 0
          self.gradients['vb_' + str(c)] = 0

        vW = (self.beta * self.gradients['vW_' + str(c)]) + ((1 - self.beta) * dW)
        vb = (self.beta * self.gradients['vb_' + str(c)]) + ((1 - self.beta) * db)
        self.gradients['vW' + str(c)]  = vW
        self.gradients['vb' + str(c)]  = vb
        self.gradients['vW_' + str(c)] = vW
        self.gradients['vb_' + str(c)] = vb

      if self.optimezer == 'rmsprop' or self.optimezer == 'adam':
        if 'sW_' + str(c) not in self.gradients.keys() or 'sb_' + str(c) not in self.gradients.keys():
          self.gradients['sW_' + str(c)] = 0
          self.gradients['sb_' + str(c)] = 0
        
        sW = (self.beta * self.gradients['sW_' + str(c)]) + ((1 - self.beta2) * np.square(dW))
        sb = (self.beta * self.gradients['sb_' + str(c)]) + ((1 - self.beta2) * np.square(db))
        self.gradients['sW' + str(c)]  = sW
        self.gradients['sb' + str(c)]  = sb
        self.gradients['sW_' + str(c)] = sW
        self.gradients['sb_' + str(c)] = sb

      # ignore calculation of dZ for the input layer
      if c > 1:
        dZ = np.dot(self.layers[c-1].weights.T, dZ) * activations['A' + str(c-1)] * (1 - activations['A' + str(c-1)])
        
    self.update()

  def update(self): 
    for c in range(len(self.layers)):

      if self.optimezer == 'gd' or self.optimezer == 'sgd':
        self.layers[c].weights = self.layers[c].weights - self.lr * self.gradients['dW' + str(c+1)]
        self.layers[c].biais   = self.layers[c].biais   - self.lr * self.gradients['db' + str(c+1)]

      elif self.optimezer == 'sgd_momentum':
        self.layers[c].weights = self.layers[c].weights - self.lr * self.gradients['vW' + str(c+1)]
        self.layers[c].biais   = self.layers[c].biais   - self.lr * self.gradients['vb' + str(c+1)]
      
      elif self.optimezer == 'rmsprop':
        self.layers[c].weights = self.layers[c].weights - self.lr * (self.gradients['dW' + str(c+1)]/np.sqrt(self.gradients['sW' + str(c+1)] + self.epislon))
        self.layers[c].biais   = self.layers[c].biais   - self.lr * (self.gradients['db' + str(c+1)]/np.sqrt(self.gradients['sb' + str(c+1)] + self.epislon))

      elif self.optimezer == 'adam':
        self.layers[c].weights = self.layers[c].weights - self.lr * (self.gradients['vW' + str(c+1)]/np.sqrt(self.gradients['sW' + str(c+1)] + self.epislon))
        self.layers[c].biais   = self.layers[c].biais   - self.lr * (self.gradients['vb' + str(c+1)]/np.sqrt(self.gradients['sb' + str(c+1)] + self.epislon))
      
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
  
class Optimezer:
  def __init__(self, name, epislon, beta=None, beta2=None):
    self.name    = name
    self.epislon = epislon
    self.beta    = beta
    self.beta2   = beta2
  
  def __str__(self) -> str:
    return self.name


class SGDMomentum(Optimezer):
  def __init__(self, epislon=1e-7, beta=0.9):
    return super().__init__('sgd_momentum', epislon, beta)


class RMSProp(Optimezer):
  def __init__(self, epislon=1e-7, beta=0.999):
    return super().__init__('rmsprop', epislon, beta2=beta)
  

class Adam(Optimezer):
  def __init__(self, epislon=1e-7, beta1=0.9, beta2=0.999):
    return super().__init__('adam', epislon, beta1, beta2=beta2)

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
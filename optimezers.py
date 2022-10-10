import numpy as np

from metrics import cost


class Optimezer:
  def __init__(self):
    self.layers = []
    self.optimezer = None
    self.loss = None
    self.metrics = None

  def optimeze(self, epoch, train_data, test_data, batch_size):
    if self.optimezer not in alias:
      raise ValueError(f"optimezer '{self.optimezer}' does'nt exist")

    print(f'Epoch #{epoch + 1}: ', end='')

    if self.optimezer == 'gd':
      return self.gd(train_data, test_data)

  def gd(self, train_data, test_data):
    x_train, y_train = train_data
    predicted = self.forward(x_train)

    _cost = np.sum(cost(self.loss, y_train, predicted))
    print(f'cost={_cost}')

    self.update_parameters(_cost)

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


  def forward(self, input):
    for layer in self.layers:
      output = layer.output(input)
      input  = output
    return output

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
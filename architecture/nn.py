from optimezers import Model


class NeuralNetwork(Model):
  def __init__(self):
    self.layers = []
    self.is_compiled = False
    self.name = None
  
  def add(self, layer):
    self.layers.append(layer)

  def compile(self, loss, optimezer, metrics=[]):
    self.loss = loss
    self.optimezer = optimezer
    self.metrics = metrics

    for layer in self.layers:
      layer.initialize()
    
    self.is_compiled = True

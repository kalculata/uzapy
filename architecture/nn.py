from optimezers import Model


class NeuralNetwork(Model):
  def __init__(self):
    self.layers = []
    self.is_compiled = False
    self.name = None
  
  def add(self, layer):
    self.layers.append(layer)

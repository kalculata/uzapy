from optimezers import Model
from architecture.base import Base

class NeuralNetwork(Model, Base):
  def __init__(self):
    self.layers = []
    self.is_compiled = False
    self.name = None
  
  def add(self, layer):
    self.layers.append(layer)

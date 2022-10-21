from optimezers import Model
from architecture.base import Base

class NeuralNetwork(Model, Base):
  def __init__(self, name=None):
    self.layers = []
    self.is_compiled = False
    self.name = name

  def __str__(self) -> str:
    return 'Neural Network'
  
  def add(self, layer):
    self.layers.append(layer)

class Dropout:
  def __init__(self, frac, name=None):
    self.name = name
    self.trainable     = False

    self.frac = frac

  def __str__(self) -> str:
    return 'Dropout'
  
  def initialize(self):
    pass

  def output(self, input):
    pass

  def info(self):
    return {
      'type'       : self.__str__(),
      'nodes'      : str(100),
      'parameters' : str(0),
      'activation' : '-',
      'name'       : self.name if(self.name) else '',
    }
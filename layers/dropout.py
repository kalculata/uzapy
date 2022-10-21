class Dropout:
  def __init__(self, frac, name=None):
    self.name = name
    self.trainable     = False

    self.frac = frac

  def __str__(self) -> str:
    return 'Dropout'
  
  def _initialize(self):
    pass

  def output(self, input):
    pass

  def info(self):
    return {
      'type'       : self.__str__(),
      'output_shape'      : str(100),
      'parameters' : str(0),
      'activation' : '-',
      'name'       : self.name if(self.name) else '',
    }
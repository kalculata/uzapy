class Dropout:
  def __init__(self, frac, name=None):
    self.name         = name
    self.trainable    = False
    self.output_shape = None

    self.frac         = frac

  def __str__(self) -> str:
    return 'Dropout'
  
  def _initialize(self, prev_output_shape):
    self.output_shape = prev_output_shape

  def output(self, input):
    pass

  def info(self):
    return {
      'type'         : self.__str__(),
      'output_shape' : str(self.output_shape),
      'parameters'   : str(0),
      'activation'   : '-',
      'name'         : self.name if(self.name) else '',
    }
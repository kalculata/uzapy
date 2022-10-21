class Flatten:
  def __init__(self, name=None) -> None:
    self.trainable    = False
    self.name         = name
    self.output_shape = None

  def __str__(self) -> str:
    return 'Flatten'
  
  def _initialize(self, prev_output_shape):
    shape = 1
    for i in prev_output_shape:
      shape *= i
    self.output_shape = (shape, 1)

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
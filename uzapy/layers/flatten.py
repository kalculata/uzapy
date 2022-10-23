class Flatten:
  def __init__(self, input_shape=None, name=None) -> None:
    self.trainable    = False
    self.name         = name

    if isinstance(input_shape, int):
      self.output_shape = (input_shape, 1)
    elif isinstance(input_shape, tuple):
      shape = 1
      for i in input_shape:
        shape *= i
      self.output_shape = (shape, 1)   

  def __str__(self) -> str:
    return 'Flatten'
  
  def _initialize(self, prev_output_shape):
    shape = 1
    for i in prev_output_shape:
      shape *= i
    self.output_shape = (shape, 1)

  def output(self, input):
    return input.reshape(self.output_shape[0], -1)

  def info(self):
    return {
      'type'         : self.__str__(),
      'output_shape' : str(self.output_shape),
      'parameters'   : str(0),
      'activation'   : '-',
      'name'         : self.name if(self.name) else '',
    }
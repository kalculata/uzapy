class Input:
  def __init__(self, shape, name=None):
    self.name          = name
    self.trainable     = False
    self.output_shape  = (shape, 1)

  def __str__(self) -> str:
    return 'Input'

  def output(self, input):
    return input.reshape(self.output_shape[0], -1)

  def info(self):
    return {
      'type'        : self.__str__(),
      'output_shape': str(self.output_shape),
      'parameters' : str(0),
      'activation' : '-',
      'name'       : self.name if(self.name) else '',
    }
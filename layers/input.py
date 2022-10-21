class Input:
  def __init__(self, shape, name=None):
    self.name          = name
    self.trainable     = False
    self.shape         = shape

  def __str__(self) -> str:
    return 'Input'

  def output(self, input):
    return input

  def info(self):
    return {
      'type'       : self.__str__(),
      'nodes'      : str(self.shape),
      'parameters' : str(0),
      'activation' : '-',
      'name'       : self.name if(self.name) else '',
    }
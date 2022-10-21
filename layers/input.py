class Input:
  def __init__(self, shape, name=None):
    self.name = name
    self.trainable     = True
    self.shape = shape

  def __str__(self) -> str:
    return 'Input'
  
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
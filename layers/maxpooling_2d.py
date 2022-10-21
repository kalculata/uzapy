class MaxPooling2D:
  def __init__(self, shape, name=None):
    self.trainable     = False
    self.name = name
    self.shape = shape

  def __str__(self) -> str:
    return 'Maxpooling2D'
  
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
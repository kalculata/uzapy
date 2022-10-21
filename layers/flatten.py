class Flatten:
  def __init__(self, name=None) -> None:
    self.trainable     = False
    self.name = name

  def __str__(self) -> str:
    return 'Flatten'
  
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
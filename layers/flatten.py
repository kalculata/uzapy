class Flatten:
  def __init__(self, name=None) -> None:
    self.trainable     = False
    self.name = name

  def __str__(self) -> str:
    return 'Flatten'
  
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
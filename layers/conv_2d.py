class Conv2D:
  def __init__(self, filters, kernel_shape, strike=1, activation='relu', name=None):
    self.name          = name
    self.trainable     = True

    self.filters = filters
    self.kernel_shape  = kernel_shape
    self.strike        = strike
    self.activation    = activation

  def __str__(self) -> str:
    return 'Conv2D'
  
  def initialize(self):
    pass

  def output(self, input):
    pass

  def info(self):
    parameters = 100

    return {
      'type'       : self.__str__(),
      'nodes'      : str(100),
      'parameters' : str(parameters),
      'activation' : self.activation if(self.activation) else '-',
      'name'       : self.name if(self.name) else '',
    }
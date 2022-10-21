class Conv2D:
  def __init__(self, filters, kernel_shape, strike=1, padding=0, activation='relu', name=None):
    self.name          = name
    self.trainable     = True
    self.output_shape  = None

    self.n_filters = filters
    self.kernel_shape  = kernel_shape
    self.strike        = strike
    self.activation    = activation
    self.padding       = padding

  def __str__(self) -> str:
    return 'Conv2D'
  
  def _initialize(self, prev_output_shape):
    self.output_shape = (prev_output_shape[0] - self.kernel_shape + 1, prev_output_shape[1] - self.kernel_shape + 1, self.n_filters)

  def output(self, input):
    pass

  def info(self):
    parameters = 100

    return {
      'type'         : self.__str__(),
      'output_shape' : str(self.output_shape),
      'parameters'   : str(parameters),
      'activation'   : self.activation if(self.activation) else '-',
      'name'         : self.name if(self.name) else '',
    }
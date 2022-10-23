class MaxPooling2D:
  def __init__(self, kernel_shape=2, stride=2, name=None):
    self.trainable    = False
    self.name         = name
    self.kernel_shape = kernel_shape
    self.stride       = stride

  def __str__(self) -> str:
    return 'Maxpooling2D'
  
  def _initialize(self, prev_output_shape):
    shape             = ((prev_output_shape[0] - self.kernel_shape) // 2) + 1
    self.output_shape = (shape, shape, prev_output_shape[2])

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
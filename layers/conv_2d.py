import numpy as np

class Conv2D:
  def __init__(
    self, 
    filters, 
    kernel_shape, 
    stride=1, 
    padding=False, 
    activation='relu', 
    name=None,
    w_initializer= 'xavier_uniform',
    b_initializer= 'zeros'
  ):
    self.name          = name
    self.trainable     = True
    self.output_shape  = None
    self.parameters    = None

    self.activation    = activation
    self.w_initializer = w_initializer
    self.b_initializer = b_initializer 

    self.filters       = filters
    self.kernel_shape  = kernel_shape
    self.stride        = stride
    self.padding       = padding

  def __str__(self) -> str:
    return 'Conv2D'
  
  def _initialize(self, prev_output_shape):
    shape             = ((prev_output_shape[0] - self.kernel_shape) // self.stride) + 1
    self.output_shape = (shape, shape, self.filters)
    self.parameters   = ((np.square(self.kernel_shape) * self.filters * prev_output_shape[2]) + self.filters)

  def output(self, input):
    pass

  def info(self):
    return {
      'type'         : self.__str__(),
      'output_shape' : str(self.output_shape),
      'parameters'   : str(self.parameters),
      'activation'   : self.activation if(self.activation) else '-',
      'name'         : self.name if(self.name) else '',
    }
class Conv2D:
  def __init__(self, filters, kernel_shape, strike=1, activation='relu', name=None):
    self.filters = filters
    self.kernel_shape  = kernel_shape
    self.strike        = strike
    self.activation    = activation
    self.name          = name

  def log(self):
    pass

  def initialize(self):
    pass

  def output(self, input):
    pass

  def padding(self):
    pass
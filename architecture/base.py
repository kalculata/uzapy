class Base:
  def compile(self, loss, optimezer, metrics=[]):
    self.loss_func = loss
    self.optimezer = optimezer
    self.metrics   = metrics

    if str(self.layers[0]) != 'Input':
      raise ValueError("First layer must be of tye 'Input'")

    prev_output_shape = self.layers[0].output_shape

    for i in range(1, len(self.layers)):
      self.layers[i]._initialize(prev_output_shape)
      prev_output_shape = self.layers[i].output_shape

    self.is_compiled = True

  def log(self):
    if not self.is_compiled:
      raise RuntimeError("model isn't compiled")

    l_type       = len("Name(Type)")
    l_output     = len("Output")
    l_param      = len("Parameters")
    l_activation = len("Activation")
    l_name       = len("Name")

    for layer in self.layers:
      if l_type < len(layer.info()['name']) + len(layer.info()['type']):
        l_type  = len(layer.info()['name']) + len(layer.info()['type'])

      if l_output < len(layer.info()['output_shape']):
        l_output = len(layer.info()['output_shape'])

      if l_param < len(layer.info()['parameters']):
        l_param  = len(layer.info()['parameters'])

      if l_activation < len(layer.info()['activation']):
        l_activation = len(layer.info()['activation'])
      
      if l_name < len(layer.info()['name']):
        l_name = len(layer.info()['name'])

    headers   = f"Name(Type) {' '*(l_type+2-10)} Output {' '*(l_output+2-6)} Parameters {' '*(l_param+2-10)} Activation {' '*(l_activation+2-10)}"
    line_size = len(headers)
    print(headers)
    print("="*line_size)

    n_parameters = 0

    for layer in self.layers:
      name_type  = f"{layer.info()['name']}({layer.info()['type']})"
      type   		 = f"{name_type} {' '*(l_type+2-len(name_type))}"
      output 		 = f"{layer.info()['output_shape']} {' '*(l_output+2-len(layer.info()['output_shape']))}"
      param 		 = f"{layer.info()['parameters']} {' '*(l_param+2-len(layer.info()['parameters']))}"
      activation = f"{layer.info()['activation']} {' '*(l_activation+2-len(layer.info()['activation']))}"
      name 			 = f"{layer.info()['name']      } {' '*(l_name+2-len(layer.info()['name']))}"

      print(f"{type} {output} {param} {activation} {name}")
      print("_"*line_size)

      n_parameters += int(layer.info()['parameters'])

    print("Model Name      : ", self.name)
    print("Total paramters : ", n_parameters)


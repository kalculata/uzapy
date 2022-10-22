from tqdm import tqdm

from losses import cost
from metrics import metrics
import matplotlib.pyplot as plt

class Base:
  def compile(self, loss, optimezer, metrics=[]):
    self.loss_func = loss
    self.optimezer = optimezer
    self.metrics   = metrics

    if str(self.layers[0]) != 'Input' and str(self) != 'Dense Neural Network':
      raise ValueError("First layer must be of tye 'Input'")

    if str(self) == 'Dense Neural Network':
      prev_output_shape = self.input_nodes
      start = 0
    else:
      start = 1
      prev_output_shape = self.layers[0].output_shape

    for i in range(start, len(self.layers)):
      self.layers[i]._initialize(prev_output_shape)
      prev_output_shape = self.layers[i].output_shape

    self.is_compiled = True

  def train(self, train_data, test_data=None, lr=0.01, epoch=10, batch_size=None, shuffle=False, verbose=False):
    if not self.is_compiled:
      raise RuntimeError("model isn't compiled")
      
    self.x_train, self.y_train = train_data
    self.x_test, self.y_test   = test_data
    self.lr = lr
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.history = {'train_cost': [], 'test_cost': []}

    if verbose == True:
      for _epoch in range(epoch):
        self.optimeze()
        self.evaluate()

        _train_loss = round(self.history["train_cost"][_epoch], 2)
        _test_loss  = round(self.history["test_cost"][_epoch], 2)
        print(f'Epoch #{_epoch + 1}: train loss={_train_loss}, test loss={_test_loss}{self._display_metrics(_epoch)}')	
    else:
      for _epoch in tqdm(range(epoch)):
        self.optimeze()
        self.evaluate()

  def _display_metrics(self, epoch):
    output = ""
    for metric in self.metrics:
      output += ", train_" + metric + "=" + str(round(self.history['train_' + metric][epoch], 2))
      output += ", test_"  + metric + "=" + str(round(self.history['test_' + metric][epoch], 2))

    return output

  def evaluate(self):
    test_activations       = self.forward(self.x_test)
    train_activations      = self.forward(self.x_train)
    test_y_pred            = test_activations[ 'A' + str(len(self.layers))]
    train_y_pred           = train_activations['A' + str(len(self.layers))]

    self.history[ 'test_cost'].append(cost(self.loss_func, self.y_test, test_y_pred))
    self.history['train_cost'].append(cost(self.loss_func, self.y_train, train_y_pred))

    for metric in self.metrics:
      if 'test_' + metric not in self.history.keys():
        self.history['train_'  + metric] = []
        self.history['test_'  + metric] = []
      self.history['train_' + metric].append(metrics(metric, self.y_train, train_y_pred))
      self.history['test_'  + metric].append(metrics(metric, self.y_test,  test_y_pred))
		
  def predict(self, input):
    return self.forward(input)['A' + str(len(self.layers))]
	
  def plot_history(self, metric, lloc='up', plot='all'):
    if plot == 'all' or plot == 'train':
      plt.plot(self.history['train_' + metric], label='train ' + metric)
    if plot == 'all' or plot == 'test':
      plt.plot(self.history['test_'  + metric], label="test " + metric)

    if lloc == 'up':
      plt.legend(loc="upper right")
    elif lloc == 'down':
      plt.legend(loc="lower right")

    plt.show()

  def performance(self):
    for metric in self.metrics:
      print(f'train {metric}: ', round(self.history['train_' + metric][-1], 5))
      print(f'test  {metric} : ',  round(self.history['test_'  + metric][-1], 5))

  def log(self):
    if not self.is_compiled:
      raise RuntimeError("model isn't compiled")

    l_type       = len("Name(Type)")
    l_output     = len("Output shape")
    l_param      = len("#Parameters")
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

    headers   = f"Name(Type) {' '*(l_type+2-10)} Output shape {' '*(l_output+2-12)} #Parameters {' '*(l_param+2-11)} Activation {' '*(l_activation+2-10)}"
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

    print("Total parameters : ", n_parameters)
    if str(self) == 'Dense Neural Network':
      print("Input Nodes      : ", self.input_nodes)
    print("Model Name       : ", self.name)
    print("Model Type       : ", self)



import matplotlib.pyplot as plt

from tqdm import tqdm
from activations import ActivationFunction
from architecture.base import Base
from layers.dense import Dense
from losses import cost
from metrics import metrics
from optimezers import Model


class DNN(Model, Base):
	def __init__(self, input_nodes):
		super().__init__()

		self.input_nodes = input_nodes
		self.is_compiled = False
		self.name				 = None
	
	def __str__(self) -> str:
		return 'Dense Neural Network'

	def add(self, nodes, activation='relu', name=None, w_initializer='xavier_uniform', b_inititializer='zeros'):
		if not isinstance(activation, (str, ActivationFunction)) and activation is not None:
			raise ValueError('Activation function must be str or ActivationFunction instance')

		self.layers.append(Dense(
			nodes=nodes, 
			activation=activation, 
			w_initializer=w_initializer, 
			b_initializer=b_inititializer,
			name=name
		))

	def set_name(self, name):
		self.name = name

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

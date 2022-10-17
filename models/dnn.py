import matplotlib.pyplot as plt

from tqdm import tqdm
from activations import ActivationFunction
from layers.dense import Dense
from losses import cost
from metrics import metrics
from optimezers import Optimezer


class DNN(Optimezer):
	def __init__(self, input_nodes):
		super().__init__()

		self.input_nodes = input_nodes
		self.is_compiled = False

	def add(self, nodes, activation='relu', name=None, w_initializer='xavier_uniform', b_inititializer='zeros'):
		if not isinstance(activation, (str, ActivationFunction)) and activation is not None:
			raise ValueError('Activation function must be str or ActivationFunction instance')

		if (len(self.layers) == 0):
			self.layers.append(Dense(name, self.input_nodes, nodes, activation, w_initializer, b_inititializer))
		else:
			self.layers.append(Dense(name, self.layers[-1].nodes, nodes, activation, w_initializer, b_inititializer))

	def log(self):
		print("Nodes \t Parameters \t Activation \t Name")
		for layer in self.layers:
			layer.log()

	def compile(self, loss, optimizer, metrics=[]):
		self.loss_func = loss
		self.optimezer = optimizer
		self.metrics   = metrics

		for layer in self.layers:
			layer.initialize()

		self.is_compiled = True

	def train(self, train_data, test_data=None, lr=0.01, epoch=10, batch_size=None, shuffle=False, verbose=False):
		self.x_train, self.y_train = train_data
		self.x_test, self.y_test   = test_data
		self.lr = lr
		self.batch_size = batch_size
		self.shuffle = shuffle
		
		if verbose == True:
			for _epoch in range(epoch):
				self.optimeze()
				self.evaluate()
				print(f'Epoch #{_epoch + 1}: loss={self.history["train_cost"][_epoch]}')	
		else:
			for _epoch in tqdm(range(epoch)):
				self.optimeze()
				self.evaluate()
		
		return self.history
	
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

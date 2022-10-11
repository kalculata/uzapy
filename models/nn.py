import matplotlib.pyplot as plt

from tqdm import tqdm
from activations import ActivationFunction
from layers.dense import Dense
from optimezers import Optimezer


class NeuralNetwork(Optimezer):
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

	def compile(self, loss, optimizer='gd', metrics=[]):
		self.loss_func = loss
		self.optimezer = optimizer
		self.metrics   = metrics

		for metric in self.metrics:
			self.history['train_' + metric] = []
			self.history['test_'  + metric] = []

		for layer in self.layers:
			layer.initialize()

		self.is_compiled = True

	def train(self, train_data, test_data=None, lr=0.01, epoch=10, batch_size=None, verbose=0):
		self.lr = lr
		
		if verbose == 1:
			for _epoch in range(epoch):
				self.optimeze(train_data, test_data, batch_size)
				print(f'Epoch #{_epoch + 1}: loss={self.history["train_cost"][_epoch]}')	
		else:
			for _epoch in tqdm(range(epoch)):
				self.optimeze(train_data, test_data, batch_size)
		
	def predict(self, input):
		return self.forward(input)['A' + str(len(self.layers))]
	
	def plot_history(self, metric, lloc='up'):
		plt.plot(self.history['train_' + metric], label="train "+ metric)
		plt.plot(self.history['test_'  + metric], label="test " + metric)

		if lloc == 'up':
			plt.legend(loc="upper right")
		elif lloc == 'down':
			plt.legend(loc="lower right")
			
		plt.show()

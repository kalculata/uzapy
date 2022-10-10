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

	def compile(self, loss, optimizer='gd', metrics=None):
		self.loss      = loss
		self.optimezer = optimizer
		self.metrics   = metrics

		for layer in self.layers:
			layer.initialize()

		self.is_compiled = True

	def train(self, train_data, test_data, epoch=10, batch_size=None, verbose=0):
		for _epoch in range(epoch):
			self.optimeze(_epoch, train_data, test_data, batch_size)

	def predict(self, input):
		return self.forward(input, self.layers)

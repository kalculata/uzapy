from layers.dense import Dense


class NeuralNetwork:
	def __init__(self, input_nodes):
		self._input_nodes = input_nodes
		self._layers = []
		self._is_compiled = False

		self._optimezer = None
		self._loss = None
		self._metrics = None

	def add(self, nodes, activation=None, name=None):
		if (len(self._layers) == 0):
			self._layers.append(Dense(self._input_nodes, nodes, activation, name))
		else:
			self._layers.append(Dense(self._layers[-1].nodes, nodes, activation, name))

	def log(self):
		print("Nodes \t Parameters \t Activation \t Name")
		for layer in self._layers:
			layer.log()

	def compile(self, loss, optimizer='gd', metrics=None):
		self._loss = loss
		self._optimezer = optimizer
		self._metrics = metrics

		for layer in self._layers:
			layer.initialize()

		self._is_compiled = True

	def train(self, train_data, test_data, epoch=10, batch_size=None, verbose=0):
		pass

	def _forward(self, input):
		for layer in self.layers:
			output = layer.output(input)
			input = output

	def _backward(self):
		pass

	def predict(self, input):
		return self._forward(input)

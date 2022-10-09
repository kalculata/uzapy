from layers.dense import Dense


class NeuralNetwork:
	def __init__(self, input_nodes):
		self.input_nodes = input_nodes
		self.layers      = []
		self.is_compiled = False
		self.optimezer   = None
		self.loss        = None
		self.metrics     = None

	def add(self, nodes, activation=None, name=None, w_initializer='uniform_distribution', b_inititializer='zeros'):
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
		pass

	def _forward(self, input):
		for layer in self.layers:
			output = layer.output(input)
			input  = output
		return output

	def _backward(self):
		pass

	def predict(self, input):
		return self.forward(input)
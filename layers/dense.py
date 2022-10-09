class Dense:
	def __init__(self, input_nodes, nodes, activation, name):
		self.input_nodes = input_nodes
		self.nodes = nodes
		self.activation = activation
		self.name = name

		self.weights = None
		self.biais = None

	def log(self):
		parameters = (self.input_nodes * self.nodes) + self.nodes
		print(f"{self.nodes}\t {parameters}\t {self.activation}\t {self.name}")

	def initialize(self):
		pass

	def output(self, input):
		return None



from initializers import generate, alias as initializers_name
from activations import ActivationFunction, activate, alias as activations_name



class Dense:
	def __init__(self, name, input_nodes, nodes, activation, w_initializer, b_initializer):
		self.input_nodes 	 = input_nodes
		self.nodes       	 = nodes
		self.activation  	 = activation
		self.name        	 = name
		self.weights     	 = None
		self.biais       	 = None
		self.w_initializer = w_initializer
		self.b_initializer = b_initializer

	def log(self):
		parameters = (self.input_nodes * self.nodes) + self.nodes
		print(f"{self.nodes}\t {parameters}\t {self.activation}\t {self.name}")

	def initialize(self):
		if self.w_initializer not in initializers_name:
			raise ValueError(f"initializer '{self.w_initializer}' does't exist.")
		if self.b_initializer not in initializers_name:
			raise ValueError(f"initializer '{self.w_initializer}' does't exist.")
		
		self.weights = generate(self.w_initializer, self.input_nodes, self.nodes)
		self.biais   = generate(self.b_initializer, self.nodes, for_weights=False)

	def output(self, input):
		_output = self.weights.dot(input) + self.biais

		if self.activation is None:
			return _output
		if not isinstance(self.activation, ActivationFunction) and self.activation not in activations_name:
			raise ValueError(f"activation '{self.activation}' does't exist.")
		return activate(self.activation, _output)

from initializers import generate, alias as initializers_name
from activations import ActivationFunction, activate, alias as activations_name



class Dense:
	def __init__(self, input_nodes, nodes, activation, w_initializer='xavier_uniform', b_initializer='zeros', name=None):
		self.name        	 = name
		self.trainable     = True

		self.input_nodes 	 = input_nodes
		self.nodes       	 = nodes
		self.activation  	 = activation
		self.weights     	 = None
		self.biais       	 = None
		self.w_initializer = w_initializer
		self.b_initializer = b_initializer

	def __str__(self):
		return 'Dense'

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

	def info(self):
		parameters = (self.input_nodes * self.nodes) + self.nodes

		return {
			'type'       : self.__str__(),
			'nodes'      : str(self.nodes),
			'parameters' : str(parameters),
			'activation' : self.activation if(self.activation) else '',
			'name'       : self.name if(self.name) else '',
		}
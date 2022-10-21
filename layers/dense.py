from initializers import generate, alias as initializers_name
from activations import ActivationFunction, activate, alias as activations_name



class Dense:
	def __init__(self, nodes, activation, w_initializer='xavier_uniform', b_initializer='zeros', name=None):
		self.name        	 = name
		self.trainable     = True
		self.output_shape  = None
		self.parameters    = None

		self.activation  	 = activation
		self.w_initializer = w_initializer
		self.b_initializer = b_initializer

		self.nodes       	 = nodes
		self.weights     	 = None
		self.biais       	 = None

	def __str__(self):
		return 'Dense'

	def _initialize(self, prev_input_nodes):
		if isinstance(prev_input_nodes, tuple):
			prev_input_nodes = prev_input_nodes[0]

		self.input_nodes = prev_input_nodes
		self.output_shape = (self.nodes, 1)
		self.parameters = (self.input_nodes * self.nodes) + self.nodes

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
		return {
			'type'         : self.__str__(),
			'output_shape' : str(self.output_shape),
			'parameters'   : str(self.parameters),
			'activation'   : self.activation if(self.activation) else '',
			'name'         : self.name if(self.name) else '',
		}
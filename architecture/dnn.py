import matplotlib.pyplot as plt


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


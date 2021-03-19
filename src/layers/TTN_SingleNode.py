import tensorflow    as tf
import numpy         as np
import math
import tensornetwork as tn

from tensorflow.keras.layers import Layer 
from tensorflow.keras        import activations
from tensorflow.keras        import initializers
from tensorflow.keras        import regularizers
from typing                  import List, Optional, Text, Tuple

from tensornetwork.network_components import Node


# Create keras layer using the dedicated wrapper
@tf.keras.utils.register_keras_serializable(package='tensornetwork')
# Layer class
class TTN_SingleNode(Layer):
	# layer initialization 
	def __init__(self, 
		n_contraction     : int                               , #number of features to contract to each weight tensor
		bond_dim          : int                               , #bond dimension of the weight tensors
		use_bias          : Optional[bool] = True             , #Use (or not) bias vector after each layer
		activation        : Optional[Text] = None             , #Activation function to use after each layer
		kernel_initializer: Optional[Text] = 'glorot_uniform' , #weight initialization
		bias_initializer  : Optional[Text] = 'zeros'          , #bias initialization
		kernel_regularizer                 = None             , #regularization function
		**kwargs												#parameters for parent class (such as input_shape)
		) -> None:

		# Allow specification of input_dim instead of input_shape,
		# for compatability with Keras layers that support this
		if 'input_shape' not in kwargs and 'input_dim' in kwargs:
			kwargs['input_shape'] = (kwargs.pop('input_dim'),)
		
		#initialize parent class with its paramenters
		super().__init__(**kwargs)

		#intiialize layer self values
		self.n_contraction      = n_contraction
		self.bond_dim           = bond_dim
		self.nodes              = [] #list of nodes, each node is a tensor to be contracted
		self.use_bias           = use_bias
		self.activation         = activations.get(activation         ) #get activation function from tesnorflow
		self.kernel_initializer = initializers.get(kernel_initializer) #get initializers
		self.bias_initializer   = initializers.get(bias_initializer  ) #get initializers
		self.kernel_regularizer = regularizers.get(kernel_regularizer) #get regularizer

	#layer building
	def build(self, input_shape: List[int]) -> None:

		#some checks have to be made in order to avoid consequent errors
		#input shape must be well defined
		if input_shape[-1] is None:
			raise ValueError('The last dimension of the inputs should be defined. Found `None`.')
		input_dim = int(input_shape[-1])
		
		#get number of features
		self.n_features = input_shape[1]
		#check is number of provided features is divisible 
		#by the required number of contractions
		assert self.n_features%self.n_contraction == 0, "Number of features must be divisible by number of contractions"
		
		#compute number of tensors needed for contraction
		self.n_weights = self.n_features//self.n_contraction
		#build parent layer
		super().build(input_shape)

		#get feature dimension, needed for tensor sizes
		self.feature_dim  = input_shape[2]
		#compute shape of the weights tensors
		w_shape = tuple([self.feature_dim]*self.n_contraction+[self.bond_dim])
		
		#create weights tensors
		for i in range(self.n_weights):
			#append to the list of nodes the weight tensors
			self.nodes.append( 
				self.add_weight(
					name        = 'contraction'+str(i)    , #name of the tensor
					shape       = w_shape                 , #shape (computed before)
					trainable   = True                    , #set weights to be trainable
					initializer = self.kernel_initializer , #initialize weights
					regularizer = self.kernel_regularizer   #set regularization function
				)
			)

		#if specified add bias tensor
		self.bias_var = self.add_weight(
							name        = 'bias'                          , #bias name
							shape       = (self.n_weights, self.bond_dim) , #dimension should be equal to final output dimension
							trainable   = True                            ,#set bias to be trainable
							initializer = self.bias_initializer            #initialization
						) if self.use_bias else None  #if use_bias is false the bias vector is not instantiated

	def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:  
		"""
		Executes all the contractions of the layer
		The contraction over a single input sample is done in the contract function
		and is then vectorized over the whole input dataset
		"""

		def contract(
			x          : tf.Tensor  ,  # input sample 
			nodes      : List[Node] ,  # list of weights
			n_contr    : int        ,  # number of feature to contract on each weight
			use_bias   : bool       ,  # use bias (true or false)
			bias_var   : tf.Tensor	   # bias tensor
			) -> tf.Tensor :

			x_nodes  = []
			tn_nodes = []

			#create tensornetwork nodes using a for loop on nodes and one on features
			#to the i-th node the associated features are [i*n_contr;(i+1)*n_contr[ 
			for i in range(len(nodes)):  # loop over the weight tensors
				for j in range(n_contr): # loop over input features corresponding to the weight
					#create feature nodes
					x_nodes.append(
						tn.Node(x[n_contr*i+j]          , #feature to convert to node
								name      = 'xnode'     , 
								backend   = "tensorflow"  #use tensoflwo to manage computations
						)
					)
				#create weight nodes
				tn_nodes.append(
					tn.Node(
						nodes[i]            , #weight to convert to node
						name=f'node_{i}'    , 
						backend="tensorflow"  #use tensoflwo to manage computations
					)
				)
			
			#using the same loop structure connect the edges of the nodes
			#this DOES NOT contract but only prepares for the contraction
			for i in range(len(nodes)):  # loop over the weight tensors
				for j in range(n_contr): # loop over input features corresponding to the weight
					#make connections between weight and corresponing feature
					x_nodes[n_contr*i+j][0] ^ tn_nodes[i][j]
					
			# Contract each weight tensor to its feature
			# using tensornetwork contractor
			result = []
			for i in range(len(nodes)): #loop over weights
				result.append(
					tn.contractors.greedy( #use Tn contractor
						[x_nodes[n_contr*i+j] for j in range(n_contr)]+[tn_nodes[i]] #weight node and list of connected features
					)
				)

			#revert the result to a Tensoflow tensor and add (if specified) the bias tensor
			result= tf.convert_to_tensor([r.tensor for r in result])
			if use_bias:
				result += bias_var
			return result

		#prepare input data for the vectorization of the contract function
		input_shape = list(inputs.shape) #get shape
		inputs = tf.reshape(inputs, (-1, input_shape[1], input_shape[2])) #expand dimension
		#vectorize the contraction over all the input samples
		result = tf.vectorized_map( #vectorize
				lambda vec: contract( #create a lambda function to be vectorized
					vec                , #input sample
					self.nodes         , #weight tensors
					self.n_contraction , #number of feaqture to contract
					self.use_bias      , 
					self.bias_var		#bias tensor
				), 
			inputs						#input dataset over which vectorize the lambda function
		)

		#if specified use the activation function over the output
		if self.activation is not None:
			result = self.activation(result)
		return result

	def get_config(self) -> dict:
		"""
		Returns a python dictionary containing the configuration of the layer.
		The same layer can be reinstantiated later
		(without its trained weights) from this configuration.
		"""
		config = {}

		# Include the TTN layer specific argument
		args = ['n_contraction', 'bond_dim', 'use_bias']
		#get arguments and add them to config dictionary
		for arg in args:
			config[arg] = getattr(self, arg)
		# get the activation function
		config['activation'] = activations.serialize(getattr(self, 'activation'))

		# get the kernel(weights) and bias initializers
		custom_initializers = ['kernel_initializer', 'bias_initializer']
		for initializer_arg in custom_initializers:
			config[initializer_arg] = initializers.serialize(
					getattr(self, initializer_arg))

		# Get parent config
		base_config = super().get_config()
		#return base and TTN layer config parameters
		return dict(list(base_config.items()) + list(config.items()))

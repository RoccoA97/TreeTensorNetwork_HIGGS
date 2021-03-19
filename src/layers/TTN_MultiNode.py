import tensorflow as tf
from tensorflow.keras.layers import Layer  # type: ignore
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from typing import List, Optional, Text, Tuple
import tensornetwork as tn
from tensornetwork.network_components import Node
import numpy as np
import math

# Create keras layer using the dedicated wrapper

@tf.keras.utils.register_keras_serializable(package='tensornetwork')
class TN_layer_MultiNode(Layer):
  #layer initialization
  def __init__(self,
               bond_dim: int,
               out_dim  :int,
               use_bias: Optional[bool] = True,
               activation: Optional[Text] = None,
               kernel_initializer: Optional[Text] = 'glorot_uniform',
               bias_initializer: Optional[Text] = 'zeros',
               kernel_regularizer = None,
               **kwargs) -> None:

    # Allow specification of input_dim instead of input_shape,
    # for compatability with Keras layers that support this
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super().__init__(**kwargs) #initialize parent class
    self.bond_dim = bond_dim
    self.out_dim = out_dim
    self.nodes = []
    self.use_bias = use_bias
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)

  #layer building
  def build(self, input_shape: List[int]) -> None:
    if input_shape[-1] is None:
      raise ValueError('The last dimension of the inputs to should be defined. Found `None`.')

    input_dim = int(input_shape[-1])


    self.n_features = input_shape[1]
    assert self.n_features%2 == 0, "Must have an even number of features to contract"
    self.n_sites = self.n_features//2
    self.n_weights = 3*self.n_features//2

    super().build(input_shape)

    self.feature_dim  = input_shape[2]

    #create a weight node for each feature in each site plus an output node
    #each input will be contracted to a tensor and then 
    #each of these will be contracted to the respective  output tensor
    for i in range(self.n_sites):
      self.nodes.append(
          self.add_weight(name='contr_in_left'+str(i),
                          shape=(self.feature_dim, self.bond_dim,
                                 self.bond_dim),
                          trainable=True,
                          initializer=self.kernel_initializer,
                          regularizer=self.kernel_regularizer))
      self.nodes.append(
          self.add_weight(name='contr_in_right'+str(i),
                          shape=(self.feature_dim, self.bond_dim,
                                 self.bond_dim),
                          trainable=True,
                          initializer=self.kernel_initializer,
                          regularizer=self.kernel_regularizer))
      self.nodes.append(
          self.add_weight(name='contr_out'+str(i),
                          shape=(self.bond_dim, self.bond_dim,
                                 self.out_dim),
                          trainable=True,
                          initializer=self.kernel_initializer,
                          regularizer=self.kernel_regularizer))    
      
    #initialize bias tensor is required
    self.bias_var = self.add_weight(
        name='bias',
        shape=(self.n_sites, self.out_dim),
        trainable=True,
        initializer=self.bias_initializer) if self.use_bias else None

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:  # pylint: disable=unused-argument
    #function for the contraction, will be vectorized over the input files
    def f(x: tf.Tensor, nodes: List[Node], #num_nodes: int,
           in_leg_dim: int,
           use_bias: bool, bias_var: tf.Tensor) -> tf.Tensor:

      x_nodes = []
      tn_nodes = []
      #create nodes and connect them
      for i in range(self.n_sites):
        x_nodes .append(tn.Node(x[2*i]   , name='xnode'    , backend="tensorflow"))
        x_nodes .append(tn.Node(x[2*i+1] , name='xnode'    , backend="tensorflow"))
        tn_nodes.append(tn.Node(nodes[3*i  ] , name=f'node_{i}', backend="tensorflow"))
        tn_nodes.append(tn.Node(nodes[3*i+1] , name=f'node_{i}', backend="tensorflow"))
        tn_nodes.append(tn.Node(nodes[3*i+2] , name=f'node_{i}', backend="tensorflow"))
        
        
        x_nodes[2*i   ][0]  ^ tn_nodes[3*i  ][0]
        x_nodes[2*i+1 ][0]  ^ tn_nodes[3*i+1][0]
        tn_nodes[3*i  ][1]  ^ tn_nodes[3*i+1][1] 
        tn_nodes[3*i  ][2]  ^ tn_nodes[3*i+2][0] 
        tn_nodes[3*i+1][2]  ^ tn_nodes[3*i+2][1] 

        #creates the structure
        #     I1     I2   -> input features
        #     |      |
        #     W1 --- W2   -> intermediate weight tensors
        #       \    /
        #         O       -> output tensor

      result = []
      #contract all the nodes
      for i in range(self.n_sites):
        result.append((x_nodes [2*i   ] @ tn_nodes[3*i  ]) @
                      (x_nodes [2*i+1 ] @ tn_nodes[3*i+1]) @
                       tn_nodes[3*i+2])
      #revert to tensorflow tensor
      result= tf.convert_to_tensor([r.tensor for r in result])
      if use_bias:
        result += bias_var

      return result

    input_shape = list(inputs.shape)
    inputs = tf.reshape(inputs, (-1, input_shape[1], input_shape[2]))
    #vectorize contraction over input dataset
    result = tf.vectorized_map(
        lambda vec: f(vec, self.nodes, self.n_features,
                   self.use_bias, self.bias_var), inputs)
    if self.activation is not None:
      result = self.activation(result)


    return result


  def get_config(self) -> dict:
    """Returns a python dictionary containing the configuration of the layer.
    The same layer can be reinstantiated later
    (without its trained weights) from this configuration.
    """
    config = {}

    args = ['out_dim', 'bond_dim', 'use_bias']
    for arg in args:
      config[arg] = getattr(self, arg)

    config['activation'] = activations.serialize(getattr(self, 'activation'))

    custom_initializers = ['kernel_initializer', 'bias_initializer']
    for initializer_arg in custom_initializers:
      config[initializer_arg] = initializers.serialize(
          getattr(self, initializer_arg))

    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

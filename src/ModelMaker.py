import os
from re import M
import tensorflow        as tf
import tensornetwork     as tn
import math

from tensorflow.keras.layers      import Activation
from tensorflow.keras.layers      import BatchNormalization
from tensorflow.keras.models      import Sequential
from tensorflow.keras             import regularizers

from layers.TTN_SingleNode import TTN_SingleNode

def Make_SingleNode_Model(
        input_shape,
        n_contr        = 2     , #number of feature to contract in each site
        activation     = None  , #activcation function tu use
        use_batch_norm = False , #use (or not) batch normalization
        bond_dim       = 10    , #boind dimension of the layers
        verbose        = False , #verbosity, print the inital configuration
        use_reg        = True    #use (or not) kernel regularization
    ):


    KER_REG_L2 = 1.0e-4  #layer parameter regualrization
    n_layers   = int(math.log(input_shape[0], n_contr)) #number of layers to create
    
    if verbose:
        print("input_shape", input_shape)
        print("n_contr"    , n_contr)
        print("n_layers"   , n_layers)
    tn_model = Sequential()

    #if the batch normalization is required the architecture has to be created step by step
    if use_batch_norm:
        #first layer, input shape is required and parameter
        tn_model.add(
            TTN_SingleNode(
                bond_dim           = bond_dim    , #bond dimension of the weights
                n_contraction      = n_contr     , #number of feature to contract to each weight tensor
                input_shape        = input_shape , 
                kernel_regularizer = regularizers.l2(KER_REG_L2) if use_reg else None  #use regularization if specified
            )
        )
        #add batch normalization after the layer but before the activation
        tn_model.add(
            BatchNormalization(
                epsilon  = 1e-06 ,
                momentum = 0.9   ,
                weights  = None
            )
        )
        #if required use activation function
        if activation is not None:
            tn_model.add(Activation(activation))

        #intermediate layers, same as before but withoput specifing input shape
        for _ in range(n_layers-2):
            tn_model.add(
                TTN_SingleNode(
                    bond_dim           = bond_dim ,
                    n_contraction      = n_contr  ,
                    kernel_regularizer = regularizers.l2(KER_REG_L2) if use_reg else None
                )
            )
            tn_model.add(
                BatchNormalization(
                    epsilon  = 1e-06 ,
                    momentum = 0.9   ,
                    weights  = None
                )
            )
            if activation is not None:
                tn_model.add(Activation(activation))

        #last layer with bond dimension 1
        tn_model.add(
            TTN_SingleNode(
                bond_dim           = 1           , #bond dimension must be one to obtain a single value
                use_bias           = True        ,
                n_contraction      = n_contr     ,
                input_shape        = input_shape ,
                kernel_regularizer = regularizers.l2(KER_REG_L2) if use_reg else None
            )
        )
        tn_model.add(
            BatchNormalization(
                epsilon  = 1e-06 ,
                momentum = 0.9   ,
                weights  = None
            )
        )
        #activation must be used to interpret results as a probability
        tn_model.add(Activation('sigmoid'))

    #without batch normalization the activation can be directly included inside the model creation
    else:
        #first layer
        tn_model.add(
            TTN_SingleNode(
                bond_dim           = bond_dim     , #bond dimension
                activation         = activation   , #activation function
                input_shape        = input_shape  , #input shape
                n_contraction      = n_contr      , #contraction per site
                kernel_regularizer = regularizers.l2(KER_REG_L2) if use_reg else None #regularization
            )
        )
        #intermediate layers, same as previous layer but without input shape required
        for _ in range(n_layers-2):
            tn_model.add(
                TTN_SingleNode( 
                    bond_dim=bond_dim     , 
                    activation=activation ,
                    n_contraction=n_contr , 
                    kernel_regularizer=regularizers.l2(KER_REG_L2) if use_reg else None 
                )
            )
        #last layer, activation imposed to sigmoid to interpret results as probabilities
        tn_model.add(
            TTN_SingleNode( 
                bond_dim=1            , #bond dim = 1 to get a single value
                activation='sigmoid'  , #sigmoid activation to obtain a probability
                n_contraction=n_contr , 
                kernel_regularizer=regularizers.l2(KER_REG_L2) if use_reg else None
            )
        )
    return tn_model



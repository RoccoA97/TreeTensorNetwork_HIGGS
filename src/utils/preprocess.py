import math
import json
import argparse
import tensorflow        as tf
import tensornetwork     as tn
import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import datetime
from . import utils
from  sklearn.preprocessing import StandardScaler


"""
Executes the preprocessing of a given dataset.
Given the feature map and its order and the number of contraction 
to make in each site the dataset is padded and the map is applied.
The dataset is then divided in Train test and validation sets
and returned as such
"""

def Preprocess(
        data                            , 
        N_train         =   4000000     , #number of samples in the training set
        N_val           =    500000     , #number of samples in the validation set
        N_test          =    500000     , #number of samples in the testset
        feature_map     =   'polynomial', #feature map
        map_order       =   2           , #map order
        con_order       =   2           , #number of contraction in each site
        verbose         =   False         #verbosity of the output (for debug)
    ):
    

    #check for correct dimensions
    assert data.shape[0] >= N_train+N_test+N_val, "Request division is bigger than provided data"
    
    allowed_maps = {'polynomial': utils.PolynomialMap,
                    'spherical' : utils.SphericalMap }

    #check if the requested map is one of the allowed
    if feature_map is not None:
        assert feature_map in list(allowed_maps.keys()), \
                        "Requested map does not exist, allowed are:"\
                        +str(list(allowed_maps.keys()))
        assert map_order > 1, "Map order must be at least 2"
    assert con_order >1 , "Contraction order must be at least 2"

    #divide x and y data and stanrdardize X dataset
    #the standardization is executed using only the training sample
    x_data = utils.Standardize(data.iloc[:,1:].to_numpy(), N_train)
    y_data =                   data.iloc[:,0 ].to_numpy()

    if verbose: print("Data shape")
    if verbose: print("x_data shape: ",x_data.shape, "y_data shape: ",y_data.shape)
  
    #pad dataset depending on the contraction order
    x_data = utils.PadToOrder(x_data, con_order)

    if verbose: print("Padded data shape")
    if verbose: print("x_data shape: ",x_data.shape, "y_data shape: ",y_data.shape)

    #if requested apply the feature map
    if feature_map is not None:
        x_data = allowed_maps[feature_map](x_data, order=map_order)

    if verbose: print("Mapped data shape")
    if verbose: print("x_data shape: ",x_data.shape, "y_data shape: ",y_data.shape)

    #split the dataset in train test and and validation
    x_train = x_data[               : N_train              ] 
    x_val   = x_data[ N_train       :(N_train+N_val       )]
    x_test  = x_data[(N_train+N_val):(N_train+N_val+N_test)]

    y_train = y_data[               : N_train              ]     
    y_val   = y_data[ N_train       :(N_train+N_val       )]
    y_test  = y_data[(N_train+N_val):(N_train+N_val+N_test)]

    if verbose: print("Train, validation, test data shape")
    if verbose: print("x_train shape: ", x_train.shape, "y_train shape: ",y_train.shape)
    if verbose: print("x_val   shape: ", x_val  .shape, "y_val   shape: ",y_val  .shape)
    if verbose: print("x_test  shape: ", x_test .shape, "y_test  shape: ",y_test .shape)
    
    #if feature map is of order 1 the dimension should be expanded in order
    #to make the shape compatible with the tensorial structre of the TTN layer
    if len(x_train.shape)==2:
        x_train = np.reshape(x_train , (x_train.shape[0], x_train.shape[1], 1))
        x_val   = np.reshape(x_val   , (x_val  .shape[0], x_val  .shape[1], 1))            
        x_test  = np.reshape(x_test  , (x_test .shape[0], x_test .shape[1], 1))
    
    #return the sets
    return x_train, x_val, x_test, y_train, y_val, y_test 


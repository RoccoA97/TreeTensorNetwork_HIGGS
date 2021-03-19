import utils.preprocess as preprocess
import os
import pandas as pd
import os
import math
import tensorflow        as tf
import tensornetwork     as tn
import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import datetime
import json

from tensorflow.keras.models      import load_model
from sklearn.preprocessing        import MinMaxScaler
from layers.TTN_SingleNode        import TTN_SingleNode


#load the dataset from the specified data path
DATA_PATH = "../data/"
N = 11000000

if os.path.isfile(DATA_PATH + "HIGGS.csv.gz"):
    data = pd.read_csv(
                DATA_PATH         + 'HIGGS.csv.gz'       ,  
                compression='gzip', error_bad_lines=False, 
                nrows=N           , header=None          
            )
elif os.path.isfile(DATA_PATH + "HIGGS.csv"):
    data = pd.read_csv(
                DATA_PATH + 'HIGGS.csv',    nrows=N  ,
                error_bad_lines=False  , header=None 
            )
else:
    print("Error: Data file not found")

#preprocess the dataset using an order 2 spherical map
#and preprare for a 2 contraction structure typology
x_train, x_val, x_test, y_train, y_val, y_test = \
    preprocess.Preprocess(
        data                       , 
        feature_map  = 'spherical' , #map typology
        map_order    = 2           , #map order
        con_order    = 2           , #number of contraction per site
        verbose      = True        , #verbose (print shapes for debugging)
        N_train      = 10000000    , #train set size
        N_val        = 500000      , #validation set size
        N_test       = 500000        #test set size
    )


#load the model from a specific directory and evaluate their performance
mdir        = "../models"       #model directory  
model_names = os.listdir(mdir)  #get all model names

evals = {}
for mn in model_names:
    #parse model name and load it
    par = mn.split('__')
    model = load_model(os.path.join(mdir, mn))
    
    #evaluate model performance
    ev = model.evaluate(x_val,y_val)
    evals[par[4]]= ev #save the bond dimension as key

outdir = '../JsonFiles' #output directory
#save results
with open(os.path.join(outdir, "evaluation.json"), 'w') as fp:
        json.dump(evals, fp)
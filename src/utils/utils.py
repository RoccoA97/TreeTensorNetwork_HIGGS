import numpy             as np
import scipy
import math

#standardize the dataset using ONLY the training samples
def Standardize(x, nt):
    """
    Standardize each feature diving by the absolute maximum, 
    distributions will be in [-1,1] for features with negative values
    or in [0,1] for features with only positive ones.
    """
    for j in range(x.shape[1]): #loop over features
        vec      = x  [:, j] #get feature vector
        vec_norm = vec[:nt]  #take only training part
        vec      = vec / np.max(np.abs(vec_norm)) #normalize using max of training
        x[:,j]   = vec #overwrite original feature with normalized one 
    return x

def scale(X, scaler):
    #Scale each feature of 3D array using a given scaler
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])     
    return X

def flatten(X):
    #Flatten a 3D array.
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :] #get flattened feature
    return(flattened_X)


# apply spherical map to input data
#loop is executed over map order not over samples 
#this allows to process the dataset faster using numpy vectorization
def SphericalMap(x, order=2, dtype=np.float32):
    x_map = np.zeros((x.shape[0],x.shape[1],order), dtype=dtype)
    for i in range(order): #loop over map order
        comb_coef    = np.sqrt(scipy.special.comb(order-1,i)) #term coefficient
        #get i-th order term for all the samples in the dataset
        x_map[:,:,i] = comb_coef * np.power(np.cos(x),order-1-i) * np.power(np.sin(x),i)
    return x_map


# apply polynomial map to input data
#loop is executed over map order not over samples 
#this allows to process the dataset faster using numpy vectorization
def PolynomialMap(x, order=2, dtype=np.float32):
    x_map = np.zeros((x.shape[0],x.shape[1],order+1), dtype=dtype)
    for i in range(order+1): #loop over order, 
        x_map[:,:,i] = np.power(x,i) #get i-th order term for all the samples in the dataset
    return x_map

#given a number of contraction pad the array in irder to make 
#the number of feature divisible by the number of contractions
def PadToOrder(x, con_order):
    #estimate the needed number of padding columns
    n_pad = int(con_order**(math.ceil(math.log(x.shape[1], con_order)))-x.shape[1])
    #append n_pad columns to the dataset
    x     = np.append(x, np.ones((x.shape[0], n_pad)), axis=1) 
    return x

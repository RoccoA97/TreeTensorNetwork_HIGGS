# TTN Models

This folder contained the trained architecture of some of the models we presented in the project report. Not all the models are included due to their high dimension.

The architecture of the model used for the final performance evaluation is stored in the file:

``` FinalModel__nc_2__map_spherical_5__bd_50__bs_10000__ep_500__opt_adam__act_elu__model.hdf5 ```

The model naming is done with the following convention:
- each double underscore separates two model hyperparameters;
- each underscore separates an hyperparameter from its value(s);
- the date at the start of the name represents the date when the training ended.

The hyperparameters name represents:
- **`etr`** number of samples in the training set;
- **`nc`** number of features contracted in each site;
- **`map`** type and order of the feature map applied in the preprocessing;
- **`db`** bond dimension of the weight tensors;
- **`bs`** batch size used during the training;
- **`ep`** number of training epochs;
- **`opt`** optimizer used in the training;
- **`act`** activation function used.

# Model Training and Tuning results

In this folder the results of some of the training used to tune and characterize the various TTN architectures are contained.



## Characterization
Here we saved the results for the following measures:
- "pure" TTN structure vs optimized TTN:
    - **`pure_model_history.json`** standard TTN training history;
    - **`opti_model_history.json`** optimized TTN training history;
- performance dependence on bond dimension
    - **`bond_dim_behav.json`** time scaling and number of parameters for different bond dimension;
    - **`bond_dim_eval.json`** performance of the validation set after the training for different bond dimensions;
- time scaling with different batch sizes:
    - **`batch_size_times.json`**;
- model performance with different feature maps in the preprocessing:
    - **`map_pol_perf.json`** performance with polynomial maps;
    - **`map_poli_times.json`** time scaling with polynomial maps;
    - **`map_sph_perf.json`** performances with spherical maps;
    - **`map_spher_times.json`** time scaling with spherical maps.



## History

In this folder the training history of some of the trained models is contained.
The naming is done with the following convention:
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

# Tree Tensor Netowrk Implementation

This folders contains all the code needed to implement a TTN classifier using the TensorFlow and TensorNetwork framoworks and the Keras API.



## Implementation

The code for the implementation is contained in the following folders:
- **`layers`** contains custom TensorFlow layers that can be used to create a full model, in particular:
    - **`TTN_SingleNode.py`** has a layer structure were `N_contraction` inputs features are contracted to the same weight tensor;
    - **`TTN_MultiNode.py`** has a layer structure were each input is connected to a weight tensor and then couples of input and weight tensors are contracted to an output tensor;
        - !! This layer works and the model is able to make good prediction but it was not fully tested, so some unwanted errors may occur;
- **`utils`** contains the functions used to preprocess the dataset, in particular the `preprocess` class can be used to perform all the preprocessing steps with specified parameters;
- the **`Modelmaker.py`** file contains the function needed to create a TTN classifier depending on a series of parameters, including many optimizations like the kernel regularization and the batch normalization.



## Model Training

In the **`TTN_train_example.ipynb`** a full example of the TTN implementation training and performance evaluation is shown. Some python and bash scripts are also provided in order to train a specific model architecture.

In the **`scripts`** folder two script for training model with different complexity and differently preprocessed data are contained.

The **`run_train_telegrad.py`** script implements a parser which allows to select all the parameters for the dataset preprocessing, model creation and model training. A full list of all the paramenters with their description can be obtained running:

```
python run_train_telegrad.py -h
```

Inside the script the Telegrad callback, whose implementation is contained in the **`telegrad`** folder, can be used. The callback can be connected to a Telegram bot (you will need to create your own) allowing to receive the training results "live" on Telegram.



## Model Analysis

We provide the code used to evaluate the performances of different types of model architectures.
The **`Characterization.ipynb`** contains the code to the time scaling and some of the performance analyses showed in the final report. Due to the intensive computational cost, some of the analyses were performed using the `evaluate.py` script, which evaluates the performances on a validation set of all the models (already trained) in a given directory.

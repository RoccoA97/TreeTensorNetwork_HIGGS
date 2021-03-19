################################################################################
# IMPORT PACKAGES
################################################################################
import os
import math
import json
import argparse
import tensorflow        as tf
import tensornetwork     as tn
import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import datetime

# from sklearn
from sklearn.preprocessing        import MinMaxScaler
from sklearn.metrics              import roc_curve
from sklearn.metrics              import roc_auc_score
from sklearn.metrics              import roc_auc_score

# from tf-notification-callback
from telegrad.telegram_callback import TelegramCallback

# from user defined modules
import utils.preprocess as preprocess
from layers.TTN_SingleNode import TTN_SingleNode
from ModelMaker            import Make_SingleNode_Model

# number of cores for tensorflow
# tf.config.threading.set_intra_op_parallelism_threads(6)
################################################################################





################################################################################
# COMMAND-LINE PARSER
################################################################################
# parser
parser = argparse.ArgumentParser(description="Train model on HIGGS dataset")


# standard arguments
### dataset preprocessing
#parser.add_argument('-ev' ,  "--events"     , type=int,   dest="events",      default=2000000, help="Number of event samples")
parser.add_argument('-etr',  "--ev_train"   , type=int,   dest="ev_train",    default=1000000, help="Number of training event samples")
parser.add_argument('-eva',  "--ev_val"     , type=int,   dest="ev_val" ,     default= 500000, help="Number of validation event samples")
parser.add_argument('-ete',  "--ev_test"    , type=int,   dest="ev_test",     default= 500000, help="Number of test event samples")
#parser.add_argument('-ft' ,  "--train_frac" , type=float, dest="train_frac",  default=0.9,     help="Fraction of dataset to use as training dataset")
parser.add_argument('-map',  "--feature_map", type=str,   dest="feature_map", default=None,    help="Map applied to the dataset during preprocessing")
parser.add_argument('-mo' ,  "--map_order"  , type=int,   dest="map_order",   default=2,       help="Order of applied map during preprocessing")
parser.add_argument('-val',  "--validation" ,             dest="validation",  action="store_true", default=False, help="Use validation dataset")
### network architecture
parser.add_argument('-nc',  "--n_contr",     type=int,   dest="n_contr",    default=2,       help="Number of features to contract at each node")
parser.add_argument('-bd',  "--bond_dim",    type=int,   dest="bond_dim",   default=15,      help="Bond dimension for TN layers")
parser.add_argument('-act', "--activation",  type=str,   dest="activation", default="relu",  help="Activation function for training")
parser.add_argument('-bn',  "--batch_norm",              dest="batch_norm", action="store_true", default=False, help="Use Batch Normalization")
### network training hyperparameters
parser.add_argument('-bs',  "--batch_size",  type=int,   dest="batch_size", default=10000,   help="Batch size for training")
parser.add_argument('-ep',  "--epochs",      type=int,   dest="epochs",     default=200,     help="Number of epochs of training")
parser.add_argument('-opt', "--optimizer",   type=str,   dest="optimizer",  default="adam",  help="Optimizer for training")
### verbosity
parser.add_argument('-v',   "--verbose",     type=int,   dest="verbose",    default=2,       help="Verbosity (0=silent, 1=prog bar, 2=one line per epoch)")
### telegrad callback
parser.add_argument('-tc',  "--telegram",                dest="telegram",   action="store_true", default=False, help="Enable Telegram callback")
parser.add_argument('-p',   "--patience",    type=int,   dest="patience",   default=1,       help="Patience of Telegram callback")


# instantiate parser
args = parser.parse_args()


# parser parameters
### dataset preprocessing
EVENTS       = args.ev_train + args.ev_val + args.ev_test
EVENTS_TRAIN = args.ev_train
EVENTS_VAL   = args.ev_val
EVENTS_TEST  = args.ev_test
#TRAIN_FRAC   = args.train_frac
FEATURE_MAP  = args.feature_map
MAP_ORDER    = args.map_order
VALIDATION   = args.validation
### network architecture
N_CONTR      = args.n_contr
BOND_DIM     = args.bond_dim
ACTIVATION   = args.activation
BATCH_NORM   = args.batch_norm
### network training hyperparameters
BATCH_SIZE   = args.batch_size
EPOCHS       = args.epochs
OPTIMIZER    = args.optimizer
### verbosity
VERBOSE      = args.verbose
### telegrad callback
TELEGRAM     = args.telegram
PATIENCE     = args.patience


# logdir
LOGDIR = "../logdir/"


# other parameters
KER_REG_L2 = 1.0e-4
ACT_REG_L2 = 1.0e-4
ACT_REG_L1 = 1.0e-4
################################################################################





################################################################################
# LOAD DATA
################################################################################
# data path
DATA_PATH = "../data/"


# load data
if os.path.isfile(DATA_PATH + "HIGGS.csv.gz"):
    data = pd.read_csv(DATA_PATH + "HIGGS.csv.gz", nrows=EVENTS, compression='gzip', error_bad_lines=False, header=None)
elif os.path.isfile(DATA_PATH + "HIGGS.csv"):
    data = pd.read_csv(DATA_PATH + "HIGGS.csv",    nrows=EVENTS,                     error_bad_lines=False, header=None)
################################################################################





################################################################################
# PREPROCESS DATA
################################################################################
x_train, x_val, x_test, y_train, y_val, y_test = preprocess.Preprocess(
    data,
    feature_map   = FEATURE_MAP,
    map_order     = MAP_ORDER,
    con_order     = N_CONTR,
    verbose       = VERBOSE,
    N_train       = EVENTS_TRAIN,
    N_val         = EVENTS_VAL if VALIDATION else 0,
    N_test        = EVENTS_TEST
)
################################################################################





################################################################################
# BUILD AND COMPILE MODEL
################################################################################
# build model
tn_model = Make_SingleNode_Model(
    input_shape    = (x_train.shape[1:]),
    n_contr        = N_CONTR,
    bond_dim       = BOND_DIM,
    activation     = ACTIVATION,
    use_batch_norm = BATCH_NORM
)


# model summary
if VERBOSE>0: tn_model.summary()


# delaration of auc metric
auc_metric = tf.keras.metrics.AUC(
    num_thresholds   = 200,
    curve            = 'ROC',
    summation_method = 'interpolation',
    name             = None,
    dtype            = None,
    thresholds       = None,
    multi_label      = False,
    label_weights    = None
)


# model compilation
tn_model.compile(
    optimizer = OPTIMIZER,
    loss      = 'binary_crossentropy',
    metrics   = ['accuracy', auc_metric]
)
################################################################################





################################################################################
# CALLBACKS
################################################################################
datetime  =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

model_id  =  datetime                                         + "__"
model_id += "etr_" + str(EVENTS_TRAIN)                        + "__"
model_id += "nc_"  + str(N_CONTR)                             + "__"
model_id += "map_" + FEATURE_MAP + "_" + str(MAP_ORDER)       + "__"
model_id += "bd_"  + str(BOND_DIM)                            + "__"
model_id += "bs_"  + str(BATCH_SIZE)                          + "__"
model_id += "ep_"  + str(EPOCHS)                              + "__"
model_id += "opt_" + OPTIMIZER                                + "__"
model_id += "act_" + ACTIVATION                               + "__"
################################################################################





################################################################################
# CALLBACKS
################################################################################
# tensorboard callback
tfboard_dir  = LOGDIR
tfboard_dir += "tfboard/"
tfboard_dir += datetime

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir        = tfboard_dir,
    histogram_freq = 1
)


# checkpoint callback
checkpoint_path  =  LOGDIR
checkpoint_path += "models/"
checkpoint_path +=  model_id
checkpoint_path += "model.hdf5"
checkpoint_dir   = os.path.dirname(checkpoint_path)

if not os.path.isdir(checkpoint_dir):
    os.system("mkdir " + checkpoint_dir)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath       = checkpoint_path,
    save_best_only = True,
    monitor        = "val_auc",
    mode           = "max",
    verbose        = 0
)


# telegram callback
if TELEGRAM:
    telegram_callback = TelegramCallback(
        bot_token    = "1667956254:AAGCLtj06l2_s85qtsX3Gpf34tEC3MN_AC4",
        chat_id      = "-539175758",
        modelName    = "TTN model",
        loss_metrics = ['loss', 'val_loss'],
        acc_metrics  = ['accuracy', 'val_accuracy', "auc", "val_auc"],
        log_metrics  = ['val_accuracy', "val_auc"],
        patience     = PATIENCE,
        getSummary   = False
    )


# time callback
class TimeCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times     = []
        self.epochs    = []
        self.timetaken = tf.timestamp()
    def on_epoch_end(self,epoch,logs = {}):
        time_epoch = tf.timestamp()
        self.times.append(time_epoch - self.timetaken)
        self.timetaken = time_epoch
        self.epochs.append(epoch)
    def on_train_end(self,logs = {}):
        self.times = [t.numpy() for t in self.times]

time_callback = TimeCallback()


# callback list
if TELEGRAM:
    callbacks = [tensorboard_callback, checkpoint_callback, telegram_callback, time_callback]
else:
    callbacks = [tensorboard_callback, checkpoint_callback, time_callback]
################################################################################





################################################################################
# TRAIN MODEL
################################################################################
# fit model
history = tn_model.fit(
    x_train,
    y_train,
    validation_data = (x_test,y_test),
    epochs          = EPOCHS,
    batch_size      = BATCH_SIZE,
    callbacks       = callbacks,
    verbose         = VERBOSE
)


# add time to history
history.history['times'] = time_callback.times
################################################################################





################################################################################
# PREDICTIONS FROM TRAINED MODEL
################################################################################
# compute validation accuracy and AUC
val_acc = max(history.history["val_accuracy"])
val_auc = max(history.history["val_auc"])


# print results
if VERBOSE>0: print("Validation acc:", val_acc)
if VERBOSE>0: print("Validation AUC:", val_auc)
################################################################################





################################################################################
# SAVE MODEL AND HISTORY
################################################################################
history_path  =  LOGDIR
history_path += "history/"
history_path +=  model_id
history_path += "history.json"
history_dir   = os.path.dirname(history_path)

if not os.path.isdir(history_dir):
    os.system("mkdir " + history_dir)

with open(history_path, 'w') as file:
    json.dump(history.history, file)
################################################################################

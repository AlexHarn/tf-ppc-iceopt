import tensorflow as tf
import numpy as np

RANDOM_SEED = False  # seed or False
# -------------------------------- TensorFlow ---------------------------------
TF_FLOAT_PRECISION = tf.float32
TF_CPU_ONLY = False
TF_HITLIST_LEN = 700000

# ----------------------------- Ice Mocel Config ------------------------------
ICE_MODEL_PATH = '/home/aharnisch/modded-PPC/real/ice/'

# ------------------------------- Flasher Data --------------------------------
# DATA_PATH = '/net/big-tank/POOL/users/aharnisch/flasher_data_charge_only/'
DATA_PATH = '/net/big-tank/POOL/users/aharnisch/fake_flasher_data/'

# ----------------------------- Simulation Data -------------------------------
# The following sets the path to the modified PPC executable, which simulates
# flasher runs without absorption and logs the traveled distance for each
# photon in each layer. It is assumed to be in an ice/ folder with
# configuration files.
PATH_NO_ABS_PPC = '/home/aharnisch/modded-PPC/no_abs/ice/ppc'

# The following sets the path to the unmodified PPC executable, which is used
# to simulate fake data to fit to. It is assumed to be in an ice/ folder, which
# contains the necessary configuration files, including the ice parameters
# which we try to recover.
PATH_REAL_PPC = '/home/aharnisch/modded-PPC/real/ice/ppc'

# The number of layers to fit, which needs to be the same as in the respective
# PPC configuration.
N_LAYERS = 171
N_DOMS = 5160

# Flashing string, for now we only flash this one string. Should not make a
# difference when comparing to simulation anyways since there are no model
# errors. String 36 is in the middle of deep core.
FLASHER_STRING = 36

# The simulated photon data directory
PHOTON_PATH = '/net/big-tank/POOL/users/aharnisch/iceopt_photons/'

# --------------------------------- Training ----------------------------------
# depth, scatc, absc, delta_t = np.loadtxt('icemodel.dat', unpack=True)
INITIAL_ABS = [0.008 for i in range(N_LAYERS)]
# INITIAL_ABS = absc[::-1]
MAX_STEPS = 100000000
# The number of hits to rescale to. We rescale to this fixed amount of hits
# every time to make the loss more comparable for different emitter DOMs
RESCALED_HITS = 100000

# -------------------------------- Optimizer ----------------------------------
# The initial learning rate
INITIAL_LEARNING_RATE = 0.001
# True or False to activate/deactivate learning rate decay
LEARNING_DECAY = False
# Decay modes: Linear or Exponential
LEARNING_DECAY_MODE = 'Exponential'
# decrease the INITIAL_LEARNING_RATE every LEARNING_STEPS steps by
# LEARNING_DECR linearly or exponentially
LEARNING_DECR = 0.95
LEARNING_STEPS = 10

# supported optimizers: Adam, GradientDescent
OPTIMIZER = 'Adam'
ADAM_SETTINGS = dict(beta1=0.9, beta2=0.999, epsilon=1e-08)

# --------------------------------- Logging -----------------------------------
WRITE_INTERVAL = 1  # how many steps between each write

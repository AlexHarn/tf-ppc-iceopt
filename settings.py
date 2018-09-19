import tensorflow as tf

RANDOM_SEED = False  # seed or False
# -------------------------------- TensorFlow ---------------------------------
TF_FLOAT_PRECISION = tf.float32
TF_CPU_ONLY = False
TF_HITLIST_LEN = 700000

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

# The data directory
PATH_DATA = '/net/big-tank/POOL/users/aharnisch/iceopt_photons/'

# Choose whether to run new simulations or load from an existing dataset in
# PATH_DATA
RUN_SIMULATIONS = False

# --------------------------------- Training ----------------------------------
INITIAL_ABS = [100 for i in range(N_LAYERS)]
MAX_STEPS = 100000000
N_PHOTONS = int(1.5e7)
PHOTONS_PER_FLASH = N_PHOTONS/60

# -------------------------------- Optimizer ----------------------------------
# The initial learning rate
INITIAL_LEARNING_RATE = 1
# The number of optimizer steps to perform per simulation
OPTIMIZER_STEPS_PER_SIMULATION = 1
# True or False to activate/deactivate learning rate decay
LEARNING_DECAY = False
# Decay modes: Linear or Exponential
LEARNING_DECAY_MODE = 'Exponential'
# decrease the INITIAL_LEARNING_RATE every LEARNING_STEPS steps by
# LEARNING_DECR linearly or exponentially
LEARNING_DECR = 0.95
LEARNING_STEPS = 10

# supported optimizers: Adam, GradientDescent
OPTIMIZER = 'GradientDescent'
ADAM_SETTINGS = dict(beta1=0.9, beta2=0.999, epsilon=1e-08)

# --------------------------------- Logging -----------------------------------
WRITE_INTERVAL = 5  # how many steps between each write

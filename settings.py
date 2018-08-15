import tensorflow as tf


RANDOM_SEED = False  # seed or False
# -------------------------------- TensorFlow ---------------------------------
TF_FLOAT_PRECISION = tf.float32
TF_CPU_ONLY = True

# ----------------------------------- PPC -------------------------------------
# The following sets the path to the modified PPC executable, which simulates
# flasher runs without absorption and logs the traveled distance for each
# photon in each layer. It is assumed to be in an ice/ folder with
# configuration files.
PATH_NO_ABS_PPC = "/home/alex/PPC_absorb_later/ice/ppc"

# The following sets the path to the unmodified PPC executable, which is used
# to simulate fake data to fit to. It is assumed to be in an ice/ folder, which
# contains the necessary configuration files, including the ice parameters
# which we try to recover.
PATH_REAL_PPC = "/home/alex/PPC_real/ice/ppc"

# The number of layers to fit, which needs to be the same as in the respective
# PPC configuration.
N_LAYERS = 171
N_DOMS = 5160

# --------------------------------- Training ----------------------------------
# One step includes one simulation of FLASHES_PER_STEP flasher board flashes
# using PPC. Each

MAX_STEPS = 100000000
BATCHES_PER_STEP = 10
PHOTONS_PER_FLASH = int(10**7)
FLASHES_PER_STEP = 60

# -------------------------------- Optimizer ----------------------------------
# How many times to reuse the same data for optimization
OPTIMIZER_STEPS_PER_SIMULATION = 1
# The initial learning rate
INITIAL_LEARNING_RATE = .1
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

import tensorflow as tf

# --------------------------------- General -----------------------------------
# The random seed to use. Seed or False. Kind of pointless at this point,
# relict from earlier version.
RANDOM_SEED = False

# The number of layers to fit, which needs to be the same as in the respective
# PPC configuration.
N_LAYERS = 171
N_DOMS = 5160

# -------------------------------- TensorFlow ---------------------------------
TF_FLOAT_PRECISION = tf.float32
TF_CPU_ONLY = False
# The length of the simulated photons array, this is the amount of photons we
# optimize on each step. More is better, but it is limited by available GPU
# memory. With the current program 700000 seems to be about the maximum a Tesla
# P40 can handle without running out of memory.
TF_HITLIST_LEN = 700000

# ----------------------------- Ice Mocel Config ------------------------------
# This is the path to the ice model to use to calculate the intrinsic
# absorption coefficient and wavelength dependency of the dust absorption
# coefficient. So the relevevant files are icemodel.dat for the delta
# temperature values and icemodel.par for the 6 parameter ice model parameters.
ICE_MODEL_PATH = '/home/aharnisch/modded-PPC/real/ice/'

# ------------------------------- Flasher Data --------------------------------
DATA_PATH = '/net/big-tank/POOL/users/aharnisch/fake_flasher_data/'

# ----------------------------- Simulation Data -------------------------------
# The simulated photon data directory
PHOTON_PATH = '/net/big-tank/POOL/users/aharnisch/iceopt_photons/'

# --------------------------------- Training ----------------------------------
# Flashing string, for now we only flash this one string. Should not make a
# difference when comparing to simulation anyways since there are no model
# errors. String 36 is in the middle of deep core. String 69 is in the top
# right vorner of the second to last xy layer of strings (minimally effected by
# deep core.)
FLASHER_STRINGS = [36]

# If this flag is set to true, the gradient is averaged over an entire string
# before fed to the optimizer. This is not the same as evaluating string
# batches. The batches are still performed on individual emitter DOMs but
# instead of applying the gradient each time we evaluate it for all DOMs on a
# string and then feed the acuumulated gradient to the optimizer. This might
# make the gradient more stable becaue it includes information on all layers.
# It also smooths out the loss significantly which is helpful when debugging.
# If it is set to False the gradient is applied on every dom batch each time.
GRADIENT_AVERAGING = True

# The initial absorption coefficients to start with.
INITIAL_ABS = [0.01 for i in range(N_LAYERS)]

# The smallest allowed absorption coeffizient, values below are clipped on
# every step
MIN_ABS = 0.001

# The maximum number of training steps to perform.
MAX_STEPS = 200

# The number of hits to rescale to. We rescale to this fixed amount of hits
# every time to make the loss more comparable for different emitter DOMs.  The
# reason we have to rescale at all is the fact that we don't know how many
# photons have been emitted on data.
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

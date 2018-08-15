from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import trange

from logger import Logger
from model import Model
from ppc_wrapper import PPCWrapper
import settings

# ------------------------------ Initialization -------------------------------
# set random seeds
if settings.RANDOM_SEED:
    tf.set_random_seed(settings.RANDOM_SEED)
    np.random.seed(settings.RANDOM_SEED)

# define placeholders
tf_data_hits = tf.placeholder(dtype=settings.TF_FLOAT_PRECISION,
                              shape=(settings.N_DOMS))

tf_simulated_photons = tf.placeholder(dtype=settings.TF_FLOAT_PRECISION,
                                      shape=(None, settings.N_LAYERS + 1))

# initialize the model
model = Model(settings.INITIAL_ABS)

# initialize the PPC wrapper
ppc = PPCWrapper(settings.PATH_NO_ABS_PPC, settings.PATH_REAL_PPC)

# define hitlists
hits_true = tf_data_hits

hits_pred = model.tf_expected_hits(tf_simulated_photons)

# Dimas likelihood, take the logarithm for stability
mu = (hits_pred + hits_true)/2
logLR_doms = hits_pred*(
    tf.log(mu) - tf.log(hits_pred)) + \
    hits_true*(tf.log(mu) - tf.log(hits_true))

loss = -tf.reduce_sum(tf.where(tf.is_nan(logLR_doms),
                               tf.zeros_like(logLR_doms), logLR_doms))

# crate variable for learning rate
tf_learning_rate = tf.Variable(settings.INITIAL_LEARNING_RATE,
                               trainable=False,
                               dtype=settings.TF_FLOAT_PRECISION)

# create update operation for learning rate
if settings.LEARNING_DECAY_MODE == 'Linear':
    update_learning_rate = tf.assign(tf_learning_rate, tf_learning_rate -
                                     settings.LEARNING_DECR)
elif settings.LEARNING_DECAY_MODE == 'Exponential':
    update_learning_rate = tf.assign(tf_learning_rate, tf_learning_rate *
                                     settings.LEARNING_DECR)
else:
    raise ValueError(settings.LEARNING_DECAY_MODE +
                     " is not a supported decay mode!")

# initialize the optimizer
if settings.OPTIMIZER == 'Adam':
    optimizer = tf.train.AdamOptimizer(tf_learning_rate,
                                       **settings.ADAM_SETTINGS)
elif settings.OPTIMIZER == 'GradientDescent':
    optimizer = tf.train.GradientDescentOptimizer(tf_learning_rate)
else:
    raise ValueError(settings.OPTIMIZER+" is not a supported optimizer!")

# create operation to minimize the loss
optimize = optimizer.minimize(loss)

# grab all trainable variables
trainable_variables = tf.trainable_variables()

if __name__ == '__main__':
    if settings.TF_CPU_ONLY:
        config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        # don't allocate the entire vRAM initially
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)

    # initialize all variables
    session.run(tf.global_variables_initializer())
    # --------------------------------- Run -----------------------------------
    # initialize the logger
    logger = Logger(logdir='./log/', overwrite=True)
    logger.register_variables(['loss'] + ['l_abs_pred_{}'.format(i) for i in
                                          range(settings.N_LAYERS)],
                              print_all=True)
    logger.message("Starting...")

    for step in range(1, settings.MAX_STEPS + 1):

        # For now we simply flash all DOMs on string 60 in each step.
        data_hits = ppc.simulate_flash(60, 1, settings.PHOTONS_PER_FLASH)
        simulated_photons = \
            ppc.simulate_flash_no_abs(60, 1, settings.PHOTONS_PER_FLASH)
        for dom in trange(2, 61):
            data_hits += ppc.simulate_flash(60, dom,
                                            settings.PHOTONS_PER_FLASH)
            batch = ppc.simulate_flash_no_abs(60, dom,
                                              settings.PHOTONS_PER_FLASH)
            simulated_photons = np.concatenate((simulated_photons, batch))

        # compute and apply gradients and get the loss with this data
        step_loss = session.run([optimize, loss],
                                feed_dict={tf_data_hits: data_hits,
                                           tf_simulated_photons:
                                           simulated_photons})[1]

        # get updated parameters
        result = session.run(model.l_abs)
        logger.log(step, [step_loss] + result.tolist())

        if step % settings.WRITE_INTERVAL == 0:
            logger.write()

        if settings.LEARNING_DECAY and step % settings.LEARNING_STEPS == 0:
            learning_rate = session.run(update_learning_rate)
            logger.message("Learning rate decreased to {:2.4f}"
                           .format(learning_rate), step)
            if learning_rate <= 0:
                break
    logger.message("Done.")

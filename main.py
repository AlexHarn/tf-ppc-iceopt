from __future__ import division
import tensorflow as tf
import numpy as np

from logger import Logger
from model import Model
from ppc_wrapper import PPCWrapper
from data_handler import DataHandler
import settings

# ------------------------------ Initialization -------------------------------
print("Initializing...")
# set random seeds
if settings.RANDOM_SEED:
    tf.set_random_seed(settings.RANDOM_SEED)
    np.random.seed(settings.RANDOM_SEED)

# define placeholders
tf_data_hits_placeholder = tf.placeholder(dtype=settings.TF_FLOAT_PRECISION,
                                          shape=(settings.N_DOMS))

tf_simulated_photons_placeholder = \
        tf.placeholder(dtype=settings.TF_FLOAT_PRECISION,
                       shape=(settings.TF_HITLIST_LEN, settings.N_LAYERS + 1))

# define data variables
tf_data_hits = tf.get_variable("data_hits", dtype=settings.TF_FLOAT_PRECISION,
                               shape=(settings.N_DOMS), trainable=False)
tf_simulated_photons = tf.get_variable("simulated_photons",
                                       dtype=settings.TF_FLOAT_PRECISION,
                                       shape=(settings.TF_HITLIST_LEN,
                                              settings.N_LAYERS + 1),
                                       trainable=False)

# define operations to initialize data variables
init_data = [tf_data_hits.assign(tf_data_hits_placeholder),
             tf_simulated_photons.assign(tf_simulated_photons_placeholder)]

# initialize the model
model = Model(settings.INITIAL_ABS)

# define operation to reset trained parameters
reset_paras = model.l_abs.assign(settings.INITIAL_ABS)

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

# create operation to optimize
optimize = optimizer.minimize(loss)

if __name__ == '__main__':
    if settings.TF_CPU_ONLY:
        config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        # don't allocate the entire vRAM initially
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)

    # initialize all variables
    session.run(tf.global_variables_initializer())

    # initialize the logger
    logger = Logger(logdir='./log/', overwrite=True)
    logger.register_variables(['loss'] + ['l_abs_pred_{}'.format(i) for i in
                                          range(settings.N_LAYERS)],
                              print_variables=['loss'])
    # initialize the PPC wrapper
    ppc = PPCWrapper(settings.PATH_NO_ABS_PPC, settings.PATH_REAL_PPC)

    # initialize the data handler
    data_handler = DataHandler(ppc, settings.PATH_DATA)

    if not settings.RUN_SIMULATIONS:
        # load dataset
        data_hits = data_handler.init_string_dataset(settings.FLASHER_STRING)
        logger.message("Loaded string dataset with {} batches."
                       .format(data_handler.n_batches))

    # --------------------------------- Run -----------------------------------
    logger.message("Starting...")
    for global_step in range(settings.MAX_STEPS):
        if settings.RUN_SIMULATIONS:
            # Flash all DOMs on the choosen flasher string
            logger.message("Running PPC to flash DOMs...",
                           global_step*settings.OPTIMIZER_STEPS_PER_SIMULATION)
            data_hits, simulated_photons = \
                data_handler.generate_string_batch(settings.FLASHER_STRING,
                                                   settings.PHOTONS_PER_FLASH)

        else:
            # get next batch
            logger.message("Loading next batch...",
                           global_step*settings.OPTIMIZER_STEPS_PER_SIMULATION)
            simulated_photons = data_handler.load_next_batch()

        # calculate the scaling factor
        scale = settings.TF_HITLIST_LEN/len(simulated_photons)
        logger.message("Scaling factor is {0:.3f}".format(scale),
                       global_step*settings.OPTIMIZER_STEPS_PER_SIMULATION)

        # initialize tf data variables
        session.run(init_data,
                    feed_dict={tf_data_hits_placeholder: data_hits*scale,
                               tf_simulated_photons_placeholder:
                               simulated_photons[:settings.TF_HITLIST_LEN]})

        for optimizer_step in range(settings.OPTIMIZER_STEPS_PER_SIMULATION):
            step = global_step*settings.OPTIMIZER_STEPS_PER_SIMULATION \
                + optimizer_step + 1
            # compute and apply gradients and get the loss with this data
            logger.message("Running TensorFlow session to get gradients...",
                           step)

            step_loss, step_hits_pred = \
                session.run([optimize, loss, hits_pred])[1:]

            # get updated parameters
            result = session.run(model.l_abs)

            # log everything
            logger.log(step, [step_loss] + result.tolist())
            # logger.save_hitlists(step, data[0], step_hits_pred)

            # and save it once every write interval
            if step % settings.WRITE_INTERVAL == 0:
                logger.write()

        if settings.LEARNING_DECAY and step % settings.LEARNING_STEPS == 0:
            learning_rate = session.run(update_learning_rate)
            logger.message("Learning rate decreased to {:2.4f}"
                           .format(learning_rate), step)
            if learning_rate <= 0:
                break
    logger.message("Done.")

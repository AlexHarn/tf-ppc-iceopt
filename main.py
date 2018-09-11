from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import tqdm

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
tf_data_hits = tf.placeholder(dtype=settings.TF_FLOAT_PRECISION,
                              shape=(settings.N_DOMS))

tf_simulated_photons = tf.placeholder(dtype=settings.TF_FLOAT_PRECISION,
                                      shape=(None, settings.N_LAYERS + 1))

# initialize the model
model = Model(settings.INITIAL_ABS)

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

# grab all trainable variables
trainable_variables = tf.trainable_variables()

# define variables to save the gradients in each batch
accumulated_gradients = [tf.Variable(tf.zeros_like(tv.initialized_value()),
                                     trainable=False) for tv in
                         trainable_variables]

# define operation to reset the accumulated gradients to zero
reset_gradients = [gradient.assign(tf.zeros_like(gradient)) for gradient in
                   accumulated_gradients]

# compute the gradients
gradients = optimizer.compute_gradients(loss, trainable_variables)

# Note: Gradients is a list of tuples containing the gradient and the
# corresponding variable so gradient[0] is the actual gradient. Also divide
# the gradients by BATCHES_PER_STEP so the learning rate still refers to
# steps not batches.

# define operation to evaluate a batch and accumulate the gradients
evaluate_batch = [
    accumulated_gradient.assign_add(gradient[0]/settings.BATCHES_PER_STEP)
    for accumulated_gradient, gradient in zip(accumulated_gradients,
                                              gradients)]

# define operation to apply the gradients
apply_gradients = optimizer.apply_gradients([
    (accumulated_gradient, gradient[1]) for accumulated_gradient, gradient
    in zip(accumulated_gradients, gradients)])

# define variable and operations to track the average batch loss
average_loss = tf.Variable(0., trainable=False)
update_loss = average_loss.assign_add(loss)
reset_loss = average_loss.assign(0.)

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
    data_handler = DataHandler(ppc)

    # --------------------------------- Run -----------------------------------
    logger.message("Starting...")

    for global_step in range(settings.MAX_STEPS):
        logger.message("Running PPC to flash DOMs...",
                       global_step*settings.OPTIMIZER_STEPS_PER_SIMULATION)
        # Flash all DOMs on string 63
        batches = data_handler.generate_string_batches(
            63, settings.BATCHES_PER_STEP, settings.PHOTONS_PER_FLASH)

        for optimizer_step in range(settings.OPTIMIZER_STEPS_PER_SIMULATION):
            # initialize arrays to log real and predicted hits
            step_hits_true = np.zeros(settings.N_DOMS, dtype=np.int32)
            step_hits_pred = np.zeros(settings.N_DOMS, dtype=np.float)

            step = global_step*settings.OPTIMIZER_STEPS_PER_SIMULATION \
                + optimizer_step + 1
            # compute and apply gradients and get the loss with this data
            logger.message("Running TensorFlow session to get gradients...",
                           step)
            for batch in tqdm(batches, leave=False):
                batch_hits_pred = \
                    session.run([evaluate_batch, update_loss, hits_pred],
                                feed_dict={tf_data_hits:
                                           batch['data_hits'],
                                           tf_simulated_photons:
                                           batch['simulated_photons']})[2]

                # add the hits up for logging
                step_hits_true += batch['data_hits']
                step_hits_pred += batch_hits_pred

            # apply accumulated gradients
            session.run(apply_gradients)

            # get loss
            step_loss = session.run(average_loss)

            # reset variables for next step
            session.run([reset_gradients, reset_loss])

            # get updated parameters
            result = session.run(model.l_abs)

            # log everything
            logger.log(step, [step_loss] + result.tolist())
            logger.save_hitlists(step, step_hits_true, step_hits_pred)

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

from __future__ import division
import tensorflow as tf
import numpy as np

from logger import Logger
from model import Model
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

# placeholder to feed the photon paths. Shape(N_PHOTONS, N_LAYERS + 2),
# +2 because of the DOM id and wavelength in nm
tf_simulated_photons_placeholder = \
        tf.placeholder(dtype=settings.TF_FLOAT_PRECISION,
                       shape=(settings.TF_HITLIST_LEN, settings.N_LAYERS + 2))

# define data variables
tf_data_hits = tf.get_variable("data_hits", dtype=settings.TF_FLOAT_PRECISION,
                               shape=(settings.N_DOMS), trainable=False)
tf_simulated_photons = tf.get_variable("simulated_photons",
                                       dtype=settings.TF_FLOAT_PRECISION,
                                       shape=(settings.TF_HITLIST_LEN,
                                              settings.N_LAYERS + 2),
                                       trainable=False)

# define operations to initialize data variables
init_data = [tf_data_hits.assign(tf_data_hits_placeholder),
             tf_simulated_photons.assign(tf_simulated_photons_placeholder)]

# initialize the model
model = Model(settings.INITIAL_ABS, settings.ICE_MODEL_PATH)

# define operation to reset trained parameters
reset_paras = model.abs_coeff.assign(settings.INITIAL_ABS)

# define hitlists
hits_true = tf_data_hits
hits_pred = model.tf_expected_hits(tf_simulated_photons)

hits_pred = tf.where(hits_true > 0, hits_pred, tf.zeros_like(hits_pred))
hits_true = tf.where(hits_pred > 0, hits_true, tf.zeros_like(hits_true))

# rescale the total number of hits to be equal for simulation and data to
# correct for the flasher LED output uncertainty. We rescale to a fixed
# constant to make the loss more comparable between different batches
hits_true_rescaled = hits_true*settings.RESCALED_HITS/tf.reduce_sum(hits_true)
hits_pred_rescaled = hits_pred*settings.RESCALED_HITS \
    / tf.reduce_sum(tf.stop_gradient(hits_pred))

# definine the objective function
if settings.LOSS == 'Simple Poisson':
    mu = (hits_pred_rescaled + hits_true_rescaled)/2
    logLR_doms = hits_pred_rescaled*(
        tf.log(mu) - tf.log(hits_pred_rescaled)) + \
        hits_true_rescaled*(tf.log(mu) - tf.log(hits_true_rescaled))
    loss = -tf.reduce_sum(tf.where(tf.is_nan(logLR_doms),
                                   tf.zeros_like(logLR_doms), logLR_doms))
elif settings.LOSS == 'Model Error':
    logR_doms = tf.log(hits_pred_rescaled/hits_true_rescaled)
    loss = tf.sqrt(tf.nn.moments(tf.where(tf.is_nan(logR_doms),
                                          tf.zeros_like(logR_doms), logR_doms),
                                 axes=[0])[1])
else:
    raise ValueError(settings.LOSS + " is not a supported objective function!")


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

if settings.GRADIENT_AVERAGING:
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
        accumulated_gradient.assign_add(gradient[0])
        for accumulated_gradient, gradient in zip(accumulated_gradients,
                                                  gradients)]

    # define operation to apply the gradients
    apply_gradients = optimizer.apply_gradients([
        (accumulated_gradient, gradient[1]) for accumulated_gradient, gradient
        in zip(accumulated_gradients, gradients)])

    # define variable and operations to track the average batch loss
    tf_step_loss = tf.Variable(0., trainable=False)
    update_loss = tf_step_loss.assign_add(loss)
    reset_loss = tf_step_loss.assign(0.)
else:  # No gradient averaging
    optimize = optimizer.minimize(loss)

# create operations to clip coefficients
clipped = tf.where(model.abs_coeff < settings.MIN_ABS,
                   tf.ones_like(model.abs_coeff)*settings.MIN_ABS,
                   model.abs_coeff)
clipped = tf.where(model.abs_coeff > settings.MAX_ABS,
                   tf.ones_like(model.abs_coeff)*settings.MAX_ABS,
                   clipped)
clip_coefficients = model.abs_coeff.assign(clipped)

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
    logger.register_variables(['loss'] + ['abs_coeff_pred_{}'.format(i) for i
                                          in range(settings.N_LAYERS)],
                              print_variables=['loss'])

    # initialize the data handler
    data_handler = DataHandler(data_dir=settings.DATA_PATH,
                               photon_dir=settings.PHOTON_PATH)

    # get a string iterator
    string_iter = data_handler.get_string_iterator()

    # --------------------------------- Run -----------------------------------
    logger.message("Starting...")
    step = 1
    while step <= settings.MAX_STEPS:
        for string in settings.FLASHER_STRINGS:
            logger.message("Loading first batch for string {}..."
                           .format(string), step)
            n_doms = 0
            for dom, data_hits, simulated_photons \
                    in string_iter(string):
                logger.message("Loaded next batch. DOM is {}".format(dom),
                               step)
                n_doms += 1

                # initialize tf data variables
                session.run(
                    init_data,
                    feed_dict={tf_data_hits_placeholder: data_hits,
                               tf_simulated_photons_placeholder:
                               simulated_photons[:settings.TF_HITLIST_LEN]})

                if settings.GRADIENT_AVERAGING:
                    # compute gradients and update the loss with this data
                    logger.message(
                        "Running TensorFlow session to get gradients...", step)

                    # calculate gradients for this dom
                    session.run([evaluate_batch, update_loss])
                else:
                    # compute and apply gradients and get the loss with this
                    # data
                    logger.message(
                        "Running TensorFlow session to get gradients...", step)
                    step_loss = session.run([optimize, loss])[1]

                    # get updated parameters and reset negative coefficients
                    result = session.run([model.abs_coeff,
                                          clip_coefficients])[0]

                    # log everything
                    logger.log(step, [step_loss] + result.tolist())

                logger.message("Done with this batch.", step)
                if not settings.GRADIENT_AVERAGING:
                    step += 1
                    # and save it once every write interval
                    if step % settings.WRITE_INTERVAL == 0:
                        logger.write()
                    if settings.LEARNING_DECAY and \
                       step % settings.LEARNING_STEPS == 0:
                        learning_rate = session.run(update_learning_rate)
                        logger.message("Learning rate decreased to {:2.4f}"
                                       .format(learning_rate), step)
                        if learning_rate <= 0:
                            break

            if settings.GRADIENT_AVERAGING:
                logger.message("Applying gradients...", step)
                session.run(apply_gradients)

                # get updated parameters and reset negative coefficients
                result = session.run([model.abs_coeff, clip_coefficients])[0]

                # get the loss
                step_loss = session.run(tf_step_loss)/n_doms

                # log everything
                logger.log(step, [step_loss] + result.tolist())

                # and save it once every write interval
                if step % settings.WRITE_INTERVAL == 0:
                    logger.write()

                # reset variables for next step
                session.run([reset_gradients, reset_loss])
                step += 1

                if settings.LEARNING_DECAY \
                        and step % settings.LEARNING_STEPS == 0:
                    learning_rate = session.run(update_learning_rate)
                    logger.message("Learning rate decreased to {:2.4f}"
                                   .format(learning_rate), step)
                    if learning_rate <= 0:
                        break

    logger.message("Done.")

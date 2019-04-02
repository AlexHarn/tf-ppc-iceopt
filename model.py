import tensorflow as tf
import numpy as np
from tqdm import trange

import settings


class Model:
    """
    Holds the absorption coefficients of the model we are trying to fit.

    Parameters
    ----------
    initial_abs : Array of floats
        The initial absorption coeffiecients for all layers
    """
    def __init__(self, initial_abs, ice_model_dir):
        abs_above = tf.Variable(initial_abs[0:settings.FIRST_INSIDE_LAYER],
                                dtype=settings.TF_FLOAT_PRECISION,
                                trainable=False)
        self.abs_inside = tf.Variable(
            initial_abs[settings.FIRST_INSIDE_LAYER:settings.FIRST_INSIDE_LAYER
                        + settings.N_INSIDE_LAYERS],
            dtype=settings.TF_FLOAT_PRECISION)
        abs_below = tf.Variable(initial_abs[settings.FIRST_INSIDE_LAYER +
                                            settings.N_INSIDE_LAYERS:],
                                dtype=settings.TF_FLOAT_PRECISION,
                                trainable=False)

        self.abs_coeff = tf.concat([abs_above, self.abs_inside, abs_below],
                                   axis=0)

        # load the delta tau table
        depth, scatc, absc, delta_t = np.loadtxt(ice_model_dir+'icemodel.dat',
                                                 unpack=True)
        self._delta_t = delta_t[::-1]

        # load 6 parameter ice model parameters
        paras = np.loadtxt(ice_model_dir+'icemodel.par', unpack=True)[0]
        self._ice_parameters = {'alpha': paras[0],
                                'kappa': paras[1],
                                'A': paras[2],
                                'B': paras[3]
                                }

    def tf_expected_hits(self, simulated_photons):
        """
        Calculates the expected hits for each DOM.

        Parameters
        ----------
        simulated_photons : TF tensor, shape(?, N_LAYER + 1)
            The placeholder for simulated photons, where each row belongs to a
            single photon. The first column is the id of the hit DOM while all
            remaining columns contain the travel distance in the respective
            layer.

        Returns
        -------
        The expected hits for each DOM. TF Tensor of shape (N_DOMS,).
        """
        # dirty, maybe seperate input
        traveled_distances = simulated_photons[:, 2:]
        dom_ids = simulated_photons[:, 0]
        wavelengths = tf.tile(tf.expand_dims(
            simulated_photons[:, 1], 1), [1, settings.N_LAYERS])

        # calculate the the total absorption coefficient (as defined in section
        # 4 of the SPICE paper)
        a_dust = tf.expand_dims(self.abs_coeff, 0) \
            * (wavelengths/400.)**(-self._ice_parameters['kappa'])
        a_intrinsic = self._ice_parameters['A']*tf.exp(
            -self._ice_parameters['B']/wavelengths) \
            * (1 + .01*self._delta_t)
        a_total = a_dust + a_intrinsic

        # calculate hit probability p for each photon, which is the 1 - p_abs
        # where p_abs is the probability for the photon to be absorbed at a
        # distance smaller than the traveled distance before it hit the DOM
        p = tf.exp(-tf.reduce_sum(a_total*traveled_distances, axis=1))

        hitlist = []
        print("Building hitlist subgraph:")
        for i in trange(settings.N_DOMS, leave=False):
            hit_mask = tf.equal(dom_ids, i)
            hitlist.append(tf.reduce_sum(tf.where(hit_mask, p,
                                                  tf.zeros_like(p))))
        print("Done.")

        return tf.stack(hitlist)

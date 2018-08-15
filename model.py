import tensorflow as tf
import settings


class Model:
    """
    Holds the absorption coefficients of the model we are trying to fit.

    Parameters
    ----------
    initial_abs : Array of floats
        The initial absorption lengths for all layers
    """
    def __init__(self, initial_abs):
        self.l_abs = tf.Variable(initial_abs,
                                 dtype=settings.TF_FLOAT_PRECISION)

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
        traveled_distances = simulated_photons[:, 1:]
        dom_ids = simulated_photons[:, 0]

        # calculate hit probability p for each photon, which is the 1 - p_abs
        # where p_abs is the probability for the photon to be absorbed at a
        # distance smaller than the traveled distance before it hit the DOM
        p = tf.exp(-tf.reduce_sum(
            1./tf.expand_dims(self.l_abs, 0)*traveled_distances,
            axis=1))

        hitlist = []
        for i in range(settings.N_DOMS):
            print(i)
            hitlist.append(tf.reduce_sum(tf.where(i == dom_ids, p,
                                                  tf.zeros_like(p))))

        return tf.stack(hitlist)

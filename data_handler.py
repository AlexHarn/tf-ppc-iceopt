import numpy as np
from tqdm import trange
import settings


class DataHandler:
    """
    This class takes care of handling the data and generating batches for
    training.

    Parameters
    ----------
    ppc : Instance of PPCWrapper
        The PPCWrapper object to use.
    """
    def __init__(self, ppc):
        self._ppc = ppc

    def generate_string_batch(self, string, photons_per_flash):
        """
        Generates a single batch by flashing all doms on one single string.

        Parameters
        ----------
        photons_per_flash : Integer
            The number of photons to emit per DOM.

        Returns
        -------
        A pair of data_hits and simulated_photons the way PPCWrapper returns
        them.
        """
        data_hits = np.zeros(settings.N_DOMS, dtype=np.int32)
        simulated_photons = []

        # iterate over DOMs
        for dom in trange(1, 61):
            data_hits += self._ppc.simulate_flash(string, dom,
                                                  photons_per_flash)
            simulated_photons.append(
                self._ppc.simulate_flash_no_abs(string, dom,
                                                photons_per_flash))

        simulated_photons = np.concatenate(simulated_photons)
        np.random.shuffle(simulated_photons)

        return (data_hits, simulated_photons)

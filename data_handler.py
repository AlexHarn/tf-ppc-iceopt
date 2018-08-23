import numpy as np
from tqdm import tqdm
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

    def generate_string_batches(self, string, n_batches, photons_per_flash,
                                shuffle=False):
        """
        Generates batches by flashing all strings on one single string.

        Parameters
        ----------
        n_batches : Intege
            The number of batches the entire run should be devided into.
        photons_per_flash : Integer
            The number of photons to emit per DOM.
        shuffle : Boolean
            If True the DOMs are being shuffled so that each batch does not
            necessarily contain adjacent DOMs.

        Returns
        -------
        A list of batches. Each batch is a pair of data_hits and
        simulated_photons the way PPCWrapper returns them.
        """
        # devide DOMs on into batches
        dom_batches = np.array_split(np.arange(1, 61), n_batches)

        # shuffle if asked for
        if shuffle:
            np.random.shuffle(dom_batches)

        # create list to fill up
        batches = []
        with tqdm(total=60, leave=False) as progress:
            for doms in dom_batches:
                data_hits = np.zeros(settings.N_DOMS, dtype=np.int32)
                simulated_photons = []
                for dom in doms:
                    data_hits += self._ppc.simulate_flash(63, dom,
                                                          photons_per_flash)
                    simulated_photons.append(
                        self._ppc.simulate_flash_no_abs(63, dom,
                                                        photons_per_flash))
                    progress.update(1)

                simulated_photons = np.concatenate(simulated_photons)
                batches.append({'data_hits': data_hits,
                                'simulated_photons': simulated_photons})
        return batches

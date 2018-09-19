import numpy as np
from tqdm import trange
import settings
import os


class DataHandler:
    """
    This class takes care of handling the data and generating batches for
    training.

    Parameters
    ----------
    ppc : Instance of PPCWrapper
        The PPCWrapper object to use.
    data_dir : String
        Path to the data directory, whith saved simulations. Can be populated
        by using save_string_batch().
    """
    def __init__(self, ppc=None, data_dir=None):
        self._ppc = ppc
        self._data_dir = data_dir
        if data_dir is not None:
            self._string_batch_dir = data_dir + 'string_batches/'
        self._dataset_initialized = False

    def generate_string_batch(self, string, photons_per_flash):
        """
        Generates a single batch by flashing all doms on one single string.

        Parameters
        ----------
        string : Integer
            The string to flash.
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

    def save_string_batch(self, string, photons_per_flash):
        """
        Generates a single batch by flashing all doms on one single string, and
        saves it to the string batch directory.

        Parameters
        ----------
        string : Integer
            The string to flash.
        photons_per_flash : Integer
            The number of photons to emit per DOM.
        """
        target_dir = self._string_batch_dir + str(string) + '/'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        data_hits, simulated_photons = \
            self.generate_string_batch(string, photons_per_flash)

        batch_id = 0
        while os.path.exists(target_dir + '{}_no_abs.npy'.format(batch_id)):
            batch_id += 1

        with open(target_dir + '{}_real.npy'.format(batch_id), 'wb') as handle:
            np.save(handle, data_hits)
        with open(target_dir + '{}_no_abs.npy'.format(batch_id),
                  'wb') as handle:
            np.save(handle, simulated_photons)

    def init_string_dataset(self, string, iterate=True):
        """
        Initializes all variables to use the load_next_batch method to load
        batches.

        Parameters
        ----------
        string : Integer
            The string id, it's the subdirectory name like it is written by
            save_string_batch.
        iterate : Boolean
            If True the batches are loaded iteratively, otherwise a random
            batch is given by load_next_batch every time.

        Returns
        -------
        The summed up data hits array, scaled down to single batch size.
        """
        self._dataset_initialized = True
        self._current_dataset_dir = self._string_batch_dir + str(string) + '/'
        if not os.path.exists(self._current_dataset_dir):
            raise ValueError("Dataset directory does not exist!")
        files = next(os.walk(self._current_dataset_dir))[2]
        self.n_batches = len(files)/2
        self._iterate = iterate

        # init count to iterate batches or choose a random one
        if iterate:
            self._next_batch = 0
        else:
            self._next_batch = np.random.randint(0, self.n_batches)

        # load total data hits
        data_hits = np.zeros(settings.N_DOMS, dtype=np.float)
        for batch_id in range(self.n_batches):
            with open(self._current_dataset_dir +
                      '{}_real.npy'.format(batch_id), 'rb') as handle:
                data_hits += np.load(handle)

        return data_hits/self.n_batches

    def load_next_batch(self, shuffle=False):
        """
        Loads the next batch. init_string_dataset has to be called first.

        Parameters
        ----------
        shuffle : Boolean
           Whether or not to shuffle the simulated_photons array before
           returning it. This can be useful if the entire batch is not loaded
           because of memory limitations for the gradient computation (e.g.
           when the scaling factor is < 1) to still use more of the data when
           iterating over the entire dataset multiple times.
           CAUTION: Very slow!

        Returns
        -------
        The simulated_photons array of the next batch.
        """
        if not self._dataset_initialized:
            raise DataInitError("A dataset has to be initialized first!")

        # load the batch
        with open(self._current_dataset_dir +
                  '{}_no_abs.npy'.format(self._next_batch), 'rb') as handle:
            simulated_photons = np.load(handle)

        # choose random batch if we are not iterating
        if self._iterate:
            self._next_batch += 1
            if self._next_batch >= self.n_batches:
                self._next_batch = 0
        else:
            self._next_batch = np.random.randint(0, self.n_batches)

        if shuffle:
            np.random.shuffle(simulated_photons)

        return simulated_photons


class DataInitError(Exception):
    pass

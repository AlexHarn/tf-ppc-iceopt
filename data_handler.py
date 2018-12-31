import numpy as np
import pandas as pd
import multiprocessing as mp
import os
from time import sleep


class DataHandler:
    """
    This class takes care of handling the data and generating batches for
    training.

    Parameters
    ----------
    ppc : Instance of PPCWrapper
        The PPCWrapper object to use.
    data_dir : String
        Path to the data directory.
    """
    def __init__(self, ppc=None, data_dir=None, photon_dir=None):
        self._ppc = ppc
        self._data_dir = data_dir
        self._photon_dir = photon_dir
        if photon_dir is not None:
            self._dom_batch_dir = photon_dir + 'dom_batches/'
        self._dataset_initialized = False

    def load_flasher_data(self, string, dom):
        """
        Loads the detector response from the flasher data set for the specified
        emitter DOM.

        Parameters
        ----------
        string : Integer
            The string to which the emitter DOM belongs (1 to 86).
        DOM : Integer
            The DOM on that specific string, which is the emitter OM (1 to 60).

        Returns
        -------
        None if there is no data for the specified emitter
        Otherwise: A numpy array of length 5160, where each entry is the total
        charge measured by the according DOM. The indexing is done in the usual
        way: 60*(int(string) - 1) + int(dom) - 1.
        """
        fname = self._data_dir + '{}_{}'.format(string, dom)
        if not os.path.isfile(fname):
            return None

        hits = np.loadtxt(fname)
        return hits

    def load_photons(self, string, dom):
        """
        Loads the photons, that were simulated without absorption, for the
        specified emitter dom from the photon_dir.

        Parameters
        ----------
        string : Integer
            The string to which the emitter DOM belongs (1 to 86).
        DOM : Integer
            The DOM on that specific string, which is the emitter OM (1 to 60).

        Returns
        -------
        None if there is no data in the directory for the specified dom.
        Otherwise: The simulated_photons array.
        """
        fname = self._photon_dir + '{}_{}.photons'.format(string, dom)
        if not os.path.isfile(fname):
            return None

        df = pd.read_csv(fname, header=None).fillna(0.)
        return df.values

    def get_string_iterator(self, n_threads=4, q_size=4):
        """
        This method generates a python iterator object, to iteratoe through dom
        batches. It loads the data and especially the simulated photons in
        parallel, to reduce the bottleneck. It is still the bottleneck of the
        entire program though.

        Parameters
        ----------
        n_threads : Integer
            The number of worker threads to launch for loading batches in the
            background.
        q_size : Integer
            The maximum length of the batch queue, in which the worker threads
            put the batches.

        Returns
        -------
        Python iterator object which returns the DOM id, data hits and
        simulated photons array when iterated on. The argument for the iterator
        object is the id of the string to iterate over.
        """
        class StringIter(object):
            def __init__(self, data_handler):
                self._data_handler = data_handler

            def __call__(self, string):
                self._string = string
                self._active_threads = mp.Value('i', n_threads)
                self._dom = mp.Value('i', 1)
                self._q = mp.Queue(maxsize=q_size)
                return self

            def __iter__(self):
                def _producer(next_dom, active_threads, lock):
                    while True:
                        with lock:
                            dom = next_dom.value
                            if dom < 61:
                                next_dom.value += 1
                            else:
                                active_threads.value -= 1
                        if dom == 61:
                            # wait for all threads to finish before issuing the
                            # termination signal
                            while active_threads.value > 0:
                                sleep(5)
                            self._q.put(None)
                            return
                        data_hits = \
                            self._data_handler.load_flasher_data(self._string,
                                                                 dom)
                        if data_hits is None:
                            continue
                        simulated_photons = \
                            self._data_handler.load_photons(self._string, dom)
                        self._q.put([dom, data_hits, simulated_photons],
                                    block=True)

                lock = mp.Lock()
                for i in range(n_threads):
                    thread = mp.Process(target=_producer,
                                        args=[self._dom, self._active_threads,
                                              lock])
                    thread.daemon = True
                    thread.start()

                for data in iter(self._q.get, None):
                    yield data[0], data[1], data[2]
                self._q.close()

        return StringIter(self)


class DataInitError(Exception):
    pass

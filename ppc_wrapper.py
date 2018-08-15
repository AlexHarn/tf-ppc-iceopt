import numpy as np
import subprocess
import os
import pandas as pd


class PPCWrapper:
    """
    This class is used as a python wrapper for the modified standalone version
    of PPC to propagate photons without absorption while logging their travel
    distance in each layer to be able to calculate the absorption probability
    in TensorFlow.

    In this proof of concept project it is also used to generate fake data
    using an unmodified PPC version.

    Parameters
    ----------
    path_no_abs_ppc : string
        The path to the modified PPC executable, which simulates flasher runs
        without absorption and logs the traveled distance for each photon in
        each layer. It is assumed to be in an ice/ folder with configuration
        files.
    path_real_ppc : string
        The following sets the path to the unmodified PPC executable, which is
        used to simulate fake data to fit to. It is assumed to be in an ice/
        folder, which contains the necessary configuration files, including the
        ice parameters which we try to recover.
    n_layers : integer
        Number of layers in the ice model that is used by the simulation
        without absorption. Defaults to 171.
    """
    def __init__(self, path_no_abs_ppc, path_real_ppc, n_layers=171):
        self._ppc = path_no_abs_ppc
        self._real_ppc = path_real_ppc
        self._n_layers = n_layers

        # set wavelength for all photons to 400 nm for now
        os.environ["WFLA"] = "400"

    def simulate_flash(self, string, dom, n_photons):
        """
        Simulates a single flash of DOM 'dom' on string 'string' with
        'n_photons' photons being emitted by the flasher board with absorption
        enabled to generate fake data.

        Parameters
        ----------
        string : integer
            The id of the string the DOM to flash belongs to. Can be any
            integer between 1 and 86.
        dom : integer
            The id of the DOM to flash on the given string. Can be any integer
            between 1 and 60.
        n_photons : integer
            The amount of photons to be emitted by the flasher board.

        Returns
        -------
        The hit list which length is the number of DOMs. Each entry with index
        i = 60*(string - 1) + DOM - 1 is the number of hits for the specific
        DOM.
        """
        assert isinstance(string, int) and string >= 1 and string <= 86
        assert isinstance(dom, int) and dom >= 1 and dom <= 60

        p = subprocess.Popen([self._real_ppc, str(string), str(dom),
                              str(n_photons)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             cwd=os.path.dirname(self._real_ppc))
        out, err = p.communicate()
        df = pd.read_csv(pd.compat.StringIO(out.decode()), header=None)

        # The number of DOMs is 86*60 = 5160
        hit_list = np.zeros(5160, dtype=np.int32)
        for _, hit in df.iterrows():
            hit_list[60*(int(hit[0]) - 1) + int(hit[1]) - 1] += 1
        return hit_list

    def simulate_flash_no_abs(self, string, dom, n_photons):
        """
        Simulates a single flash of DOM 'dom' on string 'string' with
        'n_photons' photons being emitted by the flasher board without
        absorption.

        Parameters
        ----------
        string : integer
            The id of the string the DOM to flash belongs to. Can be any
            integer between 1 and 86.
        dom : integer
            The id of the DOM to flash on the given string. Can be any integer
            between 1 and 60.
        n_photons : integer
            The amount of photons to be emitted by the flasher board.

        Returns
        -------
        An array where each row describes a photon, that would have hit a DOM
        if there was no absorption. The first column contains the id of the
        DOM. The remaining columns contain the travel distance in the
        respective layer. So there are N_LAYERS + 1 columns.
        """
        assert isinstance(string, int) and string >= 1 and string <= 86
        assert isinstance(dom, int) and dom >= 1 and dom <= 60

        p = subprocess.Popen([self._ppc, str(string), str(dom),
                              str(n_photons)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             cwd=os.path.dirname(self._real_ppc))
        out, err = p.communicate()
        df = pd.read_csv(pd.compat.StringIO(out.decode()),
                         header=None).fillna(0.)
        # drop time and wavelength info
        df = df.drop(columns=[2, 3])

        # convert string, DOM format to single DOM id
        for idx, photon in df.iterrows():
            df.ix[idx, 0] = int(60*(photon[0] - 1) + photon[1] - 1)
        df = df.drop(columns=[1])
        return df.values

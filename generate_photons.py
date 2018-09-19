from ppc_wrapper import PPCWrapper
from data_handler import DataHandler
import settings

# initialize the PPC wrapper
ppc = PPCWrapper(settings.PATH_NO_ABS_PPC, settings.PATH_REAL_PPC)

# initialize the data handler
data_handler = \
        DataHandler(ppc, "/net/big-tank/POOL/users/aharnisch/iceopt_photons/")

while True:
    data_handler.save_string_batch(36, settings.PHOTONS_PER_FLASH)

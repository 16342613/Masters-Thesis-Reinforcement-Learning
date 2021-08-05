from A3C_Server import A3C_Server
from A3C_Master import A3C_Master

import os
# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

masterTrainer = A3C_Master(6, 2)
server = A3C_Server("A3C Backend", 8000, masterTrainer, verboseLogging=False)
server.initialise_server()

from A3C_Server import A3C_Server
from A3C_Master import A3C_Master
from Global import Global

import os
# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
Global.masterTrainer = A3C_Master(4, 4)

server = A3C_Server("A3C Backend", 8000, verboseLogging=False)
server.initialise_server()

# import os
# import tensorflow as tf
#
# projectDirectory = r"E:\Users\mandh\Masters-Thesis-Reinforcemenet-Learning\Tanks DRL"
# replayBufferPath = "Assets/API Entry/AI/Replay Buffer.txt"
# os.chdir(projectDirectory)
# print(os.getcwd())
#
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import os

# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from DeepQLearner import DeepQLearner
from Server import Server

dql = DeepQLearner(0.99, 0.5, (0.1, 0.9), 5, (18,), 9)
dql.start_learning()

server = Server("DRL Backend", 8000, verboseLogging=False)
server.set_deep_q_learner(dql)
server.start_server()

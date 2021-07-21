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

dql = DeepQLearner(0.99, 0.5, (0.1, 0.9), 5, (19,), 10)
dql.set_up_networks()

server = Server("DRL Backend", 8000, verboseLogging=False)
server.set_deep_q_learner(dql)
server.start_server()


# message = "PREDICT >|< UnityEngine.Vector3 | UnityEngine.Vector3 | UnityEngine.Vector3 | System.Int32 | UnityEngine.Vector3 | System.Single | UnityEngine.Vector3 | System.Single >|< (5.0, 0.6, 5.2) | (0.0, 0.0, 0.0) | (0.0, 0.0, 0.0) | 260 | (8.6, 0.8, 7.2) | 175 | (0.5, 0.2, 0.5) | 100"
# splitMessage = message.split(" >|< ")
# x = server.commands[splitMessage[0]](" >|< ".join([splitMessage[i + 1] for i in range(len(splitMessage) - 1)]))
# print(x)

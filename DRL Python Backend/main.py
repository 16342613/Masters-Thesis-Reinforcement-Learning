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



dql = DeepQLearner(0.99, 0.5, (0.1, 0.9), 5, (18,), 5)
dql.start_learning()

server = Server("DRL Backend", 8000)
server.set_deep_q_learner(dql)
server.start_server()

# server.deepQLearner.predict_action(
#     "UnityEngine.Vector3 | UnityEngine.Vector3 | UnityEngine.Vector3 | System.Single | UnityEngine.Vector3 | System.Single | UnityEngine.Vector3 | System.Single >|< (5.0, 0.6, 5.2) | (0.0, 0.0, 0.0) | (0.0, 0.0, 0.0) | 500 | (0.0, 0.2, 0.0) | 175 | (0.0, 0.0, 0.0) | -1")
# PREDICT >|< UnityEngine.Vector3 | UnityEngine.Vector3 | UnityEngine.Vector3 | System.Single | UnityEngine.Vector3 | System.Single | UnityEngine.Vector3 | System.Single >|< (5.0, 0.6, 5.2) | (0.0, 0.0, 0.0) | (0.0, 0.0, 0.0) | 500 | (0.0, 0.2, 0.0) | 175 | (0.0, 0.0, 0.0) | -1

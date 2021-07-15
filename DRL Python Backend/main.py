import os
import tensorflow as tf

projectDirectory = r"E:\Users\mandh\Masters-Thesis-Reinforcemenet-Learning\Tanks DRL"
replayBufferPath = "Assets/API Entry/AI/Replay Buffer.txt"
os.chdir(projectDirectory)
print(os.getcwd())

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

x = (18,)
print(x)

# You would have four different actions: moving forward/backward, moving up/down, rotating gun on x-axis, rotating gun on y-axis


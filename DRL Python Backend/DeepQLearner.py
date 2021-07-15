import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import Model
from ReplayBuffer import ReplayBuffer


class DeepQLearner:
    def __init__(self, replayBufferPath, discountFactor, epsilon, epsilonBounds, batchSize, inputShape, actionCount):
        self.replayBufferPath = replayBufferPath
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.epsilonBounds = epsilonBounds
        self.batchSize = batchSize
        self.inputShape = inputShape
        self.actionCount = actionCount

        self.predictionNetwork = None
        self.targetNetwork = None
        self.replayBuffer = None


    def sample_replay_buffer(self):
        pass


    def create_neural_network(self):
        # Create the neural network
        inputLayer = layers.Input(self.inputShape)

        hiddenLayers = layers.Dense(10, activation="relu")(inputLayer)
        hiddenLayers = layers.Dense(10, activation="relu")(hiddenLayers)
        hiddenLayers = layers.Dense(10, activation="relu")(hiddenLayers)

        outputLayer = layers.Dense(self.actionCount, activation="linear")(hiddenLayers)

        return Model(inputs=inputLayer, outputs=outputLayer)


    def start_learning(self):
        self.predictionNetwork = self.create_neural_network()
        self.targetNetwork = self.create_neural_network()

        # Set up the replay buffer file
        self.replayBuffer = ReplayBuffer(self.replayBufferPath)

        pass

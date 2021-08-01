from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import Input
import keras

class A3C_NN(keras.Model):
    def __init__(self, inputSize, actionCount):
        super(A3C_NN, self).__init__()
        self.inputSize = inputSize
        self.actionCount = actionCount
        self.model = self._build_model()

    def predict(self, state):
        outputs = self.model.predict(state)
        return outputs

    def _build_model(self):
        inputLayer = Input(batch_shape=(None, self.inputSize))

        hiddenLayers = layers.Dense(64, activation="relu")(inputLayer)
        hiddenLayers = layers.Dense(64, activation="relu")(hiddenLayers)
        hiddenLayers = layers.Dense(64, activation="relu")(hiddenLayers)

        policyOutput = layers.Dense(self.actionCount, activation="softmax")(hiddenLayers)
        valueOutput = layers.Dense(1, activation="linear")(hiddenLayers)

        model = Model(inputs=[inputLayer], outputs=[policyOutput, valueOutput])

        return model

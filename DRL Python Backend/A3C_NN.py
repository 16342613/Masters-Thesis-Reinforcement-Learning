from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import Input
import keras

class A3C_NN(keras.Model):
    def __init__(self, inputSize, actionCount):
        super(A3C_NN, self).__init__()
        self.inputSize = inputSize
        self.actionCount = actionCount
        #self.model = self._build_model()

        self.dense1 = layers.Dense(64, activation='relu')
        # self.batchNorm1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(64, activation='relu')
        # self.batchNorm2 = layers.BatchNormalization()
        self.dense3 = layers.Dense(64, activation='relu')
        # self.batchNorm3 = layers.BatchNormalization()
        self.dense4 = layers.Dense(64, activation='relu')
        # self.batchNorm4 = layers.BatchNormalization()
        self.policy_logits = layers.Dense(self.actionCount, activation="softmax")
        self.values = layers.Dense(1, activation="linear")

    def get_prediction(self, state, parseString=True):
        outputs = self(state)
        outputs = [outputs[0].numpy(), outputs[1].numpy()]
        if parseString is False:
            return outputs
        else:
            stringOutput = ""

            for i in range(len(outputs)):
                for j in range(len(outputs[i][0])):
                    stringOutput += str(outputs[i][0][j])
                    stringOutput += " | "

                stringOutput = stringOutput[:-3]
                stringOutput += " >|< "

            return stringOutput[:-5]

    def call(self, inputs):
        # hiddenLayers = layers.Dense(64, activation="relu")(inputs)
        # hiddenLayers = layers.Dense(64, activation="relu")(hiddenLayers)
        # hiddenLayers = layers.Dense(64, activation="relu")(hiddenLayers)
        #
        # policyOutput = layers.Dense(self.actionCount, activation="softmax")(hiddenLayers)
        # valueOutput = layers.Dense(1, activation="linear")(hiddenLayers)
        #
        # return policyOutput, valueOutput

        # Forward pass

        # x = self.dense1(inputs)
        # logits = self.policy_logits(x)
        # v1 = self.dense2(inputs)
        # values = self.values(v1)

        x = self.dense1(inputs)
        # x = self.batchNorm1(x)
        x = self.dense2(x)
        # x = self.batchNorm2(x)
        x = self.dense3(x)
        # x = self.batchNorm3(x)
        x = self.dense4(x)
        # x = self.batchNorm4(x)
        logits = self.policy_logits(x)
        values = self.values(x)

        return logits, values

    def _build_model(self):
        inputLayer = Input(batch_shape=(None, self.inputSize))

        hiddenLayers = layers.Dense(64, activation="relu")(inputLayer)
        hiddenLayers = layers.Dense(64, activation="relu")(hiddenLayers)
        hiddenLayers = layers.Dense(64, activation="relu")(hiddenLayers)

        policyOutput = layers.Dense(self.actionCount)(hiddenLayers)
        valueOutput = layers.Dense(1)(hiddenLayers)

        model = Model(inputs=[inputLayer], outputs=[policyOutput, valueOutput])

        # print(model.summary())

        return model

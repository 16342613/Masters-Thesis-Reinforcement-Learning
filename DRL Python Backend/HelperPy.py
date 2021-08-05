import numpy as np


class HelperPy:
    def __init__(self):
        self.parseMapping = dict({"UnityEngine.Vector3": self.parse_vector3,
                                  "System.Single": self.parse_float,
                                  "System.Int32": self.parse_float  # Treat integers as floats
                                  })


    def parse_string_input(self, stringInput, delimiter=" | "):
        types = stringInput.split(" >|< ")[0].split(delimiter)
        splitData = stringInput.split(" >|< ")[1].split(delimiter)
        parsedData = []

        for dataIndex in range(len(splitData)):
            parsedData.extend(self.parseMapping[types[dataIndex]](splitData[dataIndex]))

        return np.array(parsedData).reshape(1, len(parsedData))


    def parse_vector3(self, inputString):
        stringData = inputString.replace("(", "").replace(")", "").split(", ")
        return [float(data) for data in stringData]


    def parse_float(self, inputString):
        return [float(inputString)]

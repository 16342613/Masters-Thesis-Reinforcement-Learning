from HelperPy import HelperPy


class StateTransition:
    def __init__(self, initialState, action, reward, newState, ID, terminalState=0):
        self.initialState = initialState
        self.action = action
        self.reward = reward
        self.newState = newState
        self.ID = ID
        self.terminalState = terminalState

    @staticmethod
    def string_to_transition(stringInput):
        helper = HelperPy()
        splitString = stringInput.split(" >|< ")

        return StateTransition(initialState=helper.parse_string_input(" >|< ".join([splitString[0], splitString[1]]))[0],
                               action=helper.parse_float(splitString[2])[0],
                               reward=helper.parse_float(splitString[3])[0],
                               newState=helper.parse_string_input(" >|< ".join([splitString[4], splitString[5]]))[0],
                               ID=int(helper.parse_float(splitString[6])[0]),
                               terminalState=int(helper.parse_float(splitString[7])[0]))

    @staticmethod
    def state_to_string(state):
        outputString = ""
        for value in state:
            outputString += str(value)

        return outputString[:-3]

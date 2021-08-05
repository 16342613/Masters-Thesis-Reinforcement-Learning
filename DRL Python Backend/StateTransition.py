from HelperPy import HelperPy

class StateTransition:
    def __init__(self, initialState, action, reward, newState, ID, terminalState = 0):
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

        return StateTransition(helper.parse_string_input(" >|< ".join([splitString[0], splitString[1]]))[0],
                                         helper.parse_float(splitString[2])[0],
                                         helper.parse_float(splitString[3])[0],
                                         helper.parse_string_input(" >|< ".join([splitString[4], splitString[5]]))[0],
                                         int(helper.parse_float(splitString[6])[0]))
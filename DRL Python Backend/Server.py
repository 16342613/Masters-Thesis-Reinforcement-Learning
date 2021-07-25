import socket
from DeepQLearner import DeepQLearner
from datetime import datetime


class Server:
    def __init__(self, name, port, maxConnections=100, verboseLogging=True):
        self.name = name
        self.port = port
        self.maxConnections = maxConnections
        self.serverSocket = None
        self.verboseLogging = verboseLogging

        self.running = False
        # Initialise an empty dql learner. This is just a placeholder
        self.deepQLearner = DeepQLearner(0, 0, 0, 0)
        # This will be populated when the dql learner ia set
        self.commands = dict()

    def set_deep_q_learner(self, deepQLearner):
        self.deepQLearner = deepQLearner
        self.commands = dict({"PREDICT": self.deepQLearner.predict_action,
                              "BUILD_BUFFER": self.deepQLearner.add_to_replay_buffer,
                              "TRAIN": self.deepQLearner.train,
                              "UPDATE_TARGET_NETWORK": self.deepQLearner.update_target_network,
                              "TEST_CONNECTION": self.__connection_test,
                              "ECHO": self.__echo,
                              "PLOT": self.deepQLearner.plot_progress,
                              "SAVE_NETWORKS": self.deepQLearner.save_models})

    def start_server(self):
        self.serverSocket = socket.socket()
        self.serverSocket.bind(('', self.port))
        self.serverSocket.listen(self.maxConnections)

        self.__log_data("Started server at " + socket.gethostname() + " on port " + str(self.port), True)

        (clientSocket, address) = self.serverSocket.accept()
        self.__log_data(clientSocket.getsockname()[0] + " has connected to the server", True)
        self.running = True

        while self.running is True:
            message = clientSocket.recv(1024).decode()
            if len(message) > 0:
                self.__log_data("Received request " + message + " from " + clientSocket.getsockname()[0])
                splitMessage = message.split(" >|< ")

                if len(splitMessage) > 1:
                    # Accept the request and send the response
                    response = self.commands[splitMessage[0]](
                        " >|< ".join([splitMessage[i + 1] for i in range(len(splitMessage) - 1)]))
                    clientSocket.send(str.encode(response))
                    self.__log_data("Sent response " + response + " to " + clientSocket.getsockname()[0])
                else:
                    # Handle the request, but no response is necessary
                    self.commands[splitMessage[0]]()

    def __log_data(self, toPrint, overrideLogPermissions=False):
        if (self.verboseLogging is True) or (overrideLogPermissions is True):
            print(datetime.now().strftime("%H:%M:%S") + " : " + toPrint)

    def __connection_test(self, clientMessage):
        return "Hello from " + socket.gethostname() + " on port " + str(
            self.port) + ". I have received your message with content < " + clientMessage + " >."

    def __echo(self, toEcho):
        try:
            self.__log_data(toEcho, overrideLogPermissions=True)
            return "1"
        except:
            return "0"

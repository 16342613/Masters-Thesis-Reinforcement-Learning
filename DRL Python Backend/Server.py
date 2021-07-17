import socket
from DeepQLearner import DeepQLearner
from datetime import datetime


class Server:
    def __init__(self, name, port, maxConnections = 100):
        self.name = name
        self.port = port
        self.maxConnections = maxConnections
        self.serverSocket = None

        self.running = False
        # Initialise an empty dql learner. This is just a placeholder
        self.deepQLearner = DeepQLearner(0, 0, 0, 0, 0, 0)
        # This will be populated when the dql learner ia set
        self.commands = dict()

    def set_deep_q_learner(self, deepQLearner):
        self.deepQLearner = deepQLearner
        self.commands = dict({"PREDICT": self.deepQLearner.predict_action})


    def start_server(self):
        self.serverSocket = socket.socket()
        self.serverSocket.bind(('', self.port))
        self.serverSocket.listen(self.maxConnections)

        self.__log_data("Started server at " + socket.gethostname() + " on port " + str(self.port))

        (clientSocket, address) = self.serverSocket.accept()
        self.__log_data(clientSocket.getsockname()[0] + " has connected to the server")
        self.running = True

        while self.running is True:
            message = clientSocket.recv(1024).decode()
            if len(message) > 0:
                self.__log_data("Received request from " + clientSocket.getsockname()[0])
                splitMessage = message.split(" >|< ")
                response = self.commands[splitMessage[0]](splitMessage[1] + " >|< " + splitMessage[2])
                clientSocket.send(str.encode(response))
                self.__log_data("Sent response to " + clientSocket.getsockname()[0])


    def __log_data(self, toPrint):
        print(datetime.now().strftime("%H:%M:%S") + " : " + toPrint)
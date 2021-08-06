import socket
from threading import Thread
from DeepQLearner import DeepQLearner
from datetime import datetime
from Global import Global

from A3C_Master import A3C_Master

class A3C_Server:
    def __init__(self, name, port, maxConnections=100, verboseLogging=True):
        self.name = name
        self.port = port
        self.maxConnections = maxConnections
        self.serverSocket = None
        self.verboseLogging = verboseLogging

        self.running = False
        # Initialise an empty dql learner. This is just a placeholder
        self.deepQLearner = DeepQLearner(0, 0, 0, 0)

        self.clients = []


    def handle_new_client(self, clientSocket):
        self.clients.append(clientSocket)
        assignedWorker = Global.masterTrainer.assign_worker(clientSocket.getsockname()[0])
        commands = dict({"PREDICT": Global.masterTrainer.global_predict,
                         "SEND_A3C_TRANSITION": assignedWorker.append_to_buffer,
                         "TEST_CONNECTION": self.__connection_test,
                         "ECHO": self.__echo,
                         "UPDATE_REWARD": self.deepQLearner.update_reward,
                         "SAVE_NETWORKS": Global.masterTrainer.save_network,
                         "PLOT": Global.masterTrainer.plot_progress})

        while True:
            message = clientSocket.recv(1024).decode()
            if len(message) > 0:
                self.__log_data("Received request " + message + " from " + clientSocket.getsockname()[0])
                splitMessage = message.split(" >|< ")

                if len(splitMessage) > 1:
                    # Accept the request and send the response

                    response = commands[splitMessage[0]](" >|< ".join([splitMessage[i + 1] for i in range(len(splitMessage) - 1)]))
                    clientSocket.send(str.encode(response))
                    self.__log_data("Sent response " + response + " to " + clientSocket.getsockname()[0])
                else:
                    # Handle the request, but no response is necessary
                    commands[splitMessage[0]]()


    def initialise_server(self):
        self.serverSocket = socket.socket()
        self.serverSocket.bind(('', self.port))
        self.serverSocket.listen(self.maxConnections)


        self.__log_data("Started server at " + socket.gethostname() + " on port " + str(self.port), True)
        self.running = True

        while True:
            (clientSocket, address) = self.serverSocket.accept()
            clientThread = Thread(target=self.handle_new_client, args=(clientSocket, ))
            clientThread.start()
            self.__log_data("Accepted client " + clientSocket.getsockname()[0], True)



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

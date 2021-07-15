import socket

class Server:
    def __init__(self):
        pass


listenSocket = socket.socket()
port = 8000
maxConnections = 999
IP = socket.gethostname()

listenSocket.bind(('', port))

listenSocket.listen(maxConnections)
print("Server started at " + IP + " on port " + str(port))

(clientSocket, address) = listenSocket.accept()
print("New connection made!")

running = True

serverCommands = dict({"Add to replay buffer":  })

while running is True:
    message = clientSocket.recv(1024).decode()
    if len(message) > 0:
        print("Request: " + message)
        clientSocket.send(b"YEE")
        print("Response: YEE")


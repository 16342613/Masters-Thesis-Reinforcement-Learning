using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Diagnostics;

public class CommunicationClient
{
    private int port = -1;
    private string hostName = "EMPTY";
    private NetworkStream stream;
    private byte[] sendData;
    private byte[] receiveData;
    private TcpClient client;
    private string logFilePath;
    private bool connectedToServer;



    public CommunicationClient(string logFilePath)
    {
        this.logFilePath = logFilePath;
        FileHandler.ClearFile(logFilePath);
    }

    public void ConnectToServer(string hostName, int port)
    {
        try
        {
            this.hostName = hostName;
            this.port = port;
            client = new TcpClient(hostName, port);
            connectedToServer = true;

            LogData("Establised connection with host <" + hostName + "> on port " + port);
        }
        catch (SocketException)
        {
            LogData("ERROR: Could not connect with host <" + hostName + "> on port " + port + "!");
        }
    }

    public void SendMessage(string message)
    {
        if (connectedToServer == false)
        {
            throw new Exception("ERROR: Please set up a connection to the server first!");
        }

        try
        {
            sendData = new byte[Encoding.ASCII.GetByteCount(message)];
            sendData = Encoding.ASCII.GetBytes(message);
            stream = client.GetStream();
            stream.Write(sendData, 0, sendData.Length);

            LogData("Sent the message to the host successfully");
        }
        catch (NullReferenceException)
        {
            LogData("ERROR: Unable to send the message to the host!");
        }
    }

    public string RequestResponse(string message, int timeout = 5000)
    {
        Stopwatch stopwatch = new Stopwatch();
        // Send the request to the server
        SendMessage(message);
        // Wait for a response from the server
        string response = ListenForResponse(timeout);

        return response;
    }

    public string ListenForResponse(int millisecondsTimeout)
    {
        try
        {
            Stopwatch stopwatch = new Stopwatch();

            while (stopwatch.ElapsedMilliseconds < millisecondsTimeout)
            {
                stream = client.GetStream();
                if (client.ReceiveBufferSize > 0)
                {
                    receiveData = new byte[client.ReceiveBufferSize];
                    stream.Read(receiveData, 0, client.ReceiveBufferSize);

                    LogData("Received the response from the host");
                    return Encoding.ASCII.GetString(receiveData);
                }
            }

            LogData("TIMEOUT");
        }
        catch (Exception e)
        {
            LogData("ERROR: " + e.Message + "!");
        }

        return null;
    }

    public void CloseConnection()
    {
        stream.Close();
        client.Close();
        LogData("Terminated the connection to the host <" + hostName + "> on port " + port);
    }

    private void LogData(string data)
    {
        FileHandler.WriteToFile(logFilePath, DateTime.Now.ToString("HH:mm:ss tt") + " : " + data);
    }


}
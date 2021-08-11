using System;
using System.Net.Sockets;
using System.Text;
using System.Diagnostics;
using System.Threading;

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
    private bool verboseLogging;
    public string IP;
    public int currentThreadID = Thread.CurrentThread.ManagedThreadId;

    public CommunicationClient(string logFilePath, string IP = "192.168.56.1", bool verboseLogging = false)
    {
        this.logFilePath = logFilePath;
        this.IP = IP;
        this.verboseLogging = true;
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

            LogData("Establised connection with host " + hostName + " on port " + port, true);
        }
        catch (SocketException)
        {
            LogData("ERROR: Could not connect with host " + hostName + " on port " + port + "!", true);
        }
    }

    public void SendMessage(string message)
    {
        if (connectedToServer == false)
        {
            //throw new Exception("ERROR: Please set up a connection to the server first!");
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
            LogData("ERROR: Unable to send the message to the host!", true);
        }
    }

    public string RequestResponse(string message, int timeout = 5000)
    {
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
            stopwatch.Start();

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

            LogData("TIMEOUT", true);
        }
        catch (Exception e)
        {
            LogData("ERROR: " + e.Message + "!", true);
        }

        return null;
    }

    public void CloseConnection()
    {
        stream.Close();
        client.Close();
        LogData("Terminated the connection to the host " + hostName + " on port " + port);
    }

    private void LogData(string data, bool criticalLog = false)
    {
        if (this.verboseLogging == true || criticalLog == true)
        {
            // FileHandler.WriteToFile(logFilePath, DateTime.Now.ToString("HH:mm:ss tt") + " : " + data);
        }
    }
}

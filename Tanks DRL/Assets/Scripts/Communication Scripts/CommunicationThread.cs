using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

class CommunicationThread
{
    public int port;
    private bool runThread;
    public int currentThreadID = Thread.CurrentThread.ManagedThreadId;

    public CommunicationClient client;
    private string serverRequest = "EMPTY";
    private string serverResponse = "EMPTY";

    public CommunicationThread(int port)
    {
        this.port = port;

        // Create a TCP client and connect to the srver
        client = new CommunicationClient("Assets/Debug/Communication Log.txt", verboseLogging: false);
        client.ConnectToServer("DESKTOP-23VITDP", this.port);
        
        runThread = true;
    }

    public void run()
    {
        while (runThread == true)
        {
            if (serverRequest != "EMPTY")
            {
                // Send the request to the server, and get the response back from the server
                serverResponse = client.RequestResponse(serverRequest);
                // Reset the request string
                serverRequest = "EMPTY";
            }

            // Sleep for 1 millisecond for stability (Is this required on the client?)
            Thread.Sleep(1);
        }
    }


    public void StopThread()
    {
        runThread = false;
    }

    public void SendRequest(string request)
    {
        serverRequest = request;
    }

    public string GetResponse()
    {
        return serverResponse;
    }

    public void ResetResponse()
    {
        serverResponse = "EMPTY";
    }

    public bool CheckResponse()
    {
        if (serverResponse != "EMPTY")
        {
            // A response has been recorded from the server
            return true;
        }
        else
        {
            // The server has not sent back a response yet
            return false;
        }
    }
}

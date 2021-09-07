using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.Threading;

public class MasterTrainerScript : MonoBehaviour
{
    List<GameObject> environments = new List<GameObject>();
    // Change the trainer script type between <> accordingly. In the end, you only need one trainer
    List<MovementTrainerAI> trainingScripts = new List<MovementTrainerAI>();
    List<CommunicationThread> communicationThreads = new List<CommunicationThread>();
    List<bool> episodeFinished = new List<bool>();

    public int environmentCount;
    public GameObject environment;
    public Vector3 offset = new Vector3(0, 0, 10);
    private int currentThreadID = Thread.CurrentThread.ManagedThreadId;
    private List<float> episodeRewards = new List<float>();
    private List<int> episodeSteps = new List<int>();

    // Start is called before the first frame update
    void Start()
    {
        SetUpEnvironments();

        environments = GameObject.FindGameObjectsWithTag("Environment").ToList();

        for (int i = 0; i < environmentCount; i++)
        {
            // This may be null depending on the type of trainer. In the end, you should only use one trainer
            trainingScripts.Add(environments[i].GetComponentInChildren<MovementTrainerAI>());
            // Initalise communication clients at separate ports
            CommunicationThread commThread = new CommunicationThread(8000 + i);
            Thread thread = new Thread(commThread.run);
            thread.Start();

            communicationThreads.Add(commThread);

            episodeFinished.Add(false);
            episodeRewards.Add(0);
            episodeSteps.Add(0);
        }

        for (int i = 0; i < trainingScripts.Count; i++)
        {
            trainingScripts[i].masterTrainer = this;
            trainingScripts[i].AIName = i.ToString();
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.M))
        {
            foreach (MovementTrainerAI trainingScript in trainingScripts)
            {
                trainingScript.TestConnection();
            }
        }

        if (Input.GetKeyDown(KeyCode.B))
        {
            Train();
        }

        if (Input.GetKeyDown(KeyCode.C))
        {
            StartCoroutine(TrainAsynchronous());
        }

        if (Input.GetKeyDown(KeyCode.S))
        {
            DoStep();
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            for (int i = 0; i < environmentCount; i++)
            {
                trainingScripts[i].epsilon = 0;
            }
            ResetAllEnvironments();
        }

        // Debug.Log(GlobalScript.globalEpisodeCount);
    }

    private void SetUpEnvironments()
    {
        Vector3 currentSpawnPoint = Vector3.zero;

        for (int i = 0; i < environmentCount; i++)
        {
            GameObject newEnvironment = GameObject.Instantiate(environment);
            newEnvironment.transform.position = currentSpawnPoint;

            currentSpawnPoint += offset;
        }
    }

    private void Train()
    {
        foreach (MovementTrainerAI trainingScript in trainingScripts)
        {
            StartCoroutine(trainingScript.TrainA3C());
        }
    }

    private IEnumerator TrainAsynchronous(int episodeCount = 1001, int maxEpisodeSteps = 200, int epsilonAnnealInterval = 2, int plotInterval = 50)
    {
        ResetAllEnvironments();

        for (int i = 0; i < episodeCount; i++)
        {

            for (int j = 0; j < maxEpisodeSteps; j++)
            {
                DoAsynchronousTrainingStep();
                yield return new WaitForSeconds(0.001f);

                if (!episodeFinished.Contains(false))
                {
                    break;
                }
            }

            for (int j = 0; j < environmentCount; j++)
            {
                trainingScripts[j].epsilon = 0.1f;
                FileHandler.WriteToFile("Assets/Debug/AI Log.txt", System.DateTime.Now.ToString("HH:mm:ss tt")
                    + " Name : " + trainingScripts[j].AIName
                    + " ==> Episode : " + i
                    + " ; Steps : " + episodeSteps[j]
                    + " ; Reward : " + episodeRewards[j]
                    + " ; Epsilon : " + trainingScripts[j].epsilon);

                communicationThreads[j].SendRequest(ServerRequests.ADD_PLOT_DATA.ToString() + " >|< " + j + " | " + episodeRewards[j]);
                Thread.Sleep(100);
                communicationThreads[j].ClearResponseQueue();
            }

            // Save the networks after a certain amount of episodes
            if ((i % plotInterval == 0) && (i > 0))
            {
                for (int j = 0; j < environmentCount; j++)
                {
                    communicationThreads[j].SendRequest(ServerRequests.SAVE_NETWORKS.ToString() + " >|< " + "0");
                }
                Thread.Sleep(5000);
                WaitForServerComplete();
                CollectServerResponses();
            }

            if ((i % epsilonAnnealInterval == 0) && (trainingScripts[0].epsilon >= 0.11f) && (i > 0))
            {
                for (int j = 0; j < environmentCount; j++)
                {
                    trainingScripts[j].epsilon = trainingScripts[j].epsilon * 0.994f;
                }
            }

            // Reset the environments to facilitate the next episode
            ResetAllEnvironments();
        }

        communicationThreads[0].SendRequest(ServerRequests.PLOT.ToString() + " >|< " + "0");
        WaitForServerComplete();
        CollectServerResponses();
    }

    private List<float> DoAsynchronousTrainingStep()
    {
        // Perform some 'global step update' rather than update the steps individually on each environment
        // Send state transition at time t across all environments to the threaded server, and return the results.

        /*for (int i = 0; i < environmentCount; i++)
        {
            if (episodeFinished[i] == true)
            {
                continue;
            }

            communicationThreads[i].SendRequest(ServerRequests.ECHO.ToString() + " >|< " + "0");
        }
        WaitForServerComplete();
        CollectServerResponses();*/

        List<EnvironmentState> initialStates = new List<EnvironmentState>();
        for (int i = 0; i < environmentCount; i++)
        {
            // If the episode in an environment has reached a terminal state, you can skip to the next environment
            if (episodeFinished[i] == true)
            {
                initialStates.Add(null);
                continue;
            }

            // Observe each environment
            EnvironmentState observedState = trainingScripts[i].ObserveEnvironment();
            initialStates.Add(observedState);

            // Send the observation to the communication thread, which should pass it to the server
            communicationThreads[i].SendRequest(ServerRequests.PREDICT.ToString() + " >|< " + initialStates[i].ToString());
        }
        // Wait for all prediction responses to return back from the server
        WaitForServerComplete();

        // Collect the responses from the server, now that you are sure that all responses have been received
        List<string> predictedActionResponses = CollectServerResponses();
        // Store the reward for the actions (R)
        List<float> rewards = new List<float>();

        int currentIndex = 0;
        // Take a particular action given the response from the server
        for (int i = 0; i < environmentCount; i++)
        {
            // If the episode in an environment has reached a terminal state, you can skip to the next environment
            if (episodeFinished[i] == true)
            {
                continue;
            }

            // Take an action, and record the action taken and the reward (This is different from the normal TakeAction method, as the response
            // from the server is already provided as an input parameter)
            object[] actionReward = new object[] { };
            actionReward = trainingScripts[i].TakeAction(state: initialStates[i], actionRepeat: 1, givenResponse: predictedActionResponses[currentIndex]);

            // Record the next state after taking the action
            EnvironmentState newState = trainingScripts[i].ObserveEnvironment();

            string stateTransitionString = initialStates[i].ToString()
            + " >|< " + actionReward[0].ToString()
            + " >|< " + actionReward[1].ToString()
            + " >|< " + newState.ToString()
            + " >|< " + initialStates[i].ID.ToString()
            + " >|< " + actionReward[2].ToString();

            // communicationThreads[i].SendRequest(ServerRequests.PREDICT.ToString() + " >|< " + initialStates[i].ToString());

            // Send the state transition to the server in order to train the A3C algorithm
            communicationThreads[i].SendRequest(ServerRequests.SEND_A3C_TRANSITION.ToString() + " >|< " + stateTransitionString);
            //rewards.Add(float.Parse(actionReward[1].ToString()));
            episodeRewards[i] += float.Parse(actionReward[1].ToString());
            episodeSteps[i] += 1;
            // If the environment has encountered a terminal state
            if (actionReward[2].ToString() == "1")
            {
                episodeFinished[i] = true;
            }

            currentIndex++;
        }
        // Some A3C appends could trigger a training command on the server, so you may need to wait for the server to finish training
        WaitForServerComplete();
        CollectServerResponses();
        return rewards;
    }

    private void WaitForServerComplete()
    {
        for (int i = 0; i < environmentCount; i++)
        {
            if (episodeFinished[i] == true)
            {
                continue;
            }

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            bool serverResponseReceived = false;
            while (serverResponseReceived == false && sw.ElapsedMilliseconds < 5000)
            {
                serverResponseReceived = communicationThreads[i].CheckResponse();
            }

            if (sw.ElapsedMilliseconds >= 5000)
            {
                throw new System.Exception("TIMEOUT!");
            }

            sw.Stop();
        }
    }

    private List<string> CollectServerResponses()
    {
        List<string> output = new List<string>();

        for (int i = 0; i < environmentCount; i++)
        {
            if (episodeFinished[i] == true)
            {
                continue;
            }

            output.Add(communicationThreads[i].GetResponse());

            // Reset the response
            // communicationThreads[i].ResetResponse();
        }

        return output;
    }

    private void ResetAllEnvironments()
    {
        for (int i = 0; i < environmentCount; i++)
        {
            // Add the rewards for the episode
            trainingScripts[i].ResetEnvironment();
            // Reset the episode finished flag
            episodeFinished[i] = false;
            // Reset the rewards
            episodeRewards[i] = 0;
            // Reset the step counter
            episodeSteps[i] = 0;

            communicationThreads[i].ClearQueues();
        }
    }

    private void DoStep()
    {
        // Perform some 'global step update' rather than update the steps individually on each environment
        // Send state transition at time t across all environments to the threaded server, and return the results.

        // Store the initial observed states (S)
        List<EnvironmentState> initialStates = new List<EnvironmentState>();

        for (int i = 0; i < environmentCount; i++)
        {
            communicationThreads[i].SendRequest(ServerRequests.ECHO.ToString() + " >|< " + "0");
        }
        WaitForServerComplete();
        CollectServerResponses();

        for (int i = 0; i < environmentCount; i++)
        {
            // Observe each environment
            EnvironmentState observedState = trainingScripts[i].ObserveEnvironment();
            initialStates.Add(observedState);

            // If the episode in an environment has reached a terminal state, you can skip to the next environment
            if (episodeFinished[i] == true)
            {
                continue;
            }

            // Send the observation to the communication thread, which should pass it to the server
            communicationThreads[i].SendRequest(ServerRequests.PREDICT.ToString() + " >|< " + initialStates[i].ToString());
        }
        // Wait for all prediction responses to return back from the server
        WaitForServerComplete();

        // Collect the responses from the server, now that you are sure that all responses have been received
        List<string> predictedActionResponses = CollectServerResponses();

        // Take a particular action given the response from the server
        for (int i = 0; i < environmentCount; i++)
        {
            // If the episode in an environment has reached a terminal state, you can skip to the next environment
            if (episodeFinished[i] == true)
            {
                continue;
            }

            // Take an action
            trainingScripts[i].TakeAction(state: initialStates[i], actionRepeat: 5, givenResponse: predictedActionResponses[i]);
        }
    }
}

using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class MovementTrainerAI : TrainerScript
{
    public GameObject agent;
    public GameObject target;

    // Start is called before the first frame update
    void Start()
    {
        client = new CommunicationClient("Assets/Debug/Communication Log.txt");
        client.ConnectToServer("DESKTOP-23VITDP", 8000);

        // Clear the log files
        HelperScript.ClearLogs();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.S))
        {
            ObserveAndTakeAction(ServerRequests.PREDICT);
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            epsilon = 0;
            ResetEnvironment();
        }

        if (Input.GetKeyDown(KeyCode.N))
        {
            Debug.Log("TEE");
            StartCoroutine(TrainA3C());
        }

        //TestReward();
    }

    private EnvironmentState ObserveEnvironment()
    {
        return new EnvironmentState(agent.transform.position, target.transform.position);
    }

    public override object[] TakeAction(EnvironmentState state, int actionRepeat = 1)
    {
        // Predict the action using the NN
        string response = client.RequestResponse(ServerRequests.PREDICT.ToString() + " >|< " + state.ToString());

        if (response.Contains(" >|< ") == true)
        {
            response = response.Split(new string[] { " >|< " }, System.StringSplitOptions.None).ToList()[0];
        }

        List<float> parsedResponse = HelperScript.ParseStringToFloats(response, " | ");
        int bestActionIndex = parsedResponse.IndexOf(parsedResponse.Max());

        // Use epsilon-greedy for exploration
        if (Random.Range(0, 1) < epsilon)
        {
            bestActionIndex = Mathf.RoundToInt(Random.Range(0, (float)(parsedResponse.Count - 1)));
        }

        for (int i = 0; i < actionRepeat; i++)
        {
            switch (bestActionIndex)
            {
                // Move left
                case 0:
                    agent.transform.position -= agent.transform.right * 0.1f;
                    break;

                // Move right
                case 1:
                    agent.transform.position += agent.transform.right * 0.1f;
                    break;

                // Move forward
                case 2:
                    agent.transform.position += agent.transform.forward * 0.1f;
                    break;

                // Move backward
                case 3:
                    agent.transform.position -= agent.transform.forward * 0.1f;
                    break;
            }
        }

        /*bool hitWall = true;
        if (agent.transform.position.z >= 20)
        {
            agent.transform.position = new Vector3(agent.transform.position.x, agent.transform.position.y, 20);
        }
        else if (agent.transform.position.z <= 0)
        {
            agent.transform.position = new Vector3(agent.transform.position.x, agent.transform.position.y, 0);
        }
        else if (agent.transform.position.x >= 20)
        {
            agent.transform.position = new Vector3(20, agent.transform.position.y, agent.transform.position.z);
        }
        else if (agent.transform.position.x <= 0)
        {
            agent.transform.position = new Vector3(0, agent.transform.position.y, agent.transform.position.z);
        }
        else
        {
            hitWall = false;
        }*/

        EnvironmentState newEnvironment = ObserveEnvironment();

        // The default reward
        float reward = 0;

        // If the agent hits the wall
        if (Vector3.Distance(newEnvironment.agentPosition, newEnvironment.targetPosition) < 0.5f)
        {
            reward = 10f;
            return new object[] { bestActionIndex, reward, true }; // Episode has finished
        }
        else
        {
            reward -= Mathf.Pow(1 - (1 / Vector3.Distance(newEnvironment.agentPosition, newEnvironment.targetPosition)), 4);
        }

        /*if (hitWall == true)
        {
            reward = -1f;
        }*/

        return new object[] { bestActionIndex, reward, false };
    }

    private object[] ObserveAndTakeAction(ServerRequests request, int actionRepeat = 5)
    {
        // Send (Old state, action, reward, new state) to replay buffer

        // Observe the environment
        EnvironmentState currentState = ObserveEnvironment();
        // Query the server to get the best action for that observation
        object[] actionReward = TakeAction(currentState, actionRepeat);
        EnvironmentState newState = ObserveEnvironment();

        // Send the state transition data to the server, which puts it into the replay buffer
        string stateTransitionString = currentState.ToString()
            + " >|< " + actionReward[0].ToString()
            + " >|< " + actionReward[1].ToString()
            + " >|< " + newState.ToString()
            + " >|< " + currentState.ID.ToString();

        // Return the size of the replay buffer in the client
        return new object[] { client.RequestResponse(request.ToString() + " >|< " + stateTransitionString), actionReward[2], actionReward[1] };
    }

    public IEnumerator TrainA3C(int episodeCount = 500, int maxEpisodeSteps = 100, int epsilonAnnealInterval = 1, int plotInterval = 100)
    {
        ResetEnvironment();

        int baselineRepeat = 0;
        float baselineReward = 0;
        // epsilon = 0;

        for (int i = 0; i < baselineRepeat; i++)
        {
            for (int j = 0; j < maxEpisodeSteps; j++)
            {
                object[] stepOutput = ObserveAndTakeAction(ServerRequests.PREDICT);

                if ((j % maxEpisodeSteps == 0) && (j > 0))
                {
                    ResetEnvironment();
                }

                baselineReward += float.Parse(stepOutput[2].ToString());

                yield return null;
            }
        }


        FileHandler.WriteToFile("Assets/Debug/AI Log.txt", System.DateTime.Now.ToString("HH:mm:ss tt") + " ==> Baseline Reward : " + (baselineReward / baselineRepeat));
        ResetEnvironment();

        List<float> averageRewards = new List<float>();
        List<float> stepCounts = new List<float>();
        int stepCount = 0;
        for (int i = 0; i < episodeCount; i++)
        {
            // Perform an episode
            float episodeReward = 0;

            for (int j = 0; j < maxEpisodeSteps; j++)
            {
                object[] stepOutput = ObserveAndTakeAction(ServerRequests.SEND_A3C_TRANSITION);
                stepCount++;

                episodeReward += float.Parse(stepOutput[2].ToString());

                // Reset the environment and leave this episode if the episode has terminated
                if (bool.Parse(stepOutput[1].ToString()) == true) { break; }

                yield return new WaitForSeconds(0.001f);
            }

            if (stepCount > 100000)
            {
                stepCount = 0;
            }

            episodeRewards.Add(episodeReward);
            GlobalScript.episodeRewards.Add(episodeReward);
            stepCounts.Add(stepCount);

            FileHandler.WriteToFile("Assets/Debug/AI Log.txt", System.DateTime.Now.ToString("HH:mm:ss tt") + "Name : " + this.AIName + " ==> Episode : " + i + " ; Steps : " + stepCounts[i] + " ; Reward : " + episodeRewards[i] + " ; Epsilon : " + epsilon);


            if ((i % plotInterval == 0) && (i > 0))
            {
                if (AIName == "0")
                {
                    //PlotProgressA3C(AIName, episodeRewards.Skip(episodeRewards.Count - plotInterval).ToList());
                }
                //PlotProgress(averageRewards.Skip(averageRewards.Count - (plotInterval/10)).ToList());
                client.SendMessage(ServerRequests.SAVE_NETWORKS.ToString());
            }

            if ((i % epsilonAnnealInterval == 0) && (epsilon >= 0.1f))
            {
                epsilon = epsilon * 0.998f;
            }

            GlobalScript.globalEpisodeCount++;

            ResetEnvironment();
        }
    }

    public override void ResetEnvironment()
    {
        agent.transform.position = new Vector3(0, 0, 0);
    }

    private void TestReward()
    {
        EnvironmentState newEnvironment = ObserveEnvironment();
        float reward = -Mathf.Pow(1 - (1 / Vector3.Distance(newEnvironment.agentPosition, newEnvironment.targetPosition)), 4);

        Debug.Log(reward);
    }
}

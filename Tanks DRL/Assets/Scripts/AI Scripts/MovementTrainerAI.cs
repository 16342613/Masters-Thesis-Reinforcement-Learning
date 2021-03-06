using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class MovementTrainerAI : TrainerScript
{
    public GameObject agent;
    public GameObject target;
    public bool train = false;
    // public List<int> bestActionIndexes = new List<int>() { 0, 0, 0, 0 };

    // Start is called before the first frame update
    void Start()
    {
        //client = new CommunicationClient("Assets/Debug/Communication Log.txt", verboseLogging: false);
        //client.ConnectToServer("DESKTOP-23VITDP", 8000);

        // Clear the log files
        HelperScript.ClearLogs();
    }

    // Update is called once per frame
    void Update()
    {
        train = false;
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
            train = false;
            //StartCoroutine(TrainA3C());
            StartCoroutine(TrainDQN());
        }

        if (Input.GetKeyDown(KeyCode.X))
        {
            train = false;
        }


        // HelperScript.PrintList(bestActionIndexes);
        // TestReward();
    }

    public EnvironmentState ObserveEnvironment()
    {
        RaycastHit frontRay;
        RaycastHit leftRay;
        RaycastHit rightRay;
        RaycastHit rearRay;

        float frontRayDistance = 100f;
        float rearRayDistance = 100f;
        float leftRayDistance = 100f;
        float rightRayDistance = 100f;

        float dotProduct = Vector3.Dot(agent.transform.forward, (target.transform.localPosition - agent.transform.localPosition).normalized);

        if (Physics.Raycast(agent.transform.position, agent.transform.forward, out frontRay, 20))
        {
            if (frontRay.collider.gameObject.tag == "Obstacle" && (frontRay.collider.gameObject.transform.root.GetInstanceID() == agent.transform.root.GetInstanceID()))
            {
                frontRayDistance = frontRay.distance;
            }
        }
        if (Physics.Raycast(agent.transform.position, -agent.transform.right, out rearRay, 20))
        {
            if (rearRay.collider.gameObject.tag == "Obstacle" && (rearRay.collider.gameObject.transform.root.GetInstanceID() == agent.transform.root.GetInstanceID()))
            {
                rearRayDistance = rearRay.distance;
            }
        }
        if (Physics.Raycast(agent.transform.position, agent.transform.right, out leftRay, 20))
        {
            if (leftRay.collider.gameObject.tag == "Obstacle" && (leftRay.collider.gameObject.transform.root.GetInstanceID() == agent.transform.root.GetInstanceID()))
            {
                leftRayDistance = leftRay.distance;
            }
        }
        if (Physics.Raycast(agent.transform.position, -agent.transform.forward, out rightRay, 20))
        {
            if (rightRay.collider.gameObject.tag == "Obstacle" && (rightRay.collider.gameObject.transform.root.GetInstanceID() == agent.transform.root.GetInstanceID()))
            {
                rightRayDistance = rightRay.distance;
            }
        }

        return new EnvironmentState(agent.transform.localPosition / 20f, target.transform.localPosition / 20f, frontRayDistance / 20f, rearRayDistance / 20f, leftRayDistance / 20f, rightRayDistance / 20f, agent.transform.localEulerAngles / 360f, Vector3.Angle((target.transform.position - agent.transform.position).normalized, agent.transform.forward.normalized) / 360f);
    }

    public override object[] TakeAction(EnvironmentState state = null, int actionRepeat = 1, string givenResponse = null)
    {
        string response = "";
        if (givenResponse == null)
        {
            response = client.RequestResponse(ServerRequests.PREDICT.ToString() + " >|< " + state.ToString());
        }
        else
        {
            response = givenResponse;
        }

        float value = 0;

        if (response.Contains(" >|< ") == true)
        {
            List<string> splitString = response.Split(new string[] { " >|< " }, System.StringSplitOptions.None).ToList();
            value = float.Parse(splitString[1]);
            response = splitString[0];
        }

        List<float> parsedResponse = HelperScript.ParseStringToFloats(response, " | ");
        int bestActionIndex = 0; //HelperScript.SampleProbabilityDistribution(parsedResponse);//parsedResponse.IndexOf(parsedResponse.Max());

        // FOR DQN
        if (train == true)
        {
            bestActionIndex = parsedResponse.IndexOf(parsedResponse.Max());
        }
        // FOR A3C
        else
        {
            bestActionIndex = HelperScript.SampleProbabilityDistribution(parsedResponse);//parsedResponse.IndexOf(parsedResponse.Max());
        }

        System.Random random = new System.Random();
        double rnum = random.NextDouble();
        // Use epsilon-greedy for exploration
        if (rnum < epsilon)
        {
            System.Random rnd = new System.Random();
            bestActionIndex = rnd.Next(parsedResponse.Count);// Mathf.RoundToInt(Random.Range(0, (float)(parsedResponse.Count - 1)));
        }

        // bestActionIndexes[bestActionIndex] += 1;
        float oldDistance = Vector3.Distance(state.agentPosition, state.targetPosition);
        float oldDotProduct = Vector3.Dot(agent.transform.forward, (target.transform.localPosition - agent.transform.localPosition).normalized);

        for (int i = 0; i < actionRepeat; i++)
        {
            switch (bestActionIndex)
            {
                // Go left
                case 0:
                    agent.transform.position -= agent.transform.right * 1f;
                    break;

                // Turn right
                case 1:
                    agent.transform.position += agent.transform.right * 1f;
                    break;
                
                // Go Forward
                case 2:
                    agent.transform.position += agent.transform.forward * 1f;
                    break;

                // Go Backward
                case 3:
                    agent.transform.position -= agent.transform.forward * 1f;
                    break;
                    /*
                    // Move forward
                    case 3:
                        agent.transform.position += agent.transform.forward * 0.1f;
                        break;

                    // Move backward
                    case 4:
                        agent.transform.position -= agent.transform.forward * 0.1f;
                        break;
                    */
            }

            /*
            switch (bestActionIndex)
            {
                // Turn Left
                case 0:
                    agent.transform.localEulerAngles -= new Vector3(0, 1, 0);
                    break;

                // Turn Right
                case 1:
                    agent.transform.localEulerAngles += new Vector3(0, 1, 0);
                    break;
            }
            */
        }

        bool hitWall = true;
        if (agent.transform.localPosition.z >= 20)
        {
            agent.transform.localPosition = new Vector3(agent.transform.localPosition.x, agent.transform.localPosition.y, 19);
        }
        else if (agent.transform.localPosition.z <= 0)
        {
            agent.transform.localPosition = new Vector3(agent.transform.localPosition.x, agent.transform.localPosition.y, 1);
        }
        else if (agent.transform.localPosition.x >= 20)
        {
            agent.transform.localPosition = new Vector3(19, agent.transform.localPosition.y, agent.transform.localPosition.z);
        }
        else if (agent.transform.localPosition.x <= 0)
        {
            agent.transform.localPosition = new Vector3(1, agent.transform.localPosition.y, agent.transform.localPosition.z);
        }
        else
        {
            hitWall = false;
        }

        EnvironmentState newEnvironment = ObserveEnvironment();
        float newDistance = Vector3.Distance(newEnvironment.agentPosition, newEnvironment.targetPosition);
        float newDotProduct = Vector3.Dot(agent.transform.forward, (target.transform.localPosition - agent.transform.localPosition).normalized);
        float angleDifference = Vector3.Angle((target.transform.position - agent.transform.position).normalized, agent.transform.forward.normalized) / 360f;

        // The default reward
        float reward = 0; //-(angleDifference * angleDifference);
        //reward -= newDistance;

        if (newDistance < oldDistance)
        {
            reward += 0.1f;
        }
        else
        {
            reward -= 0.1f;
        }

        // If the agent wants to turn right when the target is to its right
        if ((HelperScript.AngleDir(agent.transform.forward.normalized, (target.transform.localPosition - agent.transform.localPosition).normalized, agent.transform.up.normalized) == 1) && (bestActionIndex == 1)) {
            //reward *= 0.1f;
        }
        // If the agent wants to turn left when the target is to its left
        if ((HelperScript.AngleDir(agent.transform.forward.normalized, (target.transform.localPosition - agent.transform.localPosition).normalized, agent.transform.up.normalized) == -1) && (bestActionIndex == 0))
        {
            //reward *= 0.1f;
        }

        /*if (oldDotProduct < newDotProduct)
        {
            reward += Mathf.Abs(newDotProduct - oldDotProduct) * 3;
        }
        else
        {
            reward -= Mathf.Abs(newDotProduct - oldDotProduct) * 3;
        }

        if (newDistance < oldDistance)
        {
            //reward += 0.1f;
        }
        else
        {
            //reward -= 0.1f;
        }*/


        if (newDistance < 0.1f)//Vector3.Dot(agent.transform.forward, (target.transform.localPosition - agent.transform.localPosition).normalized) > 0.999f)
        {
            reward += 5f;
            return new object[] { bestActionIndex, reward, 1 }; // Episode has finished
        }
        else
        {
            //reward -= Mathf.Pow(1 - (1 / (Vector3.Distance(newEnvironment.agentPosition, newEnvironment.targetPosition) * 20f)), 10);
        }

        //reward = reward * 0.1f;

        if (hitWall == true)
        {
            //reward -= 0.2f;
            //return new object[] { bestActionIndex, reward, 1 }; // Episode has finished
        }

        if (newEnvironment.forwardRay < 0.05f)
        {
            //reward -= 0.2f;
            //return new object[] { bestActionIndex, reward, 1 }; // Episode has finished
        }
        if (newEnvironment.backwardRay < 0.05f)
        {
            //reward -= 0.2f;
            //return new object[] { bestActionIndex, reward, 1 }; // Episode has finished
        }
        if (newEnvironment.leftRay < 0.05f)
        {
            //reward -= 0.2f;
            //return new object[] { bestActionIndex, reward, 1 }; // Episode has finished
        }
        if (newEnvironment.rightRay < 0.05f)
        {
            //reward -= 0.1f;
            //return new object[] { bestActionIndex, reward, 1 }; // Episode has finished
        }

        if (AIName == "0")
        {
            HelperScript.PrintList(parsedResponse);
            Debug.Log(reward + " ; " + value + " ; " + bestActionIndex);
        }


        /*
        float newDotProduct = Vector3.Dot(agent.transform.forward, (target.transform.localPosition - agent.transform.localPosition).normalized);

        float reward = 0;

        EnvironmentState newEnvironment = ObserveEnvironment();
        if (newEnvironment.forwardRay < 5)
        {
            return new object[] { bestActionIndex, 1f, 1 };
        }

        if (oldDotProduct < newDotProduct)
        {
            reward += 0.1f;

            if (newDotProduct > 0.999f)
            {
                return new object[] { bestActionIndex, 1f, 1 };
            }
        }
        else
        {
            reward -= 0.2f;
        }

        if (AIName == "0")
        {
            HelperScript.PrintList(parsedResponse);
            Debug.Log(reward + " ; " + value + " ; " + newDotProduct);
        }
        */
        return new object[] { bestActionIndex, reward, 0 };
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
            + " >|< " + currentState.ID.ToString()
            + " >|< " + actionReward[2].ToString();

        // Return the size of the replay buffer in the client
        return new object[] { client.RequestResponse(request.ToString() + " >|< " + stateTransitionString), actionReward[2], actionReward[1] };
    }

    public IEnumerator TrainDQN(int episodeCount = 1001, int maxEpisodeSteps = 200, int epsilonAnnealInterval = 2, int plotInterval = 100, int networkUpdateInterval = 25)
    {
        ResetEnvironment();

        int baselineRepeat = 25;
        float baselineReward = 0;
        epsilon = 1f;

        for (int i = 0; i < baselineRepeat; i++)
        {
            for (int j = 0; j < maxEpisodeSteps; j++)
            {
                object[] stepOutput = ObserveAndTakeAction(ServerRequests.BUILD_BUFFER);

                baselineReward += float.Parse(stepOutput[2].ToString());

                yield return null;
            }

            ResetEnvironment();
        }

        FileHandler.WriteToFile("Assets/Debug/AI Log.txt", System.DateTime.Now.ToString("HH:mm:ss tt") + " ==> Baseline Reward : " + (baselineReward / baselineRepeat));
        ResetEnvironment();

        List<float> stepCounts = new List<float>();
        int stepCount = 0;
        for (int i = 0; i < episodeCount; i++)
        {
            if (train == false) { yield break; }

            // Perform an episode
            float episodeReward = 0;

            for (int j = 0; j < maxEpisodeSteps; j++)
            {
                stepCounts.Add(0);
                object[] stepOutput = ObserveAndTakeAction(ServerRequests.BUILD_BUFFER);
                stepCount++;


                episodeReward += float.Parse(stepOutput[2].ToString());

                if ((j % 100 == 0) && (j > 0))
                {
                    client.SendMessage(ServerRequests.TRAIN.ToString());
                }

                // Reset the environment and leave this episode if the episode has terminated
                if (int.Parse(stepOutput[1].ToString()) == 1) { break; }
                stepCounts[i]++;
                yield return new WaitForSeconds(0.001f);
            }

            episodeRewards.Add(episodeReward);
            GlobalScript.episodeRewards.Add(episodeReward);

            FileHandler.WriteToFile("Assets/Debug/AI Log.txt", System.DateTime.Now.ToString("HH:mm:ss tt") + " Name : " + this.AIName + " ==> Episode : " + i + " ; Steps : " + stepCounts[i] + " ; Reward : " + episodeRewards[i] + " ; Epsilon : " + epsilon);


            if ((i % plotInterval == 0) && (i > 0))
            {
                //PlotProgress(episodeRewards.Skip(episodeRewards.Count - plotInterval).ToList());
                client.SendMessage(ServerRequests.SAVE_NETWORKS.ToString());
            }

            if ((i % epsilonAnnealInterval == 0) && (epsilon >= 0.1f) && (i > 0))
            {
                epsilon = epsilon * 0.998f;
            }

            if (i % networkUpdateInterval == 0)
            {
                client.RequestResponse(ServerRequests.UPDATE_TARGET_NETWORK.ToString() + " >|< Placeholder");
            }

            GlobalScript.globalEpisodeCount++;

            ResetEnvironment();
        }
    }

    public IEnumerator TrainA3C(int episodeCount = 501, int maxEpisodeSteps = 200, int epsilonAnnealInterval = 2, int plotInterval = 100)
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
            if (train == false) { yield break; }

            // Perform an episode
            float episodeReward = 0;

            for (int j = 0; j < maxEpisodeSteps; j++)
            {
                stepCounts.Add(0);
                object[] stepOutput = ObserveAndTakeAction(ServerRequests.SEND_A3C_TRANSITION);
                stepCount++;


                episodeReward += float.Parse(stepOutput[2].ToString());

                // Reset the environment and leave this episode if the episode has terminated
                if (int.Parse(stepOutput[1].ToString()) == 1) { break; }
                stepCounts[i]++;
                yield return new WaitForSeconds(0.001f);
            }

            if (stepCount > 100000)
            {
                stepCount = 0;
            }

            episodeRewards.Add(episodeReward);
            GlobalScript.episodeRewards.Add(episodeReward);

            FileHandler.WriteToFile("Assets/Debug/AI Log.txt", System.DateTime.Now.ToString("HH:mm:ss tt") + " Name : " + this.AIName + " ==> Episode : " + i + " ; Steps : " + stepCounts[i] + " ; Reward : " + episodeRewards[i] + " ; Epsilon : " + epsilon);


            if ((i % plotInterval == 0) && (i > 0))
            {
                if (AIName == "0")
                {
                    //PlotProgressA3C(AIName, episodeRewards.Skip(episodeRewards.Count - plotInterval).ToList());
                }
                //PlotProgress(averageRewards.Skip(averageRewards.Count - (plotInterval/10)).ToList());
                client.SendMessage(ServerRequests.SAVE_NETWORKS.ToString());
            }

            if ((i % epsilonAnnealInterval == 0) && (epsilon >= 0.1f) && (i > 0))
            {
                epsilon = epsilon * 0.998f;
            }

            GlobalScript.globalEpisodeCount++;

            ResetEnvironment();
        }
    }

    public override void ResetEnvironment()
    {
        //target.transform.localPosition = new Vector3(Random.Range(1, 19f), 0, Random.Range(1, 19f));
        //agent.transform.localPosition = new Vector3(Random.Range(1, 19f), 0, Random.Range(1, 19f));
        agent.transform.localPosition = new Vector3(Random.Range(1, 19f), 0, Random.Range(1, 19f));
        //agent.transform.localEulerAngles = new Vector3(0, Random.Range(0, 360f), 0);
    }

    private void TestReward()
    {
        EnvironmentState newEnvironment = ObserveEnvironment();
        float reward = -Mathf.Pow(1 - (1 / (Vector3.Distance(newEnvironment.agentPosition, newEnvironment.targetPosition) * 20f)), 10);

        Debug.Log(reward);
    }
}

using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;

public class ArmourTrainerAI : TrainerScript
{
    public GameObject shooter;
    public GameObject target;
    public GameObject round;
    private TankControllerScript targetScript;

    public Vector3 maximumBounds;
    public Vector3 minimumBounds;
    public float proximityThreshold;
    public float aimVariance = 30;
    public Transform environmentTransform;

    public string replayBufferPath;
    public MasterTrainerScript masterTrainer;

    // Start is called before the first frame update
    void Start()
    {
        targetScript = target.GetComponent<TankControllerScript>();
        environmentTransform = this.gameObject.transform.parent.transform;

        client = new CommunicationClient("Assets/Debug/Communication Log.txt");
        client.ConnectToServer("DESKTOP-23VITDP", 8000);

        // Clear the log files
        HelperScript.ClearLogs();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            AimShooter();
            FireRound();
        }

        if (Input.GetKeyDown(KeyCode.S))
        {
            ObserveAndTakeAction(ServerRequests.PREDICT);
        }

        if (Input.GetKeyDown(KeyCode.X))
        {
            //Train();
            StartCoroutine(Train());
        }

        if (Input.GetKeyDown(KeyCode.U))
        {
            client.SendMessage(ServerRequests.UPDATE_TARGET_NETWORK.ToString());
        }

        if (Input.GetKeyDown(KeyCode.T))
        {
            TestConnection();
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            epsilon = 0;
            ResetEnvironment();
        }

        if (Input.GetKeyDown(KeyCode.N))
        {
            StartCoroutine(TrainA3C());
        }

        //TestReward();
    }

    public void TestConnection()
    {
        string response = client.RequestResponse(ServerRequests.TEST_CONNECTION.ToString() + " >|< Connection test! ");
        Debug.Log(response);
    }

    /// <summary>
    /// Used in training to aim the shooter at the tank
    /// </summary>
    private void AimShooter()
    {
        shooter.transform.position = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
        while (Vector3.Distance(shooter.transform.position, target.transform.position) < proximityThreshold)
        {
            shooter.transform.position = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
        }

        shooter.transform.LookAt(target.transform.position);
        bool aimingAtTarget = false;

        /*while (aimingAtTarget == false)
        {
            shooter.transform.Rotate(Random.Range(-aimVariance, aimVariance), Random.Range(-aimVariance, aimVariance), 0);
            RaycastHit hit;
            if (Physics.Raycast(shooter.transform.position, shooter.transform.forward, out hit, Mathf.Infinity))
            {
                if (hit.collider.transform.root.tag == "Enemy Tank")
                {
                    aimingAtTarget = true;
                }
            }
        }*/
        //shooter.transform.Rotate(Random.Range(-aimVariance, aimVariance), Random.Range(-aimVariance, aimVariance), 0);
        //shooter.transform.eulerAngles = new Vector3(shooter.transform.eulerAngles.x, shooter.transform.eulerAngles.y, 0);
    }

    private void FireRound(int previousStateID = 0)
    {
        GameObject firedRound = Instantiate(round, shooter.transform.position, shooter.transform.rotation);
        firedRound.GetComponent<Rigidbody>().velocity = firedRound.transform.forward * (firedRound.GetComponent<ShellScript>().muzzleVelocity / 10f); // should be divided by 5, not 10! This was set to 10 for training
        firedRound.GetComponent<ShellScript>().stateID = previousStateID;
        firedRound.GetComponent<ShellScript>().SetAITrainer(this);
    }

    public void UpdateReward(int stateID, float newReward)
    {
        client.RequestResponse(ServerRequests.UPDATE_REWARD.ToString() +
            " >|< " + stateID.GetType().ToString() + " | " + newReward.GetType().ToString() +
            " >|< " + stateID + " | " + newReward);
    }

    private EnvironmentState ObserveEnvironment()
    {
        Vector3 targetLocation = target.transform.position;
        Vector3 targetHullAngle = target.transform.Find("Hitbox/Hull").transform.rotation.eulerAngles;
        Vector3 targetTurretAngle = target.transform.Find("Hitbox/Turret").transform.rotation.eulerAngles;
        int targetHitpoints = targetScript.GetHitpoints();
        int targetMaxHitpoints = targetScript.GetMaxHitpoints();

        Vector3 shooterLocation = shooter.transform.position;
        Vector3 shooterDirection = shooter.transform.forward.normalized;
        float shooterPenetration = round.GetComponent<ShellScript>().penetration;
        Vector3 aimedLocation = Vector3.zero;
        int tankIndex = -1;
        float plateThickness = -1;
        RaycastHit hit;

        if (Physics.Raycast(shooter.transform.position, shooter.transform.forward, out hit, 200))
        {
            if (hit.collider.gameObject.tag == "Armour Plate")
            {
                // aimedLocation = hit.collider.transform.InverseTransformPoint(hit.point);
                aimedLocation = hit.point;
                plateThickness = hit.collider.transform.GetComponent<ArmourPlateScript>().armourThickness;
                tankIndex = hit.collider.transform.GetComponentInParent<TankControllerScript>().tankIndex;
            }
            else
            {
                aimedLocation = shooterLocation + (shooterDirection * 200f);
            }
        }

        return new EnvironmentState(targetLocation, targetHullAngle, targetTurretAngle, targetHitpoints, targetMaxHitpoints, tankIndex, shooterLocation, shooterDirection, shooterPenetration, aimedLocation, plateThickness);
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
        string oldHitpointsString = targetScript.GetHitpoints().ToString();
        string oldHitpointsDeepCopy = string.Copy(oldHitpointsString);
        int oldHitpoints = int.Parse(oldHitpointsDeepCopy);
        Vector3 oldIdealRotation = (Quaternion.LookRotation(HelperScript.GetDirection(state.shooterLocation, state.enemyLocation), Vector3.up) * Vector3.forward).normalized;
        float oldForwardDissimilarity = (Mathf.Abs(oldIdealRotation.x - state.shooterForward.x) + Mathf.Abs(oldIdealRotation.y - state.shooterForward.y) + Mathf.Abs(oldIdealRotation.z - state.shooterForward.z)) / 3f;

        
        // Use epsilon-greedy for exploration
        if (Random.Range(0, 1) < epsilon)
        {
            bestActionIndex = Mathf.RoundToInt(Random.Range(0, (float)(parsedResponse.Count - 1)));
        }
        

        //bestActionIndex = HelperScript.SampleProbabilityDistribution(parsedResponse);
        bool firedRound = false;

        for (int i = 0; i < actionRepeat; i++)
        {
            /*switch (bestActionIndex)
            {
                // Shoot
                case 0:
                    FireRound(state.ID);
                    firedRound = true;
                    break;

                // Move left
                case 1:
                    shooter.transform.position -= shooter.transform.right * 0.1f;
                    break;

                // Move right
                case 2:
                    shooter.transform.position += shooter.transform.right * 0.1f;
                    break;

                // Move up
                case 3:
                    shooter.transform.position += shooter.transform.up * 0.1f;
                    break;

                // Move down
                case 4:
                    shooter.transform.position -= shooter.transform.up * 0.1f;
                    break;

                // Look left
                case 5:
                    shooter.transform.eulerAngles += new Vector3(0, 1, 0);
                    break;

                // Look right
                case 6:
                    shooter.transform.eulerAngles += new Vector3(0, -1, 0);
                    break;

                // Look up
                case 7:
                    shooter.transform.eulerAngles += new Vector3(-1, 0, 0);
                    break;

                // Look down
                case 8:
                    shooter.transform.eulerAngles += new Vector3(1, 0, 0);
                    break;

                // Take no action
                case 9:
                    break;
            }

            if (firedRound == true) { break; }*/

            switch (bestActionIndex)
            {
                // Move left
                case 0:
                    shooter.transform.position -= shooter.transform.right * 0.1f;
                    break;

                // Move right
                case 1:
                    shooter.transform.position += shooter.transform.right * 0.1f;
                    break;

                    /*// Move up
                    case 2:
                        shooter.transform.position += shooter.transform.up * 0.1f;
                        break;

                    // Move down
                    case 3:
                        shooter.transform.position -= shooter.transform.up * 0.1f;
                        break;*/
            }
        }

        EnvironmentState newEnvironment = ObserveEnvironment();
        if (bestActionIndex == 0)
        {
            newEnvironment.firedRound = 1;
        }
        else
        {
            newEnvironment.firedRound = 0;
        }

        int newHitpoints = targetScript.GetHitpoints();

        Vector3 newIdealRotation = (Quaternion.LookRotation(HelperScript.GetDirection(newEnvironment.shooterLocation, newEnvironment.enemyLocation), Vector3.up) * Vector3.forward).normalized;
        float newForwardDissimilarity = (Mathf.Abs(newIdealRotation.x - newEnvironment.shooterForward.x) + Mathf.Abs(newIdealRotation.y - newEnvironment.shooterForward.y) + Mathf.Abs(oldIdealRotation.z - newEnvironment.shooterForward.z)) / 3f;
        //float reward = 1 / (Vector3.Angle(newEnvironment.shooterForward, idealRotation) * (1 / Vector3.Distance(newEnvironment.shooterLocation, newEnvironment.enemyLocation))) * (1 / Vector3.Distance(newEnvironment.shooterLocation, newEnvironment.enemyLocation)) * 750;

        // The default reward
        float reward = 0;


        if (newForwardDissimilarity < oldForwardDissimilarity)
        {
            //reward += 0.1f;
        }
        else
        {
            //reward -= 0.1f;
        }

        // If the terminal state has been reached
        if (oldHitpoints == 0)
        {
            // You may need to change the final reward
            return new object[] { bestActionIndex, 10, true };
        }
        // This is a bit dodgy! The shell has a travel time so the direct action which fired the shell may not be rewarded!
        // It may also be a good idea to reward even more if the action kills the enemy tank
        else if (oldHitpoints > newHitpoints)
        {
            //reward += (oldHitpoints - newHitpoints);
            reward += (1 - ((oldHitpoints - newHitpoints) / newEnvironment.enemyMaxHitpoints)) * 10f;
        }

        //float reward = 0;//1/Vector3.Distance(newEnvironment.enemyLocation, newEnvironment.shooterLocation);

        if (Mathf.Abs(state.enemyLocation.x - state.shooterLocation.x) > Mathf.Abs(newEnvironment.enemyLocation.x - newEnvironment.shooterLocation.x))
        {
            reward += 0.2f;
        }
        else
        {
            reward = 0f;
        }
        // If the shooter is aiming at the tank, then give a positive reward
        if (newEnvironment.tankIndex >= 0)
        {
            reward += 0.5f;
        }

        // parsedResponse.Add(reward);
        // parsedResponse.Add(parsedResponse[1] - parsedResponse[0]);
        // HelperScript.PrintList(parsedResponse);
        // Debug.Log("Action: " + bestActionIndex + " ; Reward:" + reward);

        return new object[] { bestActionIndex, reward, false };
    }

    private object[] ObserveAndTakeAction(ServerRequests request, int actionRepeat = 6)
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

    public IEnumerator Train(int episodeCount = 75, int maxEpisodeSteps = 100, int networkUpdateInterval = 1000, int epsilonAnnealInterval = 1, int plotInterval = 1000)
    {
        ResetEnvironment();
        //client.SendMessage(ServerRequests.ECHO.ToString() + " >|< Started training!");
        // Build up the replay buffer so you can sample transitions
        int setupIterations = 1000;
        float baselineReward = 0;
        epsilon = 1f;

        for (int i = 0; i < setupIterations; i++)
        {
            object[] stepOutput = ObserveAndTakeAction(ServerRequests.BUILD_BUFFER);

            if ((i % maxEpisodeSteps == 0) && (i > 0))
            {
                ResetEnvironment();
            }

            baselineReward += float.Parse(stepOutput[2].ToString());

            yield return null;
        }

        FileHandler.WriteToFile("Assets/Debug/AI Log.txt", System.DateTime.Now.ToString("HH:mm:ss tt") + " ==> Baseline Reward : " + (baselineReward / setupIterations));
        ResetEnvironment();

        //List<float> rewards = new List<float>();
        List<float> stepCounts = new List<float>();
        int stepCount = 0;
        for (int i = 0; i < episodeCount; i++)
        {
            // Spawn the shooter in a random location within the bounds
            /*shooter.transform.position = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
            while (Vector3.Distance(shooter.transform.position, target.transform.position) < proximityThreshold)
            {
                shooter.transform.position = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
                FileHandler.WriteToFile("Assets/Debug/Debug Log.txt", System.DateTime.Now.ToString("HH:mm:ss tt") + " : Retried spawning");
            }*/

            if (stepCount % networkUpdateInterval == 0)
            {
                client.RequestResponse(ServerRequests.UPDATE_TARGET_NETWORK.ToString() + " >|< Placeholder");
            }

            // Perform an episode
            //List<float> episodeDetails = PerformEpisode(maxEpisodeSteps);
            float episodeReward = 0;

            for (int j = 0; j < maxEpisodeSteps; j++)
            {
                object[] stepOutput = ObserveAndTakeAction(ServerRequests.BUILD_BUFFER);
                stepCount++;

                episodeReward += float.Parse(stepOutput[2].ToString());

                // Reset the environment and leave this episode if the episode has terminated
                if (bool.Parse(stepOutput[1].ToString()) == true) { break; }

                if ((stepCount % 50 == 0) && (stepCount > 0))
                {
                    client.SendMessage(ServerRequests.TRAIN.ToString());
                }

                yield return new WaitForSeconds(0.001f);
            }

            if (stepCount > 100000)
            {
                stepCount = 0;
            }

            episodeRewards.Add(episodeReward);
            stepCounts.Add(stepCount);

            /*if (i % 4 == 0)
            {
                client.SendMessage(ServerRequests.TRAIN.ToString());
            }*/

            //client.SendMessage(ServerRequests.TRAIN.ToString());

            FileHandler.WriteToFile("Assets/Debug/AI Log.txt", System.DateTime.Now.ToString("HH:mm:ss tt") + " ==> Episode : " + i + " ; Steps : " + stepCounts[i] + " ; Reward : " + episodeRewards[i] + " ; Epsilon : " + epsilon);

            // Anneal epsilon over time
            if ((stepCount % epsilonAnnealInterval == 0) && (epsilon >= 0.1f))
            {
                epsilon = epsilon * 0.97f;
            }

            if ((stepCount % plotInterval == 0) && (stepCount > 0))
            {
                PlotProgress(episodeRewards.Skip(episodeRewards.Count - plotInterval).ToList());
                client.SendMessage(ServerRequests.SAVE_NETWORKS.ToString());
            }

            ResetEnvironment();
        }
    }

    public IEnumerator TrainA3C(int episodeCount = 50, int maxEpisodeSteps = 100, int epsilonAnnealInterval = 1, int plotInterval = 100)
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

            if ((i % 10 == 0) && (i > 0))
            {
                // averageRewards.Add(episodeRewards.Skip(episodeRewards.Count - 10).Average());
            }

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
            // Debug.Log(stepCount);
        }
    }

    public void PlotProgressA3C(string environmentName, List<float> latestRewards)
    {
        try
        {
            string requestContent = "";

            for (int i = 0; i < latestRewards.Count; i++)
            {
                if (i < (latestRewards.Count - 1))
                {
                    requestContent += latestRewards[i] + " | ";
                }
                else
                {
                    requestContent += latestRewards[i];
                }
            }

            Debug.Log(ServerRequests.PLOT + " >|< " + environmentName + " >|< " + requestContent);
            client.RequestResponse(ServerRequests.PLOT + " >|< " + environmentName + " >|< " + requestContent);
        }
        catch (System.Exception e)
        {
            FileHandler.WriteToFile("Assets/Debug/AI Log.txt", "PLOT ERROR: " + e.Message);
        }
    }

    public override void ResetEnvironment()
    {
        // Spawn the shooter in a random location within the bounds
        /*shooter.transform.position = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
        while (Vector3.Distance(shooter.transform.position, target.transform.position) < proximityThreshold)
        {
            shooter.transform.position = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
        }
        //shooter.transform.eulerAngles = new Vector3(Random.Range(-45, 45), Random.Range(0, 360), 0);
        AimShooter();*/

        target.transform.position = new Vector3(environmentTransform.position.x + Random.Range(-7.5f, 7.5f), environmentTransform.position.y + 0.2f, environmentTransform.position.z);

        shooter.transform.position = new Vector3(Random.Range(target.transform.position.x - 7.5f, target.transform.position.x + 7.5f), target.transform.position.y, target.transform.position.z - 5f);
        shooter.transform.eulerAngles = new Vector3(0, 0, 0);

        targetScript.ResetHitpoint();
    }

    private void PlotProgress(List<float> rewards)
    {
        try
        {
            string requestContent = "";

            for (int i = 0; i < rewards.Count; i++)
            {
                if (i < (rewards.Count - 1))
                {
                    requestContent += rewards[i] + " | ";
                }
                else
                {
                    requestContent += rewards[i];
                }
            }

            client.RequestResponse(ServerRequests.PLOT + " >|< " + requestContent);
        }
        catch (System.Exception e)
        {
            FileHandler.WriteToFile("Assets/Debug/AI Log.txt", "PLOT ERROR: " + e.Message);
        }
    }

    private void TestReward()
    {
        int oldHitpoints = targetScript.GetHitpoints();

        EnvironmentState newEnvironment = ObserveEnvironment();
        Vector3 idealRotation = (Quaternion.LookRotation(HelperScript.GetDirection(newEnvironment.shooterLocation, newEnvironment.enemyLocation), Vector3.up) * Vector3.forward).normalized;

        float reward = 1 / (Vector3.Angle(newEnvironment.shooterForward, idealRotation) * (1 / Vector3.Distance(newEnvironment.shooterLocation, newEnvironment.enemyLocation))) * (1 / Vector3.Distance(newEnvironment.shooterLocation, newEnvironment.enemyLocation)) * 750;//1 / (Vector3.Angle(newEnvironment.shooterForward, idealRotation) * (1 / Vector3.Distance(newEnvironment.shooterLocation, newEnvironment.enemyLocation))) * (1 / Vector3.Distance(newEnvironment.shooterLocation, newEnvironment.enemyLocation));

        Debug.Log(idealRotation + " ; " + newEnvironment.shooterForward + " ; " + reward + " ; " + newEnvironment.tankIndex);
    }
}

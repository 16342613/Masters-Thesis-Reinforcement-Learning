using System.Collections;
using System.Collections.Generic;
using System.Linq;
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
    public bool train = false;

    // Start is called before the first frame update
    void Start()
    {
        targetScript = target.GetComponent<TankControllerScript>();
        targetScript.AITrainer = this;
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
            train = false;
            //StartCoroutine(Train());
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
            train = true;
            StartCoroutine(TrainA3C());
        }

        // TestReward();
    }

    /// <summary>
    /// Used in training to aim the shooter at the tank
    /// </summary>
    private void AimShooter()
    {
        /* shooter.transform.position = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
        while (Vector3.Distance(shooter.transform.position, target.transform.position) < proximityThreshold)
        {
            shooter.transform.position = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
        }*/

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
        shooter.transform.Rotate(Random.Range(-aimVariance, aimVariance), Random.Range(-aimVariance, aimVariance), 0);
        shooter.transform.eulerAngles = new Vector3(shooter.transform.eulerAngles.x, shooter.transform.eulerAngles.y, 0);
    }

    private void FireRound(int previousStateID = 0)
    {
        GameObject firedRound = Instantiate(round, shooter.transform.position, shooter.transform.rotation);
        firedRound.GetComponent<Rigidbody>().velocity = firedRound.transform.forward * (firedRound.GetComponent<ShellScript>().muzzleVelocity / 10f); // should be divided by 5, not 10! This was set to 10 for training
        firedRound.GetComponent<ShellScript>().stateID = previousStateID;
        firedRound.GetComponent<ShellScript>().SetAITrainer(this);

        Destroy(firedRound, 2f);
    }

    public void UpdateReward(int stateID, float newReward)
    {
        client.RequestResponse(ServerRequests.UPDATE_REWARD.ToString() +
            " >|< " + stateID.GetType().ToString() + " | " + newReward.GetType().ToString() +
            " >|< " + stateID + " | " + newReward);
    }

    private EnvironmentState ObserveEnvironment()
    {
        /*
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
        */

        Vector3 targetLocation = target.transform.position / 20f;
        Vector3 targetHullAngle = target.transform.Find("Hitbox/Hull").transform.rotation.eulerAngles / 360f;
        Vector3 targetTurretAngle = target.transform.Find("Hitbox/Turret").transform.rotation.eulerAngles / 360f;
        float targetMaxHitpoints = (float) targetScript.GetMaxHitpoints();
        float targetHitpoints = ((float) targetScript.GetHitpoints()) / targetMaxHitpoints;

        Vector3 shooterLocation = shooter.transform.position / 20f;
        Vector3 shooterDirection = shooter.transform.forward.normalized;
        float shooterPenetration = round.GetComponent<ShellScript>().penetration / 300f;
        Vector3 aimedLocation = Vector3.zero;
        Vector3 idealForward = (Quaternion.LookRotation(HelperScript.GetDirection(shooterLocation, targetLocation), Vector3.up) * Vector3.forward).normalized;
        int tankIndex = -1;
        float plateThickness = -1;
        RaycastHit hit;

        if (Physics.Raycast(shooter.transform.position, shooter.transform.forward, out hit, 200))
        {
            if (hit.collider.gameObject.tag == "Armour Plate")
            {
                if (hit.collider.gameObject.GetComponentInParent<ArmourTrainerAI>().AIName == AIName)
                {
                    // aimedLocation = hit.collider.transform.InverseTransformPoint(hit.point);
                    aimedLocation = hit.point / 20f;
                    plateThickness = hit.collider.transform.GetComponent<ArmourPlateScript>().armourThickness / 1000f;
                    tankIndex = hit.collider.transform.GetComponentInParent<TankControllerScript>().tankIndex;
                }
            }
            else
            {
                aimedLocation = (shooterLocation + (shooterDirection * 200f)) / 20f;
            }
        }

        return new EnvironmentState(targetLocation, targetHullAngle, targetTurretAngle, targetHitpoints, targetMaxHitpoints, tankIndex, shooterLocation, shooterDirection, idealForward, shooterPenetration, aimedLocation, plateThickness);
    }

    public override object[] TakeAction(EnvironmentState state, int actionRepeat = 1)
    {
        // Predict the action using the NN
        string response = client.RequestResponse(ServerRequests.PREDICT.ToString() + " >|< " + state.ToString());
        float value = 0;

        if (response.Contains(" >|< ") == true)
        {
            List<string> splitString = response.Split(new string[] { " >|< " }, System.StringSplitOptions.None).ToList();
            value = float.Parse(splitString[1]);
            response = splitString[0];
        }

        List<float> parsedResponse = HelperScript.ParseStringToFloats(response, " | ");

        int bestActionIndex = HelperScript.SampleProbabilityDistribution(parsedResponse); //parsedResponse.IndexOf(parsedResponse.Max());
        int observedHitpoints = targetScript.GetHitpoints();
        Vector3 oldIdealRotation = (Quaternion.LookRotation(HelperScript.GetDirection(state.shooterLocation, state.enemyLocation), Vector3.up) * Vector3.forward).normalized;
        float oldForwardDissimilarity = (Mathf.Abs(oldIdealRotation.x - state.shooterForward.x) + Mathf.Abs(oldIdealRotation.y - state.shooterForward.y) + Mathf.Abs(oldIdealRotation.z - state.shooterForward.z)) / 3f;

        float reward = 0;

        // Use epsilon-greedy for exploration
        if (Random.Range(0, 1) < epsilon)
        {
            System.Random rnd = new System.Random();
            bestActionIndex = rnd.Next(parsedResponse.Count);
        }

        bool firedRound = false;

        for (int i = 0; i < actionRepeat; i++)
        {
            /*
            switch (bestActionIndex)
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
            }
            */

            switch (bestActionIndex)
            {
                // Look left
                case 0:
                    shooter.transform.eulerAngles -= new Vector3(0, 1, 0);
                    break;

                // Look right
                case 1:
                    shooter.transform.eulerAngles += new Vector3(0, 1, 0);
                    break;

                // Look up
                case 2:
                    shooter.transform.eulerAngles -= new Vector3(1, 0, 0);
                    break;

                // Look down
                case 3:
                    shooter.transform.eulerAngles += new Vector3(1, 0, 0);
                    break;
            }

            if (firedRound == true) { break; }
        }

        EnvironmentState newEnvironment = ObserveEnvironment();
        bool hitWall = true;
        if (shooter.transform.localPosition.z >= 10)
        {
            shooter.transform.localPosition = new Vector3(shooter.transform.localPosition.x, shooter.transform.localPosition.y, 9);
        }
        else if (shooter.transform.localPosition.z <= -10)
        {
            shooter.transform.localPosition = new Vector3(shooter.transform.localPosition.x, shooter.transform.localPosition.y, -9);
        }
        else if (shooter.transform.localPosition.y >= 10)
        {
            shooter.transform.localPosition = new Vector3(shooter.transform.localPosition.x, 9, shooter.transform.localPosition.z);
        }
        else if (shooter.transform.localPosition.y <= -10)
        {
            shooter.transform.localPosition = new Vector3(shooter.transform.localPosition.x, -9, shooter.transform.localPosition.z);
        }
        else if (shooter.transform.localPosition.x >= 10)
        {
            shooter.transform.localPosition = new Vector3(9, shooter.transform.localPosition.y, shooter.transform.localPosition.z);
        }
        else if (shooter.transform.localPosition.x <= -10)
        {
            shooter.transform.localPosition = new Vector3(-9, shooter.transform.localPosition.y, shooter.transform.localPosition.z);
        }
        else
        {
            hitWall = false;
        }

        Vector3 newIdealRotation = (Quaternion.LookRotation(HelperScript.GetDirection(newEnvironment.shooterLocation, newEnvironment.enemyLocation), Vector3.up) * Vector3.forward).normalized;
        float newForwardDissimilarity = (Mathf.Abs(newIdealRotation.x - newEnvironment.shooterForward.x) + Mathf.Abs(newIdealRotation.y - newEnvironment.shooterForward.y) + Mathf.Abs(newIdealRotation.z - newEnvironment.shooterForward.z)) / 3f;
        //float reward = 1 / (Vector3.Angle(newEnvironment.shooterForward, idealRotation) * (1 / Vector3.Distance(newEnvironment.shooterLocation, newEnvironment.enemyLocation))) * (1 / Vector3.Distance(newEnvironment.shooterLocation, newEnvironment.enemyLocation)) * 750;


        if (newForwardDissimilarity < oldForwardDissimilarity)
        {
            reward += 0.1f;
        }
        else
        {
            reward -= 0.1f;
        }

        // If the terminal state has been reached
        if (newEnvironment.enemyHitpoints == 0)
        {
            reward += 10f;
            // You may need to change the final reward
            return new object[] { bestActionIndex, reward, 1 };
        }
        // This is a bit dodgy! The shell has a travel time so the direct action which fired the shell may not be rewarded!
        // It may also be a good idea to reward even more if the action kills the enemy tank
        /*else if (oldHitpoints > newHitpoints)
        {
            //reward += (oldHitpoints - newHitpoints);
            reward += (1 - ((oldHitpoints - newHitpoints) / newEnvironment.enemyMaxHitpoints)) * 10f;
        }*/

        // If the shooter is aiming at the tank, then give a positive reward
        if (newEnvironment.tankIndex >= 0)
        {
            reward += 10f;
            // You may need to change the final reward
            return new object[] { bestActionIndex, reward, 1 };
        }

        if (hitWall == true)
        {
            reward -= 0.1f;
        }
        if (AIName == "0")
        {
            HelperScript.PrintList(parsedResponse);
            Debug.Log(reward + " ; " + value);
        }

        return new object[] { bestActionIndex, reward, 0 };
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
            + " >|< " + currentState.ID.ToString()
            + " >|< " + actionReward[2].ToString();

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

    public IEnumerator TrainA3C(int episodeCount = 1001, int maxEpisodeSteps = 200, int epsilonAnnealInterval = 3, int plotInterval = 50)
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
        target.transform.localPosition = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));

        // Spawn the shooter in a random location within the bounds
        shooter.transform.localPosition = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
        while (Vector3.Distance(shooter.transform.localPosition, target.transform.localPosition) < proximityThreshold)
        {
            shooter.transform.localPosition = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
        }

        shooter.transform.LookAt(target.transform.position);
        shooter.transform.Rotate(Random.Range(-aimVariance, aimVariance), Random.Range(-aimVariance, aimVariance), 0);
        shooter.transform.eulerAngles = new Vector3(shooter.transform.eulerAngles.x, shooter.transform.eulerAngles.y, 0);

        // target.transform.position = new Vector3(environmentTransform.position.x + Random.Range(-9f, 9f), environmentTransform.position.y + 0.2f, environmentTransform.position.z);
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
        Vector3 newIdealRotation = (Quaternion.LookRotation(HelperScript.GetDirection(newEnvironment.shooterLocation, newEnvironment.enemyLocation), Vector3.up) * Vector3.forward).normalized;
        float newForwardDissimilarity = (Mathf.Abs(newIdealRotation.x - newEnvironment.shooterForward.x) + Mathf.Abs(newIdealRotation.y - newEnvironment.shooterForward.y) + Mathf.Abs(newIdealRotation.z - newEnvironment.shooterForward.z)) / 3f;

        //  float reward = 1 / (Vector3.Angle(newEnvironment.shooterForward, idealRotation) * (1 / Vector3.Distance(newEnvironment.shooterLocation, newEnvironment.enemyLocation))) * (1 / Vector3.Distance(newEnvironment.shooterLocation, newEnvironment.enemyLocation)) * 750;//1 / (Vector3.Angle(newEnvironment.shooterForward, idealRotation) * (1 / Vector3.Distance(newEnvironment.shooterLocation, newEnvironment.enemyLocation))) * (1 / Vector3.Distance(newEnvironment.shooterLocation, newEnvironment.enemyLocation));

        Debug.Log(newEnvironment.idealForward + " ; " + newEnvironment.shooterForward);
    }
}

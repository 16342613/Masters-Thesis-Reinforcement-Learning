using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

enum ServerRequests
{
    // Given an observation, the server returns an action which follows the policy
    PREDICT,
    // Build up the replay buffer and learn the optimal policy
    BUILD_BUFFER,
    // Train the prediction network
    TRAIN,
    // Set the discount factor (gamma)
    SET_DISCOUNT_FACTOR,
    // Update the target neural network on the server
    UPDATE_TARGET_NETWORK,
    // Test connection
    TEST_CONNECTION
}

public class ArmourTrainerAI : MonoBehaviour
{
    public GameObject shooter;
    public GameObject target;
    public GameObject round;
    private TankControllerScript targetScript;

    public Vector3 maximumBounds;
    public Vector3 minimumBounds;
    public float proximityThreshold;
    public float aimVariance = 30;

    public string replayBufferPath;
    private List<EnvironmentState> replayBuffer = new List<EnvironmentState>();
    private CommunicationClient client;

    private List<float> episodicRewards = new List<float>();
    public float epsilon = 1;


    // Start is called before the first frame update
    void Start()
    {
        targetScript = target.GetComponent<TankControllerScript>();

        client = new CommunicationClient("Assets/Debug/Communication Log.txt");
        client.ConnectToServer("DESKTOP-23VITDP", 8000);

        FileHandler.ClearFile("Assets/Debug/AI Log.txt");
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
            ObserveAndTakeAction(ServerRequests.BUILD_BUFFER);
        }

        if (Input.GetKeyDown(KeyCode.X))
        {
            Train();
        }

        if (Input.GetKeyDown(KeyCode.U))
        {
            client.SendMessage(ServerRequests.UPDATE_TARGET_NETWORK.ToString());
        }

        if (Input.GetKeyDown(KeyCode.T))
        {
            string response = client.RequestResponse(ServerRequests.TEST_CONNECTION.ToString() + " >|< Connection test! ");
            Debug.Log(response);
        }

        //Debug.Log(targetScript.GetHitpoints());
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

        while (aimingAtTarget == false)
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
        }
    }

    private void FireRound()
    {
        GameObject firedRound = Instantiate(round, shooter.transform.position, shooter.transform.rotation);
        firedRound.GetComponent<Rigidbody>().velocity = firedRound.transform.forward * (firedRound.GetComponent<ShellScript>().muzzleVelocity / 10f); // should be divided by 5, not 10! This was set to 10 for training
    }

    private EnvironmentState ObserveEnvironment()
    {
        Vector3 targetLocation = target.transform.position;
        Vector3 targetHullAngle = target.transform.Find("Hitbox/Hull").transform.rotation.eulerAngles;
        Vector3 targetTurretAngle = target.transform.Find("Hitbox/Turret").transform.rotation.eulerAngles;
        int targetHitpoints = targetScript.GetHitpoints();

        Vector3 shooterLocation = shooter.transform.position;
        //Vector3 shooterDirection = shooter.transform.forward.normalized;
        float shooterPenetration = round.GetComponent<ShellScript>().penetration;
        Vector3 aimedLocation = Vector3.zero;
        int tankIndex = -1;
        float plateThickness = -1;
        RaycastHit hit;

        if (Physics.Raycast(shooter.transform.position, shooter.transform.forward, out hit, Mathf.Infinity))
        {
            if (hit.collider.gameObject.tag == "Enemy Tank")
            {
                // aimedLocation = hit.collider.transform.InverseTransformPoint(hit.point);
                aimedLocation = hit.point;
                plateThickness = hit.collider.transform.GetComponent<ArmourPlateScript>().armourThickness;
                tankIndex = hit.collider.transform.GetComponentInParent<TankControllerScript>().tankIndex;
            }
            else
            {
                aimedLocation = hit.point;
            }
        }

        return new EnvironmentState(targetLocation, targetHullAngle, targetTurretAngle, targetHitpoints, tankIndex, shooterLocation, shooterPenetration, aimedLocation, plateThickness);
    }

    private object[] TakeAction(EnvironmentState state)
    {
        // Predict the action using the NN
        string response = client.RequestResponse(ServerRequests.PREDICT.ToString() + " >|< " + state.ToString());
        List<float> parsedResponse = HelperScript.ParseStringToFloats(response, " | ");

        int bestActionIndex = parsedResponse.IndexOf(parsedResponse.Max());
        int oldHitpoints = targetScript.GetHitpoints();

        // Use epsilon-greedy for exploration
        if (Random.Range(0, 1) < epsilon)
        {
            bestActionIndex = Mathf.RoundToInt(Random.Range(0, parsedResponse.Count - 1));
        }

        switch (bestActionIndex)
        {
            // Shoot
            case 0:
                FireRound();
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
                shooter.transform.eulerAngles += new Vector3(1, 0, 0);
                break;

            // Look down
            case 8:
                shooter.transform.eulerAngles += new Vector3(1, 0, 0);
                break;

            // Take no action
            case 9:
                break;
        }

        int newHitpoints = targetScript.GetHitpoints();
        // The default reward
        float reward = -0.1f;

        // If the terminal state has been reached
        if (oldHitpoints == 0)
        {
            // You may need to change the final reward
            return new object[] { bestActionIndex, 100, true };
        }
        // This is a bit dodgy! The shell has a travel time so the direct action which fired the shell may not be rewarded!
        // It may also be a good idea to reward even more if the action kills the enemy tank
        else if (oldHitpoints > newHitpoints)
        {
            reward = (oldHitpoints - newHitpoints) * 0.1f;
        }

        // If the shooter is aiming at the tank, then give a positive reward
        if (ObserveEnvironment().tankIndex >= 0)
        {
            reward = 0.2f;
        }

        return new object[] { bestActionIndex, reward, false };
    }

    private object[] ObserveAndTakeAction(ServerRequests request)
    {
        // Send (Old state, action, reward, new state) to replay buffer

        // Observe the environment
        EnvironmentState currentState = ObserveEnvironment();
        // Query the server to get the best action for that observation
        object[] actionReward = TakeAction(currentState);
        EnvironmentState newState = ObserveEnvironment();

        // Send the state transition data to the server, which puts it into the replay buffer
        string stateTransitionString = currentState.ToString()
            + " >|< " + actionReward[0].ToString()
            + " >|< " + actionReward[1].ToString()
            + " >|< " + newState.ToString();

        // Return the size of the replay buffer in the client
        return new object[] { client.RequestResponse(request.ToString() + " >|< " + stateTransitionString), actionReward[2], actionReward[1] };
    }

    private List<float> PerformEpisode(int maxSteps)
    {
        int steps = 0;
        float episodeReward = 0;

        while (steps < maxSteps)
        {
            object[] stepOutput = ObserveAndTakeAction(ServerRequests.BUILD_BUFFER);
            steps++;

            episodeReward += float.Parse(stepOutput[2].ToString());

            // Reset the environment and leave this episode if the episode has terminated
            if (bool.Parse(stepOutput[1].ToString()) == true) { break; }
        }

        ResetEnvironment();
        return new List<float> { (float) steps, episodeReward};
    }

    public void Train(int episodeCount = 100, int maxEpisodeSteps = 200, int networkUpdateInterval = 10, int epsilonAnnealInterval = 2, float epsilonAnnealAmount = 0.01f)
    {
        // Build up the replay buffer so you can sample transitions
        int bufferPopulation = 0;
        while (bufferPopulation != 500)
        {
            object[] stepOutput = ObserveAndTakeAction(ServerRequests.BUILD_BUFFER);
            bufferPopulation = int.Parse(stepOutput[0].ToString());

            if (bool.Parse(stepOutput[1].ToString()) == true)
            {
                ResetEnvironment();
            }
        }


        for (int i = 0; i < episodeCount; i++)
        {
            // Spawn the shooter in a random location within the bounds
            shooter.transform.position = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
            while (Vector3.Distance(shooter.transform.position, target.transform.position) < proximityThreshold)
            {
                shooter.transform.position = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
            }

            if (i % networkUpdateInterval == 0)
            {
                client.SendMessage(ServerRequests.UPDATE_TARGET_NETWORK.ToString());
            }

            // Perform an episode
            List<float> episodeDetails = PerformEpisode(maxEpisodeSteps);
            FileHandler.WriteToFile("Assets/Debug/AI Log.txt", "Episode : " + i + " ; Steps : " + episodeDetails[0] + " ; Reward : " + episodeDetails[1] + " ; Epsilon : " + epsilon);
            // Debug.Log("Episode : " + i + " ; Steps : " + episodeDetails[0] + " ; Reward : " + episodeDetails[1]);
            client.SendMessage(ServerRequests.TRAIN.ToString());

            // Anneal epsilon over time
            if ((episodeCount % epsilonAnnealInterval == 0) && (epsilon > 0))
            {
                epsilon -= epsilonAnnealAmount;
            }
        }
    }

    private void ResetEnvironment()
    {
        // Spawn the shooter in a random location within the bounds
        shooter.transform.position = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
        while (Vector3.Distance(shooter.transform.position, target.transform.position) < proximityThreshold)
        {
            shooter.transform.position = new Vector3(Random.Range(minimumBounds.x, maximumBounds.x), Random.Range(minimumBounds.y, maximumBounds.y), Random.Range(minimumBounds.z, maximumBounds.z));
        }

        targetScript.ResetHitpoint();
    }
}

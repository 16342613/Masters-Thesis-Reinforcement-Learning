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


    // Start is called before the first frame update
    void Start()
    {
        targetScript = target.GetComponent<TankControllerScript>();

        client = new CommunicationClient("Assets/Debug/Communication Log.txt");
        client.ConnectToServer("DESKTOP-23VITDP", 8000);
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
            ObserveAndTakeAction();
        }

        if (Input.GetKeyDown(KeyCode.X))
        {
            Train();
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
        string aimedPlate = "EMPTY";
        Vector3 aimedLocation = Vector3.zero;
        float plateThickness = -1;
        RaycastHit hit;

        if (Physics.Raycast(shooter.transform.position, shooter.transform.forward, out hit, Mathf.Infinity))
        {
            aimedPlate = hit.collider.gameObject.name;
            aimedLocation = hit.collider.transform.InverseTransformPoint(hit.point);
            plateThickness = hit.collider.transform.GetComponent<ArmourPlateScript>().armourThickness;
        }

        return new EnvironmentState(targetLocation, targetHullAngle, targetTurretAngle, targetHitpoints, shooterLocation, shooterPenetration, aimedPlate, aimedLocation, plateThickness);
    }

    private object[] TakeAction(EnvironmentState state)
    {
        // Predict the action using the NN
        string response = client.RequestResponse(ServerRequests.PREDICT.ToString() + " >|< " + state.ToString());
        List<float> parsedResponse = HelperScript.ParseStringToFloats(response, " | ");

        int bestActionIndex = parsedResponse.IndexOf(parsedResponse.Max());
        int oldHitpoints = targetScript.GetHitpoints();

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
        }

        int newHitpoints = targetScript.GetHitpoints();
        float reward = -0.1f;

        // This is a bit dodgy! The shell has a travel time so the direct action which fired the shell may not be rewarded!
        if (oldHitpoints > newHitpoints)
        {
            reward = (oldHitpoints - newHitpoints) * 0.1f;
        }

        return new object[] { bestActionIndex, reward };
    }

    private int ObserveAndTakeAction()
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
        return int.Parse(client.RequestResponse(ServerRequests.BUILD_BUFFER.ToString() + " >|< " + stateTransitionString));
    }

    private int PerformEpisode(int maxSteps = 100)
    {
        int steps = 0;

        while ((targetScript.GetHitpoints() > 0) || (steps < maxSteps))
        {
            ObserveAndTakeAction();
            steps++;
        }

        return steps;
    }

    public void Train()
    {
        int bufferPopulation = 0;

        System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
        sw.Start();

        while (bufferPopulation != 500)
        {
            bufferPopulation = ObserveAndTakeAction();
        }

        sw.Stop();

        Debug.Log(sw.ElapsedMilliseconds);

        
    }


}

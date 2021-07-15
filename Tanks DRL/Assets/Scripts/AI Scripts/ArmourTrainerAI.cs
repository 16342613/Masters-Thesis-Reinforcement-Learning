using System.Collections;
using System.Collections.Generic;
using UnityEngine;



public class ArmourTrainerAI : MonoBehaviour
{
    public GameObject shooter;
    public GameObject target;
    public GameObject round;

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
        Debug.Log(System.IO.Directory.GetCurrentDirectory());
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

        if (Input.GetKeyDown(KeyCode.X))
        {
            ObserveEnvironment();
        }

        if (Input.GetKeyDown(KeyCode.W))
        {
            string response = client.RequestResponse("HELLO WORLD");
            Debug.Log(response);
        }
    }

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

    private void ObserveEnvironment()
    {
        Vector3 targetLocation = target.transform.position;
        Vector3 targetHullAngle = target.transform.Find("Hitbox/Hull").transform.rotation.eulerAngles;
        Vector3 targetTurretAngle = target.transform.Find("Hitbox/Turret").transform.rotation.eulerAngles;

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

        EnvironmentState state = new EnvironmentState(targetLocation, targetHullAngle, targetTurretAngle, shooterLocation, shooterPenetration, aimedPlate, aimedLocation, plateThickness);

        //FileHandler.WriteToFile(replayBufferPath, state.ToString());
        client.SendMessage(state.ToString());
    }

    private void TakeAction()
    {
        // Take action from NN
    }

    private void SendToReplayBuffer()
    {
        // Send (Old state, action, reward, new state) to replay buffer
    }
}

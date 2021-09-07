using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class HeavyTankWCScript : TankControllerScript
{
    private List<WheelCollider> wheelColliders = new List<WheelCollider>();
    private List<GameObject> wheels = new List<GameObject>();
    public List<int> steeringWheelIndexes = new List<int>();
    public List<int> inverseSteeringWheelIndexes = new List<int>();
    public int maxSteeringAngle = 15;
    public int brakingPower = 100;

    public float gunDepression = -8;
    public float gunElevation = 25;
    public List<GameObject> shellTypes = new List<GameObject>();
    public float scrollZoomFactor = 2.5f;

    public HeavyTankWCScript() : base()
    {
        // Empty Constructor
    }

    // Start is called before the first frame update
    public override void Start()
    {
        base.Start();   // Call the Start method of the superclass

        // Set up the wheels of the light tank
        wheelColliders = transform.Find("Objects/Wheel Colliders").GetComponentsInChildren<WheelCollider>().ToList();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        // Handle the inputs if this is a player
        if (isPlayer == true)
        {
            base.HandleMouseInput(2f, gunDepression, gunElevation, scrollZoomFactor);
            HandleMovementInput();

            if (Input.GetMouseButtonDown(0))
            {
                if (base.reloaded == true)
                {
                    StartCoroutine(FireGunWithReload());
                }
            }
        }
    }

    private IEnumerator FireGunWithReload()
    {
        base.FireGun(shellTypes[0]);
        base.reloaded = false;

        yield return new WaitForSeconds(5);

        base.reloaded = true;
    }

    /// <summary>
    /// This method handles the keyboard input.
    /// </summary>
    private void HandleMovementInput()
    {
        if (Input.GetKey(KeyCode.W))
        {
            for (int i = 0; i < wheelColliders.Count; i++)
            {
                wheelColliders[i].motorTorque = 50;
                wheelColliders[i].brakeTorque = 0;
            }
        }
        else if (Input.GetKey(KeyCode.S))
        {
            for (int i = 0; i < wheelColliders.Count; i++)
            {
                wheelColliders[i].motorTorque = -30;
                wheelColliders[i].brakeTorque = 0;
            }
        }
        else if (Input.GetKey(KeyCode.W) == false && Input.GetKey(KeyCode.S) == false)
        {
            for (int i = 0; i < wheelColliders.Count; i++)
            {
                wheelColliders[i].motorTorque = 0;
                wheelColliders[i].brakeTorque = brakingPower;
            }
        }

        float steeringInput = maxSteeringAngle * Input.GetAxis("Horizontal");
        for (int i = 0; i < steeringWheelIndexes.Count; i++)
        {
            wheelColliders[steeringWheelIndexes[i]].steerAngle = steeringInput;
        }

        for (int i = 0; i < inverseSteeringWheelIndexes.Count; i++)
        {
            wheelColliders[inverseSteeringWheelIndexes[i]].steerAngle = -steeringInput;
        }
    }

    public override void HandleAIMovement(int inputCommand)
    {
        throw new System.NotImplementedException();
    }
}

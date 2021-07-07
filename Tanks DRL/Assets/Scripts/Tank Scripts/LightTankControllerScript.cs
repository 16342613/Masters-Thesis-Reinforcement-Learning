using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class LightTankControllerScript : TankControllerScript
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

    public LightTankControllerScript() : base()
    {
        // Empty Constructor
    }

    // Start is called before the first frame update
    public override void Start()
    {
        base.Start();   // Call the Start method of the superclass

        // Set up the wheels of the light tank
        wheelColliders = transform.Find("Wheel Colliders").GetComponentsInChildren<WheelCollider>().ToList();
        GameObject wheelsParent = transform.Find("Wheels").gameObject;
        foreach (Transform t in wheelsParent.transform)
        {
            wheels.Add(t.gameObject);
        }
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        // Handle the inputs if this is a player
        if (isPlayer == true)
        {
            base.HandleMouseInput(2f, gunDepression, gunElevation, scrollZoomFactor);
            HandleMovementInput();

            if(Input.GetKey(KeyCode.X))
            {
                base.FireGun(shellTypes[0]);
            }
        }

        UpdateWheelPositions();
    }

    /// <summary>
    /// This method rotates the wheels with the wheel collider.
    /// </summary>
    private void UpdateWheelPositions()
    {
        for (int i = 0; i < wheels.Count; i++)
        {
            Vector3 wheelColliderPosition = Vector3.zero;
            Quaternion wheelColliderRotation = Quaternion.identity;

            wheelColliders[i].GetWorldPose(out wheelColliderPosition, out wheelColliderRotation);
            wheels[i].transform.position = wheelColliderPosition;
            wheels[i].transform.rotation = wheelColliderRotation;
        }
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
                wheelColliders[i].motorTorque = 30;
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
}
